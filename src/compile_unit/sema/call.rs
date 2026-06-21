use std::{
	borrow::Cow,
	time::Duration,
};

use bitvec::vec::BitVec;
use internment::Intern;
use rustc_hash::FxHashMap;

use crate::{
	common::{
		Span,
		diagnostic::{
			DiagSpan,
			Diagnostic,
			Label,
		},
	},
	compile_unit::{
		Decl,
		sema::{
			self,
			AnalyzeError,
			Sema,
		},
	},
	frontend::ast::Inline,
	ir::{
		vtir,
		vuir::{
			self,
			Vuir,
		},
	},
	value,
};

/// Subset of a vuir::Opcode::DeclFn
#[derive(Debug)]
pub(super) struct VuirFnInfo<'a> {
	pub fn_decl_origin_module_vuir: &'a Vuir,
	pub ret_ty: vuir::InstructionId,
	pub body: &'a [vuir::InstructionId],
	pub params: &'a [vuir::InstructionId],
	pub span: Span,
	pub builtin: Option<vuir::BuiltinKind>,
}

impl<'a> VuirFnInfo<'a> {
	fn param_display_name(
		&self,
		param_idx: usize,
	) -> Cow<'a, str> {
		let vuir::Opcode::DeclFnParam { name, .. } = &self.fn_decl_origin_module_vuir.instructions[self.params[param_idx]] else {
			unreachable!("expected DeclFnParam");
		};

		Cow::Borrowed(name.symbol.as_ref())
	}

	fn param_idx_by_name(
		&self,
		wanted_name: &str,
	) -> Option<usize> {
		self.params
			.iter()
			.enumerate()
			.find(|(_, p)| match self.fn_decl_origin_module_vuir.instructions[*p] {
				vuir::Opcode::DeclFnParam { name, .. } => name.symbol.as_ref() == wanted_name,
				_ => unreachable!(),
			})
			.map(|(i, _)| i)
	}
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct AnalyzedCallee {
	pub fun: vtir::InstructionRef,
}

impl<'a> Sema<'a> {
	pub(super) fn get_vuir_fn_info(
		&self,
		func_decl: &value::FnDecl,
	) -> VuirFnInfo<'a> {
		let callee_type_owner = func_decl.owner_decl;
		let callee_namespace = self.cu.decls.with_mut(|decls| decls[callee_type_owner].namespace);
		let fn_decl_origin_module = self.cu.modules.with(|modules| modules[func_decl.func_decl_inst.module].clone());

		// SAFETY: we transmute lifetime to circumvent borrow checker limitations, we only use VuirFnInfo for the duration of the analyze call
		let fn_decl_origin_module_vuir: &'static Vuir = unsafe { std::mem::transmute(fn_decl_origin_module.vuir.get().unwrap()) };

		let vuir::Opcode::DeclFn {
			body,
			params,
			ret_ty,
			span,
			builtin,
			..
		} = &fn_decl_origin_module_vuir.instructions[func_decl.func_decl_inst.inst]
		else {
			unreachable!();
		};

		VuirFnInfo {
			fn_decl_origin_module_vuir,
			ret_ty: *ret_ty,
			body,
			params,
			span: *span,
			builtin: *builtin,
		}
	}

	pub fn analyze_fn_call(
		&mut self,
		call_vuir_id: vuir::InstructionId,
		block: super::BlockId,
		callee: AnalyzedCallee,
		args: &[vuir::FnCallArg],
		expected_ret_ty: &Option<vuir::InstructionRef>,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let fun = callee.fun;
		let static_callee = self.try_resolve_comptime_value(&fun).map(|func_decl_index| {
			let func_decl = self.cu.values.index_to_key(func_decl_index).as_fn_decl();
			let func_vuir_info = self.get_vuir_fn_info(func_decl);
			(func_decl_index, func_decl, func_vuir_info)
		});
		let func_type = if let Some((_, func_decl, _)) = &static_callee {
			self.cu.values.index_to_key(func_decl.ty).as_type_fn()
		} else {
			let fn_ty_idx = self.type_of(&fun);
			let value::Key::Type(value::Type::Fn(_)) = self.cu.values.index_to_key(fn_ty_idx) else {
				self.push_error(
					Diagnostic::error()
						.with_message("callee is not a function")
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			};
			self.cu.values.index_to_key(fn_ty_idx).as_type_fn()
		};
		let param_display_name = |idx: usize| -> Cow<'_, str> {
			if let Some((_, _, func_vuir_info)) = &static_callee {
				func_vuir_info.param_display_name(idx)
			} else {
				Cow::Owned(format!("#{}", idx + 1))
			}
		};

		let mut resolved_args = vec![None; func_type.params.len()];
		let mut variadic_args: Vec<(vtir::InstructionRef, value::Index, Span)> =
			Vec::with_capacity(args.len().saturating_sub(func_type.params.len()));
		let mut param_map: FxHashMap<vuir::InstructionId, vtir::InstructionRef> = FxHashMap::default();

		// setup param_map: per-param poison for comptime type params
		if let Some((_, _, func_vuir_info)) = &static_callee {
			for param in func_vuir_info.params {
				let vuir::Opcode::DeclFnParam { comptime, name, .. } = &func_vuir_info.fn_decl_origin_module_vuir.instructions[*param]
				else {
					unreachable!("expected DeclFnParam");
				};
			}
		}

		// we need a dedicated blocks to resolve parameters and return type for generics that is comptime only
		let generic_block = static_callee.as_ref().map(|_| {
			let generic_block = self.child_block(block);
			self.blocks[*generic_block].comptime = true;
			generic_block
		});

		// check the arg count match parameter count and build input args
		let input_args = {
			let mut input_args = Vec::with_capacity(args.len());
			let mut valid_params = BitVec::<u8>::repeat(false, func_type.params.len());
			for (i, arg) in args.iter().enumerate() {
				let param_idx = match arg.name {
					Some(name) => static_callee
						.as_ref()
						.and_then(|(_, _, func_vuir_info)| func_vuir_info.param_idx_by_name(&name)),
					None => (i < func_type.params.len()).then_some(i),
				};

				let param_idx = match param_idx {
					Some(idx) => {
						if valid_params[idx] {
							self.push_error(
								Diagnostic::error()
									.with_message(format!("argument `{}` was provided more than once", param_display_name(idx)))
									.with_label(Label::primary().with_span(self.diag_span(arg.span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}
						valid_params.set(idx, true);
						Some(idx)
					},
					None if arg.name.is_some() => {
						let message = if static_callee.is_some() {
							format!("no parameter named `{}` in function", arg.name.unwrap())
						} else {
							format!("named argument `{}` requires a statically known callee", arg.name.unwrap())
						};
						self.push_error(
							Diagnostic::error()
								.with_message(message)
								.with_label(Label::primary().with_span(self.diag_span(arg.span))),
						);
						return Err(AnalyzeError::AnalysisFailed);
					},
					None if func_type.var_args => None,
					None => {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"function takes {} parameters but more were supplied",
									func_type.params.len()
								))
								.with_label(Label::primary().with_span(self.diag_span(arg.span))),
						);
						return Err(AnalyzeError::AnalysisFailed);
					},
				};

				input_args.push((param_idx, arg));
			}

			let mut missing_params = false;
			for param_idx in valid_params.iter_zeros() {
				self.push_error(
					Diagnostic::error()
						.with_message(format!("missing argument `{}`", param_display_name(param_idx)))
						.with_label(Label::primary().with_span(self.diag_span(*span)))
						.with_note("use `_` to explicitly infer this argument"),
				);
				missing_params = true;
			}
			if missing_params {
				return Err(AnalyzeError::AnalysisFailed);
			}

			input_args
		};

		'arg_loop: for (param_idx, arg) in input_args {
			let arg_idx = match param_idx {
				Some(idx) => idx,
				None => {
					// C varargs are promoted below; keep vuir contextual coercions as noops here
					self.vuir_map.insert(call_vuir_id, self.cu.values.common.void_t.into());
					// resolve var arg now
					let arg_inst = self
						.analyze_comptime_block(block, arg.body)?
						.unwrap_or(self.cu.values.common.unreachable_value.into());
					let arg_ty = self.type_of(&arg_inst);
					match self.promote_variadic_arg(block, arg_inst, arg_ty, arg.span) {
						Ok((coerced_arg, coerced_ty)) => {
							variadic_args.push((coerced_arg, coerced_ty, arg.span));
							continue 'arg_loop;
						},
						Err(_) => {
							return Err(AnalyzeError::AnalysisFailed);
						},
					}
				},
			};

			// evaluate parameter type
			let param_ty = 'param_ty: {
				let ty = func_type.params[arg_idx];

				// not generic, type already known
				if ty != self.cu.values.common.generic_poison_t {
					break 'param_ty ty;
				}
				let Some((_, func_decl, func_vuir_info)) = &static_callee else {
					self.push_error(
						Diagnostic::error()
							.with_message("runtime function pointers cannot infer generic parameter types")
							.with_label(Label::primary().with_span(self.diag_span(arg.span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				};
				let generic_block = **generic_block.as_ref().expect("static callee should have a generic block");

				// need to use param_map as the vuir_map for generic substitution
				std::mem::swap(&mut self.vuir_map, &mut param_map);

				// this param is a generic type, our argument may help infer its type
				// to do that we swapped temporary our vuir_map with the param_map, if we depend on a previously analyzed argument to determine our
				// type we'll pick it up,
				//
				// fn a(comptime ty: type, value: ty)
				// 	    ^^^^^^^^^^^^^^^^^  ---------- type will be resolved since ty is present in param_map
				//      |
				//      in param_map
				let ty = self.with_different_vuir(func_vuir_info.fn_decl_origin_module_vuir, func_decl.func_decl_inst.module, |sema| {
					let vuir::Opcode::DeclFnParam { type_body, .. } = &sema.vuir.instructions[func_vuir_info.params[arg_idx]] else {
						unreachable!("expected DeclFnParam");
					};

					let ty = sema
						.analyze_comptime_block(generic_block, type_body)?
						.unwrap_or(sema.cu.values.common.unreachable_value.into());
					Ok(ty.as_interned())
				});

				std::mem::swap(&mut self.vuir_map, &mut param_map);

				break 'param_ty ty?;
			};

			// does the function ends up with a never param ? if that's the case, stop analysis right now
			if param_ty == self.cu.values.common.never_t {
				return Ok(self.cu.values.common.unreachable_value.into());
			}

			// in from_ast we use the call instruction as the coerce dst type, map it
			self.vuir_map.insert(call_vuir_id, param_ty.into());

			// analyze arg and coerce it to param_ty
			let arg_inst = self
				.analyze_comptime_block(block, arg.body)?
				.unwrap_or(self.cu.values.common.unreachable_value.into());
			let arg_ty = self.type_of(&arg_inst);
			let arg_inst = {
				let coerce_block = generic_block.as_ref().map(|b| **b).unwrap_or(block);
				self.coerce(coerce_block, param_ty, arg_inst, &arg.span)?
			};

			resolved_args[arg_idx] = Some((arg_inst, arg.span));

			// this is the right moment to also consume the arg if it's a linear value
			self.try_consume_linear_value(arg_inst, &arg.span)?;

			// add to arg_map, if the type is comptime we can directly add the inst
			// else we add a dummy alloc instruction that serve no purpose other than holding the arg_ty
			let arg_ty = self.type_of(&arg_inst);
			let param_is_comptime = if let Some((_, _, func_vuir_info)) = &static_callee {
				let vuir::Opcode::DeclFnParam { comptime, .. } =
					&func_vuir_info.fn_decl_origin_module_vuir.instructions[func_vuir_info.params[arg_idx]]
				else {
					unreachable!("expected DeclFnParam");
				};
				*comptime
			} else {
				func_type.comptime_params[arg_idx]
			};
			if param_is_comptime || self.cu.values.type_is_comptime_only(arg_ty) {
				if let Some((_, _, func_vuir_info)) = &static_callee {
					param_map.insert(func_vuir_info.params[arg_idx], arg_inst);
				}
			} else {
				// add a instruction outside of any block so we can retrieve the arg_ty as is it is used
				let arg_inst = self.instructions.push(vtir::Opcode::StackAlloc { ty: arg_ty });
				if let Some((_, _, func_vuir_info)) = &static_callee {
					param_map.insert(func_vuir_info.params[arg_idx], arg_inst.into_ref());
				}
			}
		}

		// after args are resolved, do the return type
		let resolved_ret_ty = 'ret_ty: {
			let Some((_, func_decl, func_vuir_info)) = &static_callee else {
				if func_type.ret_ty != self.cu.values.common.generic_poison_t {
					break 'ret_ty func_type.ret_ty;
				}
				self.push_error(
					Diagnostic::error()
						.with_message("runtime function pointers cannot infer return type")
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			};
			let generic_block = **generic_block.as_ref().expect("static callee should have a generic block");

			// not generic, type already known
			if func_type.ret_ty != self.cu.values.common.generic_poison_t {
				break 'ret_ty func_type.ret_ty;
			}

			// like fn params, analyze the return type now that we have a filled param_map
			std::mem::swap(&mut self.vuir_map, &mut param_map);
			let ty = self.with_different_vuir(func_vuir_info.fn_decl_origin_module_vuir, func_decl.func_decl_inst.module, |sema| {
				let vuir::Opcode::BlockComptime { instructions, .. } = &sema.vuir.instructions[func_vuir_info.ret_ty] else {
					unreachable!("expected BlockComptime");
				};

				let ty = sema
					.analyze_comptime_block(generic_block, instructions)?
					.unwrap_or(sema.cu.values.common.unreachable_value.into());
				Ok(ty.as_interned())
			});
			std::mem::swap(&mut self.vuir_map, &mut param_map);

			let ty = ty?;

			// if the return type still contains poison, try to infer from expected_ret_ty
			let ty = if ty == self.cu.values.common.generic_poison_t {
				if let Some(expected_ret_ty) = expected_ret_ty {
					let expected_ret_ty = self.resolve_type(block, expected_ret_ty, span)?;
					self.coerce(block, expected_ret_ty, vtir::InstructionRef::Interned(ty), span)?
						.as_interned()
				} else {
					// no expected type.. we can't infer the type
					ty
				}
			} else if let Some(expected_ret_ty) = expected_ret_ty {
				// ret ty fully resolved, just coerce to expected type
				let expected_ret_ty = self.resolve_type(block, expected_ret_ty, span)?;
				if ty != expected_ret_ty {
					self.coerce(block, expected_ret_ty, vtir::InstructionRef::Interned(ty), span)?
						.as_interned()
				} else {
					// expected_ret_ty match ty
					ty
				}
			} else {
				// no poison
				ty
			};

			// type still poisoned, we can't infer it
			if ty == self.cu.values.common.generic_poison_t {
				self.push_error(
					Diagnostic::error()
						.with_message(format!("cannot infer return type `{}`", self.cu.values.display_index(ty)))
						.with_label(Label::primary().with_span(self.diag_span(*span)))
						.with_label(
							Label::secondary()
								.with_message("generic return type declared here")
								.with_span(DiagSpan {
									module: func_decl.func_decl_inst.module,
									span: func_vuir_info.span,
								}),
						)
						.with_note(
							"some generic parameters could not be inferred from arguments or return type context; try adding explicit \
							 type annotations",
						),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}

			break 'ret_ty ty;
		};

		// we finished with analyses, unstack the generic block
		if let Some(generic_block) = generic_block {
			let generic_block_id = *generic_block;
			// SAFETY: they cannot ever be the same id
			let [generic_block_body, call_block] = unsafe { self.blocks.get_disjoint_unchecked_mut([generic_block_id, block]) };
			call_block.instructions.append(&mut generic_block_body.instructions);
			self.unstack_block(generic_block);
		}

		// ensure args are all resolved, the only reason why a arg couldn't is if inference failed
		let mut inference_failed = false;
		for (param_idx, arg) in resolved_args.iter().enumerate() {
			if arg.is_some() {
				continue;
			}
			self.push_error(
				Diagnostic::error()
					.with_message(format!("parameter `{}` cannot be inferred", param_display_name(param_idx)))
					.with_label(Label::primary().with_span(self.diag_span(*span)))
					.with_note("provide the argument explicitly"),
			);
			inference_failed = true;
		}
		if inference_failed {
			return Err(AnalyzeError::AnalysisFailed);
		}

		// now, build the new function type, the function value and fire up analysis
		let mut analysis_failed = false;
		let (mut runtime_args, comptime_args): (Vec<vtir::InstructionRef>, Vec<Option<value::Index>>) = {
			let mut runtime_args = Vec::with_capacity(resolved_args.len());
			let mut comptime_args = Vec::with_capacity(resolved_args.len());

			for (i, (arg, arg_span)) in resolved_args.iter().enumerate().map(|(i, opt)| (i, opt.unwrap())) {
				let name = param_display_name(i);
				if func_type.comptime_params[i] {
					if static_callee.is_none() {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"argument `{}` cannot bind a comptime parameter through a function pointer",
									name
								))
								.with_label(Label::primary().with_span(self.diag_span(arg_span))),
						);
						analysis_failed = true;
						continue;
					}
					if let Some(value) = self.try_resolve_comptime_value(&arg) {
						comptime_args.push(Some(value));
					} else {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"argument `{}` is marked as comptime and requires a compile-time known value to be provided",
									name
								))
								.with_label(Label::primary().with_span(self.diag_span(arg_span))),
						);
						analysis_failed = true;
					}
				} else {
					let arg_ty = self.type_of(&arg);

					// ensure we aren't being passed a comptime-only type
					if self.cu.values.type_is_comptime_only(arg_ty) {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"argument `{}` is a runtime arg but has been passed comptime-only type `{}`",
									name,
									self.cu.values.display_index(arg_ty),
								))
								.with_label(Label::primary().with_span(self.diag_span(arg_span)))
								.with_note("add explicit type annotations"),
						);
						analysis_failed = true;
					}

					runtime_args.push(arg);

					// comptime args is not dense
					comptime_args.push(None);

					// ensure value is comptime known if in a comptime fn call
					if self.blocks[block].comptime && self.try_resolve_comptime_value(&arg).is_none() {
						self.push_error(
							Diagnostic::error()
								.with_message(format!("argument `{}` must be a compile-time known value", name))
								.with_label(Label::primary().with_span(self.diag_span(arg_span)))
								.with_label(
									Label::secondary()
										.with_message("due to compile-time function call")
										.with_span(self.diag_span(*span)),
								),
						);
						analysis_failed = true;
					}
				}
			}

			// finally append varargs
			for (inst, ..) in variadic_args {
				runtime_args.push(inst)
			}

			(runtime_args, comptime_args)
		};

		if analysis_failed {
			return Err(AnalyzeError::AnalysisFailed);
		}

		// all args are valid etc
		let mut resolved_args_types = resolved_args
			.iter()
			.enumerate()
			.map(|(i, arg)| {
				let arg = arg.expect("argument must be resolved at this point");
				Ok(self.type_of(&arg.0))
			})
			.try_collect::<Vec<_>>()?;
		let instantiated_fn_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Fn(value::TypeFn {
			params: self.cu.values.alloc_slice(&resolved_args_types),
			comptime_params: func_type.comptime_params,
			var_args: func_type.var_args,
			ret_ty: resolved_ret_ty,
			external: func_type.external,
			callconv: func_type.callconv,
			inline: func_type.inline,
		})));

		// at this point we know everything about the function, only one thing remains:
		// force comptime if the return type is comptime-only (e.g. a function returning `type`).
		let call_is_comptime = self.blocks[block].comptime || self.cu.values.type_is_comptime_only(resolved_ret_ty);
		let call_is_inline = static_callee.is_some() && (call_is_comptime || func_type.inline == Inline::Always);

		let func_val = static_callee.as_ref().map(|(func_decl_index, func_decl, _)| {
			let comptime_args_static = self.cu.values.alloc_slice(&comptime_args);
			// need to create decl for this fn
			let owner_decl = self.cu.decls.with_mut(|decls| {
				let base_decl = &decls[func_decl.owner_decl];
				let full_qualified_name = format!("{}.{}", base_decl.full_qualified_name, instantiated_fn_ty.as_u32());
				decls.push(Decl {
					name: base_decl.name,
					full_qualified_name: full_qualified_name.into(),
					module: base_decl.module,
					namespace: base_decl.namespace,
					analysis_state: crate::compile_unit::DeclAnalysisState::TypeKnown(instantiated_fn_ty),
				})
			});
			let fn_key = value::Key::Fn(value::FnKey {
				ty: instantiated_fn_ty,
				decl: *func_decl_index,
				comptime_args: comptime_args_static,
				owner_decl,
			});
			self.cu.values.intern_non_trivial(&fn_key, value::Value::none())
		});

		if call_is_inline {
			let func_val = func_val.expect("inline calls require a statically known callee");
			let (_, func_decl, func_vuir_info) = static_callee.as_ref().expect("inline calls require a statically known callee");
			let instantiated_fn = self.cu.values.index_to_key(func_val).as_fn();
			let instantiated_fn_ty = self.cu.values.index_to_key(instantiated_fn_ty).as_type_fn();

			let caller_diag_span = self.diag_span(*span);
			let saved_fun = self.fun.take();

			// swap param_map into vuir_map,  it already has all comptime params mapped.
			// patch runtime params: param_map has StackAlloc proxies, we need actual arg values.
			let old_vuir_map = std::mem::replace(&mut self.vuir_map, param_map);
			let old_linear_slots = std::mem::take(&mut self.linear_slots);

			for (i, param_id) in func_vuir_info.params.iter().enumerate() {
				if !func_type.comptime_params[i]
					&& let Some((arg_val, _)) = resolved_args[i]
				{
					self.vuir_map.insert(*param_id, arg_val);
				}
			}

			// create a child block for inline evaluation; comptime if needed
			let inline_block = {
				let (base_type_name, namespace) = self.cu.decls.with_mut(|decls| {
					let decl = &decls[instantiated_fn.owner_decl];
					(decl.name, decl.namespace)
				});
				self.blocks.push(sema::Block {
					namespace,
					parent: Some(block),
					instructions: bumpalo::collections::Vec::new_in(self.instructions_payload_alloc),
					vuir_block: None,
					comptime: call_is_comptime,
					inlined: true,
					base_type_name,
					decl_fn_params: Default::default(),
					capture_context: self.blocks[block].capture_context.clone(),
				})
			};

			let result = self.with_different_vuir(
				func_vuir_info.fn_decl_origin_module_vuir,
				func_decl.func_decl_inst.module,
				|sema| -> Result<vtir::InstructionRef, AnalyzeError> {
					let value = if let Some(builtin) = func_vuir_info.builtin {
						Some(sema.analyze_fn_builtin_body(inline_block, instantiated_fn, builtin, caller_diag_span)?)
					} else {
						sema.fun = Some(func_val);
						sema.analyze_fn_body_at_comptime(inline_block, func_vuir_info.body, instantiated_fn_ty.ret_ty, caller_diag_span)?
					};

					let value = value.unwrap_or(sema.cu.values.common.void_value.into());
					Ok(value)
				},
			)?;

			// append inline call insts to parent block and pop
			{
				// SAFETY: they cannot ever be the same id
				let [inline_block, call_blcok] = unsafe { self.blocks.get_disjoint_unchecked_mut([inline_block, block]) };
				call_blcok.instructions.append(&mut inline_block.instructions);
			}
			self.blocks.pop();

			// restore the caller's vuir_map, discarding all callee VUIR entries
			self.vuir_map = old_vuir_map;
			self.linear_slots = old_linear_slots;
			self.fun = saved_fun;
			self.vuir_map.insert(call_vuir_id, result);
			Ok(result)
		} else {
			if static_callee.is_none() && call_is_comptime {
				self.push_error(
					Diagnostic::error()
						.with_message("compile-time calls require a statically known callee")
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}
			self.ensure_type_exist_in_runtime(resolved_ret_ty, span)?;
			if let Some(func_val) = func_val {
				self.cu.queue_runtime_function_analysis_if_needed(func_val);
			}

			let args = self.instructions_payload_alloc.alloc_slice_fill_iter(runtime_args);
			let inst = self.inst(block, vtir::Opcode::FnCall {
				callee: func_val.map(vtir::InstructionRef::Interned).unwrap_or(fun),
				args,
			});
			self.vuir_map.insert(call_vuir_id, inst);

			if resolved_ret_ty == self.cu.values.common.never_t {
				self.inst(block, vtir::Opcode::Unreachable);
				Ok(vtir::InstructionRef::Interned(self.cu.values.common.unreachable_value))
			} else {
				Ok(inst)
			}
		}
	}
}
