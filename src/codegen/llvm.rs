use std::sync::OnceLock;

use bumpalo::{
	Bump,
	collections::FromIteratorIn,
};
use inkwell::{
	AddressSpace,
	attributes::{
		Attribute,
		AttributeLoc,
	},
	basic_block::BasicBlock,
	builder::BuilderError,
	debug_info::{
		AsDIScope,
		DIFlags,
		DIFlagsConstants,
	},
	intrinsics::{
		self,
		Intrinsic,
	},
	llvm_sys::{
		self,
		support::LLVMParseCommandLineOptions,
		target_machine::*,
	},
	memory_buffer::MemoryBuffer,
	module::Linkage,
	passes::PassBuilderOptions,
	targets::{
		FileType,
		InitializationConfig,
		Target,
		TargetMachine,
		TargetTriple,
	},
	types::{
		AnyType,
		AnyTypeEnum,
		AsTypeRef,
		BasicMetadataTypeEnum,
		BasicType,
		BasicTypeEnum,
		StructType,
	},
	values::{
		AnyValue,
		AnyValueEnum,
		AsValueRef,
		BasicMetadataValueEnum,
		BasicValue,
		BasicValueEnum,
		BasicValueUse,
		FunctionValue,
		GlobalValue,
		IntValue,
		PhiValue,
		PointerValue,
		UnnamedAddress,
	},
};
use rustc_hash::FxHashMap;

use crate::{
	Build,
	codegen::{
		abi,
		llvm::{
			self,
		},
	},
	compile_unit::{
		CompilationUnit,
		DeclAnalysisState,
		DeclId,
		TypeInfoId,
		module::ModuleId,
	},
	frontend::ast,
	ir::{
		id::*,
		vtir::{
			self,
			Vtir,
		},
		vuir::{
			self,
			Vuir,
		},
	},
	value::{
		self,
		CallingConvention,
		ValueStore,
	},
};

struct DebugInfoGen<'ctx> {
	di_ctx: DebugInfoCtx<'ctx>,
	di_file: inkwell::debug_info::DIFile<'ctx>,
	di_lexical_block_stack: Vec<inkwell::debug_info::DILexicalBlock<'ctx>>,
}

#[derive(Debug)]
struct LlvmAttributes {
	alwaysinline: Attribute,
	noinline: Attribute,
	nounwind: Attribute,
	noreturn: Attribute,
	willreturn: Attribute,
	mustprogress: Attribute,
	uwtable_sync: Attribute,
	sret: Attribute,
	byval: Attribute,
	noalias: Attribute,
	nonnull: Attribute,
}

impl LlvmAttributes {
	fn new(ctx: &inkwell::context::Context) -> Self {
		macro_rules! enum_attr {
			($attr:literal) => {
				ctx.create_enum_attribute(Attribute::get_named_enum_kind_id($attr), 0)
			};

			($attr:literal, $val:literal) => {
				ctx.create_enum_attribute(Attribute::get_named_enum_kind_id($attr), $val)
			};
		}
		Self {
			alwaysinline: enum_attr!("alwaysinline"),
			noinline: enum_attr!("noinline"),
			nounwind: enum_attr!("nounwind"),
			noreturn: enum_attr!("noreturn"),
			willreturn: enum_attr!("willreturn"),
			mustprogress: enum_attr!("mustprogress"),
			uwtable_sync: enum_attr!("uwtable", 1),
			sret: enum_attr!("sret"),
			byval: enum_attr!("byval"),
			noalias: enum_attr!("noalias"),
			nonnull: enum_attr!("nonnull"),
		}
	}
}

struct LlvmIntrins {
	trap: Intrinsic,
}
impl LlvmIntrins {
	fn new(ctx: &inkwell::context::Context) -> Self {
		let trap = intrinsics::Intrinsic::find("llvm.trap").unwrap();
		Self { trap }
	}
}

struct FnLowerCtx<'a, 'ctx> {
	compilation_unit: &'a CompilationUnit,
	module: ModuleId,
	lowerer: &'a mut Lowerer<'ctx>,
	vtir_inst_to_llvm_value: FxHashMap<vtir::InstructionId, AnyValueEnum<'ctx>>,
	vtir_block_to_break_list: FxHashMap<vtir::InstructionId, BreakList<'ctx>>,
	// TODO(zino): remove these options once function-lowering state has a cleaner initialization path.
	cur_fn: Option<inkwell::values::FunctionValue<'ctx>>,
	cur_fn_args: Vec<BasicValueEnum<'ctx>>,
	cur_llvm_fn_param_idx: u32,
	cur_fn_ty: value::Index,
	di_gen: Option<DebugInfoGen<'ctx>>,
}

#[derive(Debug)]
struct BreakList<'ctx> {
	body_bb: Option<BasicBlock<'ctx>>,
	after_bb: BasicBlock<'ctx>,
	breaks: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)>,
}

impl<'a, 'ctx> FnLowerCtx<'a, 'ctx> {
	fn resolve_inst(
		&mut self,
		inst: vtir::InstructionRef,
	) -> AnyValueEnum<'ctx> {
		match inst {
			vtir::InstructionRef::Instruction(id) => self
				.vtir_inst_to_llvm_value
				.get(&id)
				.copied()
				.unwrap_or_else(|| panic!("{id} should be lowered")),
			vtir::InstructionRef::Interned(val) => self.lowerer.lower_interned_value(val),
		}
	}

	fn builder(&self) -> &inkwell::builder::Builder<'ctx> {
		&self.lowerer.builder
	}

	/// Prefer this over builder().build_alloca to put alloca at top of basic block
	fn build_alloca_at_top_of_bb<T>(
		&mut self,
		pointee_ty: T,
		name: &str,
	) -> Result<PointerValue<'ctx>, BuilderError>
	where
		T: BasicType<'ctx>,
	{
		// LLVM recommends to put the alloca instruction at the beginning of the function (the entry bb)
		// This is because the SROA and Mem2Reg passes only optimize alloca inside the entry basic block of the fn
		// So any branching, blocks, inline calls will cause allocas outside the entry block to not be traversed by those passes
		// which can have a great impact for the optimizer (in compile time and execution time)
		let prev_block = self.builder().get_insert_block().unwrap();
		let fn_entry_block = self.cur_fn.unwrap().get_first_basic_block().unwrap();

		// position at the beginning of the entry block not at the end since we may have a terminator (branch, ...) at the end
		match fn_entry_block.get_first_instruction().as_ref() {
			Some(first_instr) => self.builder().position_before(first_instr),
			None => self.builder().position_at_end(fn_entry_block),
		}

		#[expect(clippy::disallowed_methods)]
		let alloca = self.builder().build_alloca(pointee_ty, name)?;
		self.builder().position_at_end(prev_block);
		Ok(alloca)
	}

	/// Prefer this over inkwell build_load as this handle by_ref types properly
	fn build_load(
		&mut self,
		pointee_ty: value::Index,
		ptr: PointerValue<'ctx>,
	) -> Result<AnyValueEnum<'ctx>, BuilderError> {
		let val = if self.lowerer.vif_abi_type_is_by_ref(pointee_ty) {
			let pointee_ty_llvm: BasicTypeEnum = self.lowerer.lower_type_basic(pointee_ty);
			let alloca = self.build_alloca_at_top_of_bb(pointee_ty_llvm, "load.byref")?;
			let pointee_ty_layout = self
				.compilation_unit
				.values
				.type_layout(&self.compilation_unit.resolved_target, pointee_ty);
			self.builder().build_memcpy(
				alloca,
				pointee_ty_layout.align as u32,
				ptr,
				pointee_ty_layout.align as u32,
				self.lowerer
					.ctx
					.ptr_sized_int_type(&self.lowerer.target_machine.get_target_data(), None)
					.const_int(pointee_ty_layout.size, false),
			);
			alloca.as_any_value_enum()
		} else {
			let llvm_ty: BasicTypeEnum = self.lowerer.lower_type_basic(pointee_ty);

			#[expect(clippy::disallowed_methods)]
			self.builder().build_load(llvm_ty, ptr, "")?.as_any_value_enum()
		};
		Ok(val)
	}

	/// Prefer this over inkwell build_store as this handle by_ref types properly
	fn build_store(
		&mut self,
		elem_ty: value::Index,
		elem_ptr: PointerValue<'ctx>,
		value: AnyValueEnum<'ctx>,
	) -> Result<(), BuilderError> {
		if self.lowerer.vif_abi_type_is_by_ref(elem_ty) {
			let elem_ty_layout = self
				.compilation_unit
				.values
				.type_layout(&self.compilation_unit.resolved_target, elem_ty);
			let elem_ty: BasicTypeEnum = self.lowerer.lower_type_basic(elem_ty);
			self.builder().build_memcpy(
				elem_ptr,
				elem_ty_layout.align as u32,
				value.into_pointer_value(),
				elem_ty_layout.align as u32,
				self.lowerer
					.ctx
					.ptr_sized_int_type(&self.lowerer.target_machine.get_target_data(), None)
					.const_int(elem_ty_layout.size, false),
			)?;
		} else {
			let element = match value {
				AnyValueEnum::FunctionValue(fun) => fun.as_global_value().as_pointer_value().as_basic_value_enum(),
				value => value.try_into().unwrap(),
			};

			#[expect(clippy::disallowed_methods)]
			self.builder().build_store(elem_ptr, element)?;
		}

		Ok(())
	}

	fn materialize_nominal_value(
		&mut self,
		ty: value::Index,
		value: BasicValueEnum<'ctx>,
		name: &str,
	) -> Result<AnyValueEnum<'ctx>, BuilderError> {
		if self.lowerer.vif_abi_type_is_by_ref(ty) {
			let ty: BasicTypeEnum = self.lowerer.lower_type_basic(ty);
			let storage = self.build_alloca_at_top_of_bb(ty, "materialize")?;
			#[allow(clippy::disallowed_methods)]
			self.builder().build_store(storage, value)?;
			Ok(storage.as_any_value_enum())
		} else {
			Ok(value.as_any_value_enum())
		}
	}

	#[allow(
		clippy::disallowed_methods,
		reason = "ABI adaptation requires loads and stores with explicit LLVM types"
	)]
	fn lower_body_inst(
		&mut self,
		vtir: &Vtir,
		parent_bb: BasicBlock<'ctx>,
		id: vtir::InstructionId,
		inst: &vtir::Opcode,
	) -> Result<(), BuilderError> {
		let ctx = self.lowerer.ctx;

		match inst {
			vtir::Opcode::Invalid => unreachable!(),
			vtir::Opcode::Noop => {},
			vtir::Opcode::Block { instructions, ret_ty } => {
				if let Some(phi) = self.lower_block(vtir, parent_bb, id, instructions, *ret_ty)? {
					self.vtir_inst_to_llvm_value.insert(id, phi.as_any_value_enum());
				}
			},
			vtir::Opcode::Loop { instructions, ret_ty } => {
				let loop_bb = ctx.insert_basic_block_after(parent_bb, "loop");
				self.builder().build_unconditional_branch(loop_bb)?;

				let after_loop_bb = ctx.insert_basic_block_after(parent_bb, "after_loop");
				self.vtir_block_to_break_list.insert(id, BreakList {
					body_bb: Some(loop_bb),
					after_bb: after_loop_bb,
					breaks: Vec::new(),
				});
				self.builder().position_at_end(loop_bb);

				self.lower_body(vtir, loop_bb, instructions);

				self.builder().position_at_end(after_loop_bb);

				if *ret_ty != self.compilation_unit.values.common.void_t {
					let ret_ty = if self.lowerer.vif_abi_type_is_by_ref(*ret_ty) {
						self.lowerer.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum()
					} else {
						self.lowerer.lower_type_basic(*ret_ty)
					};
					let break_list = self.vtir_block_to_break_list.get(&id).unwrap();
					if !break_list.breaks.is_empty() {
						let phi = self.builder().build_phi(ret_ty, "")?;
						for (break_value, break_bb) in &break_list.breaks {
							phi.add_incoming(&[(break_value, *break_bb)]);
						}
						self.vtir_inst_to_llvm_value.insert(id, phi.as_any_value_enum());
					}
				}
			},
			vtir::Opcode::Repeat { r#loop } => {
				let break_list = self.vtir_block_to_break_list.get(r#loop).unwrap();
				self.builder().build_unconditional_branch(break_list.body_bb.unwrap())?;
			},
			vtir::Opcode::Break { block, value } => {
				let value_ty = vtir::opcodes::type_of(&self.compilation_unit.values, &vtir.instructions, value);
				let is_void = value_ty == self.compilation_unit.values.common.void_t;
				if is_void {
					let break_list = self.vtir_block_to_break_list.get(block).unwrap();
					self.builder().build_unconditional_branch(break_list.after_bb)?;
				} else {
					let value = self.resolve_inst(*value).try_into().unwrap();
					let insert_block = self.builder().get_insert_block().unwrap();
					let break_list = self.vtir_block_to_break_list.get_mut(block).unwrap();
					break_list.breaks.push((value, insert_block));
					// Keep the mutable borrow short before branching.
					self.lowerer.builder.build_unconditional_branch(break_list.after_bb)?;
				}
			},
			vtir::Opcode::StackAlloc { ty: ptr_ty } => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*ptr_ty).as_type_ptr().pointee_ty;
				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type_basic(pointee_ty);
				let alloca = self.build_alloca_at_top_of_bb(pointee_ty, "stackalloc")?;
				self.vtir_inst_to_llvm_value.insert(id, alloca.into());
			},
			vtir::Opcode::Load { ptr } => {
				let pointee_ty = {
					let ptr_ty = vtir.type_of(&self.compilation_unit.values, ptr);
					self.compilation_unit.values.index_to_key(ptr_ty).as_type_ptr().pointee_ty
				};
				let ptr = self.resolve_inst(*ptr).into_pointer_value();
				let val = self.build_load(pointee_ty, ptr)?;
				self.vtir_inst_to_llvm_value.insert(id, val);
			},
			vtir::Opcode::Store { src, dst } => {
				let src_ty = vtir.type_of(&self.compilation_unit.values, src);
				let dst_ptr_ty = vtir.type_of(&self.compilation_unit.values, dst);
				let dst_ptr_ty = self.compilation_unit.values.index_to_key(dst_ptr_ty).as_type_ptr();

				let src: BasicValueEnum = self.resolve_inst(*src).try_into().unwrap();
				let dst = self.resolve_inst(*dst).into_pointer_value();

				// if the ptr points to a bit, properly setup the src value
				let src = if let Some(packed) = &dst_ptr_ty.packed {
					let underlying_int_ty = ctx.custom_width_int_type(packed.underlying_int_bits);
					let _pointee_ty = self.lowerer.lower_type(dst_ptr_ty.pointee_ty);
					let pointee_int_ty = ctx.custom_width_int_type(packed.bit_width);

					#[allow(clippy::disallowed_methods, reason = "we directly use llvm types for build_load")]
					// first load the dst value
					let dst_val = self
						.builder()
						.build_load(underlying_int_ty, dst, "store.packed.load_underlying_int")?
						.into_int_value();

					// compute mask of values to keep
					let preserve_mask = {
						// init mask of bits we'll touch, perform a z_extend to zero exceess bits
						let mask = self
							.builder()
							.build_int_z_extend(pointee_int_ty.const_int(u64::MAX, false), underlying_int_ty, "")?;

						// shift left mask to put it at the right offset
						let mask = self
							.builder()
							.build_left_shift(mask, underlying_int_ty.const_int(packed.bit_offset as _, false), "")?;

						// mask ^ ~0 to get mask of all bits to preserve
						self.builder().build_xor(mask, underlying_int_ty.const_int(u64::MAX, false), "")?
					};

					// compute final value
					let src = {
						// clear unpreserved bits of destination
						let dst_val = self.builder().build_and(dst_val, preserve_mask, "")?;

						// src => equivalent-sized int
						let src = self
							.builder()
							.build_bit_cast(src, pointee_int_ty, "store.packed.bitcast")?
							.into_int_value();

						// z_extend src value to dst type
						let src = self.builder().build_int_z_extend(src, underlying_int_ty, "")?;

						// shift src value to right offset
						let src = self
							.builder()
							.build_left_shift(src, underlying_int_ty.const_int(packed.bit_offset as _, false), "")?;

						// and finally or src & dst
						self.builder().build_or(src, dst_val, "")?
					};

					src.as_basic_value_enum()
				} else {
					src
				};

				self.build_store(dst_ptr_ty.pointee_ty, dst, src.as_any_value_enum())?;
			},
			vtir::Opcode::FnArg { ty, .. } => {
				let fun = self.cur_fn.unwrap();
				let arg: BasicValueEnum = self.cur_fn_args[self.cur_llvm_fn_param_idx as usize];
				self.cur_llvm_fn_param_idx += 1;
				self.vtir_inst_to_llvm_value.insert(id, arg.as_any_value_enum());
			},
			vtir::Opcode::Return { value } => {
				let fn_ty = self.lowerer.compilation_unit.values.index_to_key(self.cur_fn_ty).as_type_fn();
				let ret_repr = self.lowerer.compute_fn_ret_ty_abi_repr(fn_ty);
				if ret_repr == abi::Repr::ByRef {
					let ret_ptr = self.cur_fn.unwrap().get_nth_param(0).unwrap();
					if let Some(value) = value
						&& vtir.type_of(&self.compilation_unit.values, value) != self.compilation_unit.values.common.void_t
					{
						let value_ty = vtir.type_of(&self.compilation_unit.values, value);
						let value = self.resolve_inst(*value);
						self.build_store(value_ty, ret_ptr.into_pointer_value(), value)?;
					}
					self.builder().build_return(None)?;
				} else {
					if let Some(value) = value
						&& vtir.type_of(&self.compilation_unit.values, value) != self.compilation_unit.values.common.void_t
					{
						let value_ty = vtir.type_of(&self.compilation_unit.values, value);
						let value = self.resolve_inst(*value);
						let value: BasicValueEnum = value.try_into().unwrap();

						let value = match ret_repr {
							abi::Repr::ByValue => {
								if self.lowerer.vif_abi_type_is_by_ref(value_ty) {
									let value_ty: BasicTypeEnum = self.lowerer.lower_type_basic(value_ty);
									self.builder()
										.build_load(value_ty, value.into_pointer_value(), "ret.byref.to.byval.load")?
								} else {
									value
								}
							},
							abi::Repr::AsInteger => {
								let layout = self
									.compilation_unit
									.values
									.type_layout(&self.compilation_unit.resolved_target, value_ty);
								let int_ty = self.lowerer.ctx.custom_width_int_type((layout.size * 8) as _);
								if self.lowerer.vif_abi_type_is_by_ref(value_ty) {
									self.builder().build_load(int_ty, value.into_pointer_value(), "ret.asinteger")?
								} else {
									let value_ty_llvm: BasicTypeEnum = self.lowerer.lower_type_basic(value_ty);
									let storage = self.build_alloca_at_top_of_bb(value_ty_llvm, "ret.asinteger.storage")?;
									self.builder().build_store(storage, value)?;
									self.builder().build_load(int_ty, storage, "ret.asinteger")?
								}
							},
							abi::Repr::ByRef => unreachable!(),
						};

						self.builder().build_return(Some(&value))?;
					} else {
						self.builder().build_return(None)?;
					};
				}
			},
			vtir::Opcode::FnCall { callee, args } => {
				let fn_ty_idx = vtir.type_of(&self.compilation_unit.values, callee);
				let fn_ty = self.compilation_unit.values.index_to_key(fn_ty_idx).as_type_fn();
				let callconv = fn_ty.callconv;
				let llvm_fn_ty = self.lowerer.lower_type(fn_ty_idx).into_function_type();
				let callee_param_tys = llvm_fn_ty.get_param_types();
				let sret = self.lowerer.fn_use_sret(fn_ty);

				let (args, ret_ptr) = {
					let mut values = Vec::<BasicMetadataValueEnum>::with_capacity(args.len());

					let ret_ptr = if sret {
						let ret_ty: BasicTypeEnum = self.lowerer.lower_type_basic(fn_ty.ret_ty);
						let ret_ptr = self.build_alloca_at_top_of_bb(ret_ty, "call.sret")?;
						values.push(ret_ptr.into());
						Some(ret_ptr)
					} else {
						None
					};

					let abi_param_offset = usize::from(sret);
					for (i, arg) in args.iter().enumerate() {
						let arg_ty = vtir.type_of(&self.compilation_unit.values, arg);
						let arg_ty_llvm: BasicTypeEnum = self.lowerer.lower_type_basic(arg_ty);
						let arg: BasicValueEnum = self.resolve_inst(*arg).try_into().unwrap();
						// convert from vif abi to fn abi
						let val = match self.lowerer.compute_fn_param_abi_repr(fn_ty, arg_ty) {
							abi::Repr::ByValue => {
								if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
									self.builder().build_load(arg_ty_llvm, arg.into_pointer_value(), "")?
								} else {
									arg
								}
							},
							abi::Repr::ByRef => {
								if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
									arg
								} else {
									let alloca = self.build_alloca_at_top_of_bb(arg_ty_llvm, "call.arg.abi.byref")?;
									self.builder().build_store(alloca, arg)?;
									alloca.into()
								}
							},
							abi::Repr::AsInteger => {
								let arg_ty_layout = self
									.compilation_unit
									.values
									.type_layout(&self.compilation_unit.resolved_target, arg_ty);
								let int_ty = self.lowerer.ctx.custom_width_int_type((arg_ty_layout.size * 8) as _);
								if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
									self.builder().build_load(int_ty, arg.into_pointer_value(), "")?
								} else {
									let alloca = self.build_alloca_at_top_of_bb(int_ty, "call.arg.abi.asinteger")?;
									self.builder().build_store(alloca, arg)?;
									self.builder().build_load(int_ty, alloca, "")?
								}
							},
						};
						values.push(val.into());
					}
					(values, ret_ptr)
				};

				let callee = self.resolve_inst(*callee);
				let val = match callee {
					AnyValueEnum::FunctionValue(callee_fn) => self.builder().build_direct_call(callee_fn, &args, "")?,
					AnyValueEnum::PointerValue(fn_ptr) => self.builder().build_indirect_call(llvm_fn_ty, fn_ptr, &args, "")?,
					_ => unreachable!("FnCall callee must lower to a function or function pointer"),
				};

				val.set_call_convention(self.lowerer.llvm_callconv_id(callconv) as u32);

				// if we have a ret_ptr, the actual val we returns for this call is a load to this pointer
				let ret_repr = self.lowerer.compute_fn_ret_ty_abi_repr(fn_ty);
				let val = if let Some(ret_ptr) = ret_ptr {
					// if vif abi already except a ref for this ty, we are good
					if self.lowerer.vif_abi_type_is_by_ref(fn_ty.ret_ty) {
						ret_ptr.as_any_value_enum()
					} else {
						self.build_load(fn_ty.ret_ty, ret_ptr)?
					}
				} else if ret_repr == abi::Repr::AsInteger {
					let ret_ty: BasicTypeEnum = self.lowerer.lower_type_basic(fn_ty.ret_ty);
					let abi_value = val.try_as_basic_value().unwrap_basic();
					let storage = self.build_alloca_at_top_of_bb(ret_ty, "call.ret.asinteger")?;
					self.builder().build_store(storage, abi_value)?;
					if self.lowerer.vif_abi_type_is_by_ref(fn_ty.ret_ty) {
						storage.as_any_value_enum()
					} else {
						self.builder()
							.build_load(ret_ty, storage, "call.ret.frominteger")?
							.as_any_value_enum()
					}
				} else {
					// vif abi may expect a ref, wrap in a alloc
					if self.lowerer.vif_abi_type_is_by_ref(fn_ty.ret_ty) {
						let ret_ty: BasicTypeEnum = self.lowerer.lower_type_basic(fn_ty.ret_ty);
						let alloca = self.build_alloca_at_top_of_bb(ret_ty, "")?;
						let val: BasicValueEnum = val.try_as_basic_value().unwrap_basic();
						self.builder().build_store(alloca, val)?;
						alloca.as_any_value_enum()
					} else {
						val.as_any_value_enum()
					}
				};

				self.vtir_inst_to_llvm_value.insert(id, val);
			},
			// unary
			vtir::Opcode::BoolNot { op } => {
				let op = self.resolve_inst(*op).into_int_value();
				let val = self.builder().build_not(op, "bool.not")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			// arithmtics
			vtir::Opcode::Add { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				if lhs.get_type().is_int_type() {
					let val = self.builder().build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_add(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::AddSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let add_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.sadd.sat" } else { "llvm.uadd.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let add_sat_fn = add_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let val = self.builder().build_call(add_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Sub { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				if lhs.get_type().is_int_type() {
					let val = self.builder().build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::SubSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let sub_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.ssub.sat" } else { "llvm.usub.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let sub_sat_fn = sub_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let value = self.builder().build_call(sub_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(id, value.as_any_value_enum());
			},
			vtir::Opcode::Mul { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				if lhs.get_type().is_int_type() {
					let val = self.builder().build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::MulSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let mul_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.smul.sat" } else { "llvm.umul.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let mul_sat_fn = mul_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let val = self.builder().build_call(mul_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Div { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					let val = if signed {
						self.builder()
							.build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "")?
					};
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_div(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::Rem { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);

					let val = if signed {
						self.builder()
							.build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
					};
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::Lt { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);

					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SLT, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::ULT, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OLT, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Lte { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SLE, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::ULE, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OLE, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Gt { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SGT, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::UGT, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OGT, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Gte { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SGE, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::UGE, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OGE, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BoolAnd { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = self.builder().build_and(lhs.into_int_value(), rhs.into_int_value(), "")?;

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BoolOr { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = self.builder().build_or(lhs.into_int_value(), rhs.into_int_value(), "")?;

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			// bitwise
			vtir::Opcode::Shl { lhs, rhs } | vtir::Opcode::ShlWrap { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self.builder().build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::ShlSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let shl_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.sshl.sat" } else { "llvm.ushl.sat" }).unwrap();
				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_llvm_ty = lhs_int.get_type();
				let shl_sat_fn = shl_sat_intrinsic
					.get_declaration(&self.lowerer.module, &[lhs_llvm_ty.into()])
					.unwrap();
				let val = self.builder().build_call(shl_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Shr { lhs, rhs } | vtir::Opcode::ShrWrap { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self
					.builder()
					.build_right_shift(lhs.into_int_value(), rhs.into_int_value(), signed, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::ShrSat { lhs, rhs } => {
				// Saturating right shift is the same as regular right shift
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self
					.builder()
					.build_right_shift(lhs.into_int_value(), rhs.into_int_value(), signed, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BitAnd { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self.builder().build_and(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BitOr { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self.builder().build_or(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BitXor { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);
				let val = self.builder().build_xor(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::BitNot { op } => {
				let op = self.resolve_inst(*op);
				let val = self.builder().build_not(op.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},

			vtir::Opcode::Eq { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::EQ, lhs.into_int_value(), rhs.into_int_value(), "")?
				} else if lhs.get_type().is_float_type() {
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "")?
				} else if lhs.get_type().is_pointer_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::EQ, lhs.into_pointer_value(), rhs.into_pointer_value(), "")?
				} else {
					unreachable!("eq not implemented for {lhs} and {rhs}")
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Neq { lhs, rhs } => {
				let lhs = self.resolve_inst(*lhs);
				let rhs = self.resolve_inst(*rhs);

				let val = if lhs.get_type().is_int_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::NE, lhs.into_int_value(), rhs.into_int_value(), "")?
				} else if lhs.get_type().is_float_type() {
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::ONE, lhs.into_float_value(), rhs.into_float_value(), "")?
				} else if lhs.get_type().is_pointer_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::NE, lhs.into_pointer_value(), rhs.into_pointer_value(), "")?
				} else {
					unreachable!()
				};

				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},

			// structs
			vtir::Opcode::StructInit { struct_ty, fields } => {
				let is_packed_struct = matches!(
					self.compilation_unit.values.index_to_key_value(*struct_ty),
					(value::Key::Type(value::Type::Struct(_)), value::Value::Struct(s)) if s.as_ref().is_packed()
				);
				if is_packed_struct {
					let r#struct = self.compilation_unit.values.index_to_value(*struct_ty).as_struct();
					let r#struct = r#struct.as_ref();
					let struct_ty_llvm = self.lowerer.lower_type(*struct_ty).into_int_type();

					let struct_int = struct_ty_llvm.const_int(0, false);
					let (_, struct_int) = fields.iter().try_fold((0u32, struct_int), |(cur_bit_offset, struct_int), field| {
						let field_ty = vtir.type_of(&self.compilation_unit.values, field);
						let field_ty_int = ctx.custom_width_int_type(self.compilation_unit.values.type_bit_size(field_ty));
						let struct_int: IntValue = {
							let field: BasicValueEnum = self.resolve_inst(*field).try_into().unwrap();
							let field = self.builder().build_bit_cast(field, field_ty_int, "")?.into_int_value();
							let field = self.builder().build_int_z_extend(field, struct_ty_llvm, "")?;
							let field = self
								.builder()
								.build_left_shift(field, struct_ty_llvm.const_int(cur_bit_offset as _, false), "")?;
							self.builder().build_or(struct_int, field, "")?
						};
						Ok::<(u32, inkwell::values::IntValue<'_>), BuilderError>((
							cur_bit_offset + self.compilation_unit.values.type_bit_size(field_ty),
							struct_int,
						))
					})?;
					self.vtir_inst_to_llvm_value.insert(id, struct_int.as_any_value_enum());
				} else {
					let struct_ty_llvm = self.lowerer.lower_type(*struct_ty).into_struct_type();
					let storage = self.build_alloca_at_top_of_bb(struct_ty_llvm.as_basic_type_enum(), "struct.init")?;
					let struct_def = self.compilation_unit.values.index_to_value(*struct_ty).as_struct();
					for (i, field) in fields.iter().enumerate() {
						let field_ty = struct_def.fields[i].ty;
						let field_ptr = self.builder().build_struct_gep(struct_ty_llvm, storage, i as u32, "")?;
						let field_value = self.resolve_inst(*field);
						self.build_store(field_ty, field_ptr, field_value)?;
					}
					self.vtir_inst_to_llvm_value.insert(id, storage.as_any_value_enum());
				}
			},
			vtir::Opcode::ArrayInit { array_ty, elements } => {
				let array_ty_idx = *array_ty;
				let array_ty = self.lowerer.lower_type(array_ty_idx).into_array_type();
				let elem_ty = self.compilation_unit.values.index_to_key(array_ty_idx).as_type_array().elem_ty;
				let storage = self.build_alloca_at_top_of_bb(array_ty, "array.init")?;
				for (i, element) in elements.iter().enumerate() {
					let index = self.lowerer.ctx.i32_type().const_int(i as u64, false);
					// SAFETY: sema ensures array init elements count match array type length
					let elem_ptr = unsafe {
						self.builder()
							.build_in_bounds_gep(array_ty, storage, &[self.lowerer.ctx.i32_type().const_zero(), index], "")?
					};
					let element = self.resolve_inst(*element);
					self.build_store(elem_ty, elem_ptr, element)?;
				}
				self.vtir_inst_to_llvm_value.insert(id, storage.as_any_value_enum());
			},
			vtir::Opcode::SliceInit { slice_ty, elements } => {
				let slice = self.compilation_unit.values.index_to_key(*slice_ty).as_type_slice();
				let elem_ty: BasicTypeEnum = self.lowerer.lower_type_basic(slice.pointee_ty);
				let slice_ty_llvm = self.lowerer.lower_type(*slice_ty).into_struct_type();
				let len_ty = self
					.lowerer
					.ctx
					.ptr_sized_int_type(&self.lowerer.target_machine.get_target_data(), None);
				let len = len_ty.const_int(elements.len().try_into().unwrap(), false);

				let ptr = if elements.is_empty() {
					self.lowerer.ctx.ptr_type(AddressSpace::default()).const_null()
				} else {
					let array_ty = elem_ty.array_type(elements.len().try_into().unwrap());
					let array_alloca = self.build_alloca_at_top_of_bb(array_ty, "slice.literal")?;
					for (i, element) in elements.iter().enumerate() {
						let index = self.lowerer.ctx.i32_type().const_int(i as u64, false);
						// SAFETY: `array_alloca` has `array_ty`, and `i` is bounded by the
						// number of elements used to construct that array type.
						let elem_ptr = unsafe {
							self.builder().build_in_bounds_gep(
								array_ty,
								array_alloca,
								&[self.lowerer.ctx.i32_type().const_zero(), index],
								"",
							)?
						};
						let element = self.resolve_inst(*element);
						self.build_store(slice.pointee_ty, elem_ptr, element)?;
					}

					// SAFETY: this branch is only reached for a non-empty array, so its
					// first element exists and has the slice's element type.
					unsafe {
						self.builder().build_in_bounds_gep(
							array_ty,
							array_alloca,
							&[self.lowerer.ctx.i32_type().const_zero(), self.lowerer.ctx.i32_type().const_zero()],
							"slice.literal.ptr",
						)?
					}
				};

				let undef = slice_ty_llvm.get_undef();
				let with_ptr = self.builder().build_insert_value(undef, ptr, 0, "slice.ptr")?;
				let with_len = self.builder().build_insert_value(with_ptr, len, 1, "slice.len")?;
				self.vtir_inst_to_llvm_value.insert(id, with_len.as_any_value_enum());
			},
			vtir::Opcode::AnyptrInit { value, value_ty } => {
				let resolved = self.resolve_inst(*value);
				let payload_ty: BasicTypeEnum = self.lowerer.lower_type_basic(*value_ty);
				let payload_alloca = self.build_alloca_at_top_of_bb(payload_ty, "any.payload")?;
				self.build_store(*value_ty, payload_alloca, resolved)?;

				let any_ty = self
					.lowerer
					.lower_type(self.compilation_unit.values.common.anyptr_t)
					.into_struct_type();
				let type_id_ty = self
					.lowerer
					.ctx
					.ptr_sized_int_type(&self.lowerer.target_machine.get_target_data(), None);
				let type_info_id = self
					.compilation_unit
					.type_to_type_info_id
					.find(value_ty)
					.expect("anyptr value type must have runtime type info");
				let type_info_id: usize = self
					.compilation_unit
					.type_to_type_info_id
					.kv(type_info_id)
					.1
					.load(std::sync::atomic::Ordering::Acquire)
					.into();
				let type_id = type_id_ty.const_int(type_info_id as u64, false);
				let undef = any_ty.get_undef();
				let with_ptr = self.builder().build_insert_value(undef, payload_alloca, 0, "any.ptr")?;
				let with_type_id = self.builder().build_insert_value(with_ptr, type_id, 1, "any.type")?;
				self.vtir_inst_to_llvm_value.insert(id, with_type_id.as_any_value_enum());
			},
			vtir::Opcode::StructFieldValue {
				struct_ty,
				field_idx,
				ret_ty,
			} => {
				let ty = vtir.type_of(&self.compilation_unit.values, struct_ty);
				let value::Key::Type(ty_key) = self.compilation_unit.values.index_to_key(ty) else {
					unreachable!("struct field operand has a non-type type")
				};
				let is_packed = match ty_key {
					value::Type::Struct(_) => {
						let s = self.compilation_unit.values.index_to_value(ty).as_struct();
						matches!(s.as_ref().layout, value::StructLayout::Packed { .. })
					},
					value::Type::Int { .. }
					| value::Type::Anyint
					| value::Type::Anyfloat
					| value::Type::Usize
					| value::Type::Isize
					| value::Type::F16
					| value::Type::F32
					| value::Type::F64
					| value::Type::F128
					| value::Type::Bool
					| value::Type::Void
					| value::Type::Enum(_)
					| value::Type::Union(_)
					| value::Type::Fn(_)
					| value::Type::Ptr(_)
					| value::Type::Slice(_)
					| value::Type::Array(_)
					| value::Type::NullPtr
					| value::Type::Any
					| value::Type::Anyptr
					| value::Type::GenericPoison
					| value::Type::Type
					| value::Type::Never
					| value::Type::EnumLiteral => false,
				};

				if is_packed {
					let ty = self.compilation_unit.values.index_to_value(ty).as_struct();
					let ty = ty.as_ref();
					if let value::StructLayout::Packed { packed_fields, .. } = ty.layout {
						let field_info = &packed_fields[*field_idx];
						let struct_value = self.resolve_inst(*struct_ty).into_int_value();

						let field = if field_info.offset > 0 {
							self.builder().build_right_shift(
								struct_value,
								ctx.custom_width_int_type(struct_value.get_type().get_bit_width())
									.const_int(field_info.offset as u64, false),
								false,
								"",
							)?
						} else {
							struct_value
						};

						let field = {
							let _ty: BasicTypeEnum = self.lowerer.lower_type_basic(*ret_ty);
							let field_ty_int = ctx.custom_width_int_type(self.compilation_unit.values.type_bit_size(*ret_ty));
							self.builder().build_int_truncate(field, field_ty_int, "")?
						};

						self.vtir_inst_to_llvm_value.insert(id, field.as_any_value_enum());
					} else {
						unreachable!()
					}
				} else {
					let struct_llvm_ty = self.lowerer.lower_type(ty).into_struct_type();
					let struct_ptr = self.resolve_inst(*struct_ty).into_pointer_value();
					let field_ptr = self.builder().build_struct_gep(struct_llvm_ty, struct_ptr, *field_idx as u32, "")?;
					let field = self.build_load(*ret_ty, field_ptr)?;
					self.vtir_inst_to_llvm_value.insert(id, field);
				}
			},
			vtir::Opcode::StructFieldPtr {
				struct_ptr,
				field_idx,
				ret_ty: _,
			} => {
				let struct_ty = {
					let struct_ptr_ty = vtir.type_of(&self.compilation_unit.values, struct_ptr);
					let ptr_ty = self.compilation_unit.values.index_to_key(struct_ptr_ty).as_type_ptr();
					ptr_ty.pointee_ty
				};
				let r#struct = self.compilation_unit.values.index_to_value(struct_ty).as_struct();
				let r#struct = r#struct.as_ref();

				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type_basic(struct_ty);
				let ptr = self.resolve_inst(*struct_ptr).into_pointer_value();

				// if the struct is packed, returns directly the ptr to the struct
				// the offset is encoded into the ret_ty type already so other insts should handle packed structs through that
				let ptr = if r#struct.is_packed() {
					ptr
				} else {
					self.builder().build_struct_gep(pointee_ty, ptr, *field_idx as u32, "")?
				};
				self.vtir_inst_to_llvm_value.insert(id, ptr.as_any_value_enum());
			},

			// unions
			vtir::Opcode::UnionInit {
				union_ty,
				field_idx,
				value: payload_value,
			} => match self.lowerer.lower_union_repr_for_field(*union_ty, (*field_idx).try_into().unwrap()) {
				UnionRepr::TagOnly(tag) => {
					self.vtir_inst_to_llvm_value.insert(id, tag.as_any_value_enum());
				},
				UnionRepr::Aggregate { ty: view_ty, tag, payload } => {
					let nominal_ty = self.lowerer.lower_type(*union_ty).into_struct_type();
					let alloca = self.build_alloca_at_top_of_bb(nominal_ty, "")?;
					if let Some((payload_field_idx, payload_wrapper_ty)) = payload {
						let mut payload_ptr = self.builder().build_struct_gep(view_ty, alloca, payload_field_idx, "")?;
						if let Some(payload_wrapper_ty) = payload_wrapper_ty {
							payload_ptr = self.builder().build_struct_gep(payload_wrapper_ty, payload_ptr, 0, "")?;
						}
						let payload_value = payload_value.expect("union init of a field expecting a payload but has no value");
						let payload_ty = self.compilation_unit.values.index_to_value(*union_ty).as_union().fields[*field_idx]
							.ty
							.unwrap();
						let resolved = self.resolve_inst(payload_value);
						self.build_store(payload_ty, payload_ptr, resolved)?;
					} else {
						assert!(payload_value.is_none(), "payload-less union field has a value");
					}

					if let Some((tag, tag_field_idx)) = tag {
						let tag_ptr = self.builder().build_struct_gep(view_ty, alloca, tag_field_idx, "")?;
						#[allow(
							clippy::disallowed_methods,
							reason = "lower_union_repr_for_field only returns the llvm type for the tag, which does not use by-ref \
							          semantics"
						)]
						self.builder().build_store(tag_ptr, tag)?;
					}

					self.vtir_inst_to_llvm_value.insert(id, alloca.as_any_value_enum());
				},
			},
			vtir::Opcode::UnionTag { union_val, tag_ty } => {
				let union_ty = vtir.type_of(&self.compilation_unit.values, union_val);
				match self.lowerer.lower_union_repr_for_field(union_ty, 0) {
					UnionRepr::TagOnly(_) => {
						let tag = self.resolve_inst(*union_val);
						self.vtir_inst_to_llvm_value.insert(id, tag);
					},
					UnionRepr::Aggregate { tag, .. } => {
						let (_, tag_field_idx) = tag.expect("UnionTag on bare union");
						let union_ptr = self.resolve_inst(*union_val).into_pointer_value();
						let union_ty_llvm = self.lowerer.lower_type(union_ty).into_struct_type();
						let tag_ptr = self.builder().build_struct_gep(union_ty_llvm, union_ptr, tag_field_idx, "")?;
						let tag = self.build_load(*tag_ty, tag_ptr)?;
						self.vtir_inst_to_llvm_value.insert(id, tag.as_any_value_enum());
					},
				}
			},
			vtir::Opcode::UnionFieldValue {
				union_val,
				field_idx,
				ret_ty,
			} => {
				let union_ty_idx = vtir.type_of(&self.compilation_unit.values, union_val);
				let union_llvm_ty = self.lowerer.lower_type(union_ty_idx);
				let union_ptr = self.resolve_inst(*union_val).into_pointer_value();

				let UnionRepr::Aggregate {
					payload: Some((payload_field_idx, _)),
					..
				} = self.lowerer.lower_union_repr_for_field(union_ty_idx, *field_idx as u32)
				else {
					unreachable!("UnionFieldValue requires a payload");
				};
				let payload_ptr = self
					.builder()
					.build_struct_gep(union_llvm_ty.into_struct_type(), union_ptr, payload_field_idx, "")?;
				let result = self.build_load(*ret_ty, payload_ptr)?;
				self.vtir_inst_to_llvm_value.insert(id, result);
			},

			// builtins
			vtir::Opcode::UnsafeIntCast { src, dst_ty } => {
				let src = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let src = self.resolve_inst(*src);
					src.into_int_value()
				};
				let is_signed = self.compilation_unit.values.type_is_int_signed(*dst_ty);

				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = self
					.builder()
					.build_int_cast_sign_flag(src, dst_ty.into_int_type(), is_signed, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::UnsafeFloatCast { src, dst_ty } => {
				let src = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let _src_ty = self.lowerer.lower_type(src_ty);
					let src = self.resolve_inst(*src);
					src.into_float_value()
				};
				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = self.builder().build_float_cast(src, dst_ty.into_float_type(), "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::IntToFloat { src, dst_ty } => {
				let (src, is_signed) = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let src = self.resolve_inst(*src);
					let is_signed = self.compilation_unit.values.type_is_int_signed(src_ty);
					(src.into_int_value(), is_signed)
				};
				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = if is_signed {
					self.builder().build_signed_int_to_float(src, dst_ty.into_float_type(), "")?
				} else {
					self.builder().build_unsigned_int_to_float(src, dst_ty.into_float_type(), "")?
				};
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::SizeOf { of } => {
				let of = self.lowerer.lower_type(of.as_interned());
				let val = of.size_of().expect("@size_of called on non-sized type");
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::Zeroed { ty } => {
				let ty_idx = *ty;
				let ty: BasicTypeEnum<'_> = self.lowerer.lower_type_basic(ty_idx);
				let zeroed = self.materialize_nominal_value(ty_idx, ty.const_zero(), "zeroed")?;
				self.vtir_inst_to_llvm_value.insert(id, zeroed);
			},
			vtir::Opcode::BitCast { src, dst_ty } => {
				let src: BasicValueEnum = self.resolve_inst(*src).try_into().unwrap();
				let dst_ty: BasicTypeEnum = self.lowerer.lower_type_basic(*dst_ty);
				let val = self.builder().build_bit_cast(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::AnyptrIs { value, target_ty } => {
				let value = self.resolve_inst(*value).into_struct_value();
				let runtime_type_id = self.builder().build_extract_value(value, 1, "any.type")?.into_int_value();
				let type_info_id = self
					.compilation_unit
					.type_to_type_info_id
					.find(target_ty)
					.expect("anyptr target type must have runtime type info");
				let type_info_id: usize = self
					.compilation_unit
					.type_to_type_info_id
					.kv(type_info_id)
					.1
					.load(std::sync::atomic::Ordering::Acquire)
					.into();
				let target_type_id = runtime_type_id.get_type().const_int(type_info_id as u64, false);
				let is_target = self
					.builder()
					.build_int_compare(inkwell::IntPredicate::EQ, runtime_type_id, target_type_id, "any.is")?;
				self.vtir_inst_to_llvm_value.insert(id, is_target.as_any_value_enum());
			},
			vtir::Opcode::AnyptrAs { value, target_ty } => {
				let value = self.resolve_inst(*value).into_struct_value();
				let payload_ptr = self.builder().build_extract_value(value, 0, "any.ptr")?.into_pointer_value();
				let loaded = self.build_load(*target_ty, payload_ptr)?;
				self.vtir_inst_to_llvm_value.insert(id, loaded);
			},
			vtir::Opcode::AnyptrPtr { value, ptr_ty: _ } => {
				let value = self.resolve_inst(*value).into_struct_value();
				let payload_ptr = self.builder().build_extract_value(value, 0, "any.ptr")?;
				self.vtir_inst_to_llvm_value.insert(id, payload_ptr.as_any_value_enum());
			},
			vtir::Opcode::AnyptrFromRaw { ptr, type_id } => {
				let ptr = self.resolve_inst(*ptr).into_pointer_value();
				let type_id = self.resolve_inst(*type_id).into_int_value();
				let runtime_type_id_ty = self
					.lowerer
					.ctx
					.ptr_sized_int_type(&self.lowerer.target_machine.get_target_data(), None);
				let type_id = self.builder().build_int_z_extend(type_id, runtime_type_id_ty, "any.type")?;
				let any_ty = self
					.lowerer
					.lower_type(self.compilation_unit.values.common.anyptr_t)
					.into_struct_type();
				let with_ptr = self.builder().build_insert_value(any_ty.get_undef(), ptr, 0, "any.ptr")?;
				let value = self.builder().build_insert_value(with_ptr, type_id, 1, "any.value")?;
				self.vtir_inst_to_llvm_value.insert(id, value.as_any_value_enum());
			},
			vtir::Opcode::AnyptrTypeInfo { value, ty } => {
				let value = self.resolve_inst(*value).into_struct_value();
				let runtime_type_id = self.builder().build_extract_value(value, 1, "any.type")?.into_int_value();
				let ptr_ty = self.lowerer.ctx.ptr_type(AddressSpace::default());
				let table_ptr = self
					.builder()
					.build_load(ptr_ty, self.lowerer.type_info_table.as_pointer_value(), "any.type.info.table")?
					.into_pointer_value();
				// SAFETY: runtime type IDs are dense indices into the finalized RTTI pointer table.
				let type_info_ptr_ptr = unsafe {
					self.builder()
						.build_in_bounds_gep(ptr_ty, table_ptr, &[runtime_type_id], "any.type.info.slot")?
				};
				let type_info_ptr = self
					.builder()
					.build_load(ptr_ty, type_info_ptr_ptr, "any.type.info.ptr")?
					.into_pointer_value();
				let loaded = self.build_load(*ty, type_info_ptr)?;
				self.vtir_inst_to_llvm_value.insert(id, loaded);
			},
			vtir::Opcode::Undefined { ty } => {
				let ty_idx = *ty;
				let ty: BasicTypeEnum<'_> = self.lowerer.lower_type_basic(ty_idx);

				let undef_value = match ty {
					BasicTypeEnum::IntType(int_ty) => int_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::FloatType(float_ty) => float_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::PointerType(ptr_ty) => ptr_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::ArrayType(array_ty) => array_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::StructType(struct_ty) => struct_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::VectorType(vec_ty) => vec_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::ScalableVectorType(svec_ty) => svec_ty.get_undef().as_any_value_enum(),
				};
				let undef_value = self.materialize_nominal_value(ty_idx, undef_value.try_into().unwrap(), "undefined")?;
				self.vtir_inst_to_llvm_value.insert(id, undef_value);
			},
			vtir::Opcode::SliceFromRawParts { slice_ty, ptr, len } => {
				let slice_ty = self.lowerer.lower_type(*slice_ty).into_struct_type();
				let ptr: BasicValueEnum = self.resolve_inst(*ptr).try_into().unwrap();
				let len: BasicValueEnum = self.resolve_inst(*len).try_into().unwrap();
				let undef = slice_ty.get_undef();
				let with_ptr = self.builder().build_insert_value(undef, ptr, 0, "slice.ptr").unwrap();
				let with_len = self.builder().build_insert_value(with_ptr, len, 1, "slice.len").unwrap();
				self.vtir_inst_to_llvm_value.insert(id, with_len.as_any_value_enum());
			},
			vtir::Opcode::SlicePtr { slice, ptr_ty } => {
				let slice = self.resolve_inst(*slice).into_struct_value();
				let ptr = self.builder().build_extract_value(slice, 0, "slice.ptr").unwrap();
				self.vtir_inst_to_llvm_value.insert(id, ptr.as_any_value_enum());
			},
			vtir::Opcode::SliceLen { slice } => {
				let slice = self.resolve_inst(*slice).into_struct_value();
				let len = self.builder().build_extract_value(slice, 1, "slice.len").unwrap();
				self.vtir_inst_to_llvm_value.insert(id, len.as_any_value_enum());
			},
			vtir::Opcode::PtrToInt { src, dst_ty } => {
				let src = self.resolve_inst(*src).into_pointer_value();
				let dst_ty = self.lowerer.lower_type(*dst_ty).into_int_type();
				let val = self.builder().build_ptr_to_int(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::IntToPtr { src, dst_ty } => {
				let src = self.resolve_inst(*src).into_int_value();
				let dst_ty = self.lowerer.lower_type(*dst_ty).into_pointer_type();
				let val = self.builder().build_int_to_ptr(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(id, val.as_any_value_enum());
			},
			vtir::Opcode::SliceCopyNonoverlapping { slice_ty, src, dst } => {
				let src_pointee_ty = self.compilation_unit.values.index_to_key(*slice_ty).as_type_slice().pointee_ty;
				let src_pointee_ty = self.lowerer.lower_type(src_pointee_ty);

				let src = self.resolve_inst(*src).into_struct_value();
				let dst = self.resolve_inst(*dst).into_struct_value();
				let src_ptr = self.builder().build_extract_value(src, 0, "slice.ptr")?.into_pointer_value();
				let src_len = self.builder().build_extract_value(src, 1, "slice.len").unwrap().into_int_value();
				let src_len_in_bytes = self
					.builder()
					.build_int_mul(src_len, src_pointee_ty.size_of().unwrap(), "src_len_in_bytes")?;
				let dst_ptr = self.builder().build_extract_value(dst, 0, "slice.ptr")?.into_pointer_value();
				let value = self.builder().build_memcpy(dst_ptr, 4, src_ptr, 4, src_len_in_bytes).unwrap();
				self.vtir_inst_to_llvm_value.insert(id, value.as_any_value_enum());
			},
			vtir::Opcode::SliceElemPtr { slice, index, elem_ptr_ty } => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*elem_ptr_ty).as_type_ptr().pointee_ty;
				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type_basic(pointee_ty);

				let src = self.resolve_inst(*slice).into_struct_value();
				let src_ptr = self
					.builder()
					.build_extract_value(src, 0, "slice.ptr")
					.unwrap()
					.into_pointer_value();
				let index = self.resolve_inst(*index).into_int_value();
				// SAFETY: The VTIR slice value carries a pointer to elements of `pointee_ty`,
				// and `index` is the computed element offset for this instruction.
				let value = unsafe {
					self.builder()
						.build_in_bounds_gep(pointee_ty, src_ptr, &[index], "slice.elem.ptr")?
				};
				self.vtir_inst_to_llvm_value.insert(id, value.as_any_value_enum());
			},
			vtir::Opcode::PtrElemPtr {
				array_ptr,
				index,
				elem_ptr_ty,
			} => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*elem_ptr_ty).as_type_ptr().pointee_ty;
				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type_basic(pointee_ty);
				let array_ptr = self.resolve_inst(*array_ptr).into_pointer_value();
				let index = self.resolve_inst(*index).into_int_value();
				// SAFETY: The pointer operand is typed as pointing at `pointee_ty`, and
				// `index` is the computed element offset for this instruction.
				let value = unsafe {
					self.builder()
						.build_in_bounds_gep(pointee_ty, array_ptr, &[index], "ptr.array.elem.ptr")?
				};
				self.vtir_inst_to_llvm_value.insert(id, value.as_any_value_enum());
			},

			vtir::Opcode::DbgSrcLoc { line, col } => {
				if let Some(di_gen) = self.di_gen.as_mut() {
					let location = di_gen.di_ctx.builder.create_debug_location(
						self.lowerer.ctx,
						*line as _,
						*col as _,
						di_gen.di_lexical_block_stack.last().unwrap().as_debug_info_scope(),
						None,
					);
					self.builder().set_current_debug_location(location);
				}
			},
			vtir::Opcode::Branch {
				cond,
				then_body,
				else_body,
			} => {
				let insert_bb = self.builder().get_insert_block().unwrap();
				let then_block = ctx.insert_basic_block_after(insert_bb, "then");
				let else_block = ctx.insert_basic_block_after(then_block, "else");

				let cond = self.resolve_inst(*cond).into_int_value();
				let branch = self.builder().build_conditional_branch(cond, then_block, else_block)?;

				// then
				self.builder().position_at_end(then_block);
				self.lower_body(vtir, then_block, then_body);

				// else
				self.builder().position_at_end(else_block);
				self.lower_body(vtir, else_block, else_body);

				self.vtir_inst_to_llvm_value.insert(id, branch.as_any_value_enum());
			},
			vtir::Opcode::Switch { operand, cases, else_body } => {
				let insert_bb = self.builder().get_insert_block().unwrap();
				let operand_val = self.resolve_inst(*operand).into_int_value();

				// Create basic blocks: one per case body + else
				let else_bb = ctx.insert_basic_block_after(insert_bb, "switch_else");
				let mut case_bbs = Vec::with_capacity(cases.len());
				let mut prev_bb = else_bb;
				for i in 0..cases.len() {
					let bb = ctx.insert_basic_block_after(prev_bb, &format!("switch_case_{i}"));
					case_bbs.push(bb);
					prev_bb = bb;
				}

				// Build LLVM switch cases: (value, basic_block) pairs
				let mut switch_cases = Vec::new();
				for (i, case) in cases.iter().enumerate() {
					for item in case.items.iter() {
						let item_val = self.resolve_inst(*item).into_int_value();
						switch_cases.push((item_val, case_bbs[i]));
					}
				}

				self.builder().build_switch(operand_val, else_bb, &switch_cases)?;

				// Lower each case body
				for (i, case) in cases.iter().enumerate() {
					self.builder().position_at_end(case_bbs[i]);
					self.lower_body(vtir, case_bbs[i], case.body);
				}

				// Lower else body
				self.builder().position_at_end(else_bb);
				if else_body.is_empty() {
					// Exhaustive switch: else is unreachable
					self.builder().build_unreachable()?;
				} else {
					self.lower_body(vtir, else_bb, else_body);
				}

				// For exhaustive switches (no else body), the else BB has an
				// unreachable terminator, so we need a merge block for any
				// subsequent instructions. For non-exhaustive switches, the
				// else body's lowering leaves the builder positioned correctly.
				if else_body.is_empty() {
					let merge_bb = ctx.insert_basic_block_after(prev_bb, "switch_merge");
					self.builder().position_at_end(merge_bb);
				}
			},
			vtir::Opcode::Abort => {
				let trap_fn = self.lowerer.intrins.trap.get_declaration(&self.lowerer.module, &[]).unwrap();
				self.builder().build_call(trap_fn, &[], "@abort")?;
			},
			vtir::Opcode::Unreachable => {
				self.builder().build_unreachable()?;
			},

			// invalid insts
			vtir::Opcode::StackAllocInferred { .. } => {
				unreachable!("StackAllocInferred should be replaced at uir2tir time");
			},
		};
		Ok(())
	}

	fn lower_body(
		&mut self,
		vtir: &Vtir,
		parent_bb: BasicBlock<'ctx>,
		insts: &[vtir::InstructionRef],
	) {
		for inst in insts.iter() {
			let id = inst.as_id().unwrap();
			let inst = &vtir.instructions[id];
			self.lower_body_inst(vtir, parent_bb, id, inst).unwrap();
		}
	}

	fn lower_block(
		&mut self,
		vtir: &Vtir,
		parent_bb: BasicBlock<'ctx>,
		id: vtir::InstructionId,
		instructions: &[vtir::InstructionRef],
		ret_ty: value::Index,
	) -> Result<Option<PhiValue<'ctx>>, BuilderError> {
		let after_block_bb = self.lowerer.ctx.insert_basic_block_after(parent_bb, "after_block");
		self.vtir_block_to_break_list.insert(id, BreakList {
			body_bb: None,
			after_bb: after_block_bb,
			breaks: Vec::new(),
		});
		self.lower_body(vtir, parent_bb, instructions);

		// If the body didn't terminate, add fallthrough to after_block
		let current_bb = self.builder().get_insert_block().unwrap();
		if current_bb.get_terminator().is_none() {
			self.builder().build_unconditional_branch(after_block_bb)?;
		}

		self.builder().position_at_end(after_block_bb);

		let phi = if ret_ty != self.compilation_unit.values.common.void_t {
			let ret_ty = if self.lowerer.vif_abi_type_is_by_ref(ret_ty) {
				self.lowerer.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum()
			} else {
				self.lowerer.lower_type_basic(ret_ty)
			};
			let break_list = self.vtir_block_to_break_list.get(&id).unwrap();
			if !break_list.breaks.is_empty() {
				let phi = self.builder().build_phi(ret_ty, "")?;
				for (break_value, break_bb) in &break_list.breaks {
					phi.add_incoming(&[(break_value, *break_bb)]);
				}
				Some(phi)
			} else {
				None
			}
		} else {
			None
		};
		Ok(phi)
	}

	#[allow(
		clippy::disallowed_methods,
		reason = "ABI adaptation requires loads and stores with explicit LLVM types"
	)]
	fn lower_fn_body(
		&mut self,
		interned_fn_value: value::Index,
		body: &Vtir,
	) -> Result<(), BuilderError> {
		let fn_value = self.compilation_unit.values.index_to_key(interned_fn_value).as_fn();
		let fn_ty_idx = fn_value.ty;
		let fn_ty = self.compilation_unit.values.index_to_key(fn_value.ty).as_type_fn();

		if fn_ty.external {
			let fn_value = self.resolve_inst(InstructionRef::Interned(interned_fn_value)).into_function_value();
			fn_value.set_linkage(Linkage::External);
			return Ok(());
		}

		let fn_value = self.resolve_inst(InstructionRef::Interned(interned_fn_value)).into_function_value();
		let block = self.lowerer.ctx.append_basic_block(fn_value, "entry");
		self.builder().position_at_end(block);

		if let Some(di_gen) = self.di_gen.as_mut() {
			let di_builder = &di_gen.di_ctx.builder;
			let di_cu = &di_gen.di_ctx.cu;

			// no debug info for external or builtins
			if !fn_ty.external {
				let mangled_name = fn_value.get_name().to_string_lossy();
				let di_sp_type = di_builder.create_subroutine_type(di_gen.di_file, None, &[], DIFlags::PUBLIC);
				let di_mangled_name = format!("{} {}", fn_value.print_to_string().to_string_lossy(), mangled_name);
				let di_sp = di_builder.create_function(
					di_gen.di_file.as_debug_info_scope(),
					&mangled_name,
					Some(&di_mangled_name),
					di_gen.di_file,
					0,
					di_sp_type,
					true,
					true,
					0,
					DIFlags::PROTOTYPED,
					false,
				);
				fn_value.set_subprogram(di_sp);
			}

			let di_parent_scope = fn_value
				.get_subprogram()
				.map(|s| s.as_debug_info_scope())
				.unwrap_or(di_cu.as_debug_info_scope());

			di_gen
				.di_lexical_block_stack
				.push(di_builder.create_lexical_block(di_parent_scope, di_gen.di_file, 0, 0));
		}

		self.cur_fn = Some(fn_value);
		self.cur_llvm_fn_param_idx = 0;
		self.cur_fn_ty = fn_ty_idx;

		// need to decode from the function ABI to the vif ABI
		let sret = self.lowerer.fn_use_sret(fn_ty);
		self.cur_fn_args = Vec::with_capacity(fn_ty.params.len());
		for (i, &arg_ty) in fn_ty
			.params
			.iter()
			.enumerate()
			.filter(|&(i, _)| !fn_ty.comptime_params[i])
			.map(|(_, arg_ty)| arg_ty)
			.enumerate()
		{
			let arg = fn_value.get_nth_param(if sret { 1 } else { 0 } + i as u32).unwrap();
			let arg_abi_ty = arg.get_type();
			let arg: BasicValueEnum = match self.lowerer.compute_fn_param_abi_repr(fn_ty, arg_ty) {
				abi::Repr::ByValue => {
					if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
						let alloca = self.build_alloca_at_top_of_bb(arg_abi_ty, "fn.arg.byval")?;
						self.builder().build_store(alloca, arg)?;
						alloca.into()
					} else {
						arg
					}
				},
				abi::Repr::ByRef => {
					if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
						arg
					} else {
						let pointee_ty: BasicTypeEnum = self.lowerer.lower_type_basic(arg_ty);
						self.builder().build_load(pointee_ty, arg.into_pointer_value(), "")?
					}
				},
				abi::Repr::AsInteger => {
					let arg_ty_llvm: BasicTypeEnum = self.lowerer.lower_type_basic(arg_ty);
					let alloca = self.build_alloca_at_top_of_bb(arg_abi_ty, "fn.arg.asinteger")?;
					self.builder().build_store(alloca, arg)?;

					if self.lowerer.vif_abi_type_is_by_ref(arg_ty) {
						alloca.into()
					} else {
						self.builder().build_load(arg_ty_llvm, alloca, "")?
					}
				},
			};
			self.cur_fn_args.push(arg);
		}

		self.lower_body(body, block, body.main_body);
		self.cur_fn = None;

		self.builder().unset_current_debug_location();
		if let Some(di) = self.di_gen.as_mut() {
			di.di_lexical_block_stack.pop();
		}
		Ok(())
	}
}

pub fn initialize(build_opts: &Build) -> inkwell::context::Context {
	// required for tests with cargo test
	// not great since the first build_opts win but eh we should require them here tbh only for debugging
	static WAS_INITIALIZED: OnceLock<()> = OnceLock::new();
	WAS_INITIALIZED.get_or_init(|| {
		let empty_arg = c"".as_ptr();
		let args = [
			c"vifc".as_ptr(),
			c"-x86-asm-syntax=intel".as_ptr(),
			if build_opts.dump_llvm_timings {
				c"-time-passes".as_ptr()
			} else {
				empty_arg
			},
		];

		// We need to initialize targets before CLI opts parsing to have cl::opt flags registered
		Target::initialize_native(&InitializationConfig {
			asm_parser: true,
			asm_printer: true,
			base: true,
			disassembler: false,
			info: true,
			machine_code: true,
		});

		// SAFETY: todo
		unsafe { LLVMParseCommandLineOptions(args.len() as i32, args.as_ptr(), core::ptr::null()) };
	});

	inkwell::context::Context::create()
}

struct DebugInfoCtx<'ctx> {
	builder: inkwell::debug_info::DebugInfoBuilder<'ctx>,
	cu: inkwell::debug_info::DICompileUnit<'ctx>,
	module_to_file: FxHashMap<ModuleId, inkwell::debug_info::DIFile<'ctx>>,
}

pub struct Lowerer<'ctx> {
	compilation_unit: &'ctx CompilationUnit,
	ctx: &'ctx inkwell::context::Context,
	builder: inkwell::builder::Builder<'ctx>,
	module: inkwell::module::Module<'ctx>,
	target_machine: inkwell::targets::TargetMachine,
	interned_value_to_llvm_type: FxHashMap<value::Index, AnyTypeEnum<'ctx>>,
	interned_value_to_llvm_value: FxHashMap<value::Index, AnyValueEnum<'ctx>>,
	interned_value_to_llvm_storage: FxHashMap<value::Index, PointerValue<'ctx>>,
	type_info_table: GlobalValue<'ctx>,
	attributes: LlvmAttributes,
	intrins: LlvmIntrins,
	di: Option<DebugInfoCtx<'ctx>>,
}
impl<'ctx> Lowerer<'ctx> {
	fn vif_abi_type_is_by_ref(
		&self,
		ty: value::Index,
	) -> bool {
		let (value::Key::Type(ty_key), value) = self.compilation_unit.values.index_to_key_value(ty) else {
			unreachable!("ABI classification expected a type")
		};
		match ty_key {
			// LLVM performance recommendation: do not create values of aggregate type https://llvm.org/docs/Frontend/PerformanceTips.html#avoid-creating-values-of-aggregate-type
			// so we *almost* don't do that
			value::Type::Union(_) => {
				let value::Value::Union(_) = value else {
					unreachable!("union type without union value")
				};
				self.compilation_unit
					.values
					.type_union_layout(&self.compilation_unit.resolved_target, ty)
					.payload
					.size != 0
			},
			value::Type::Struct(_) => {
				let value::Value::Struct(r#struct) = value else {
					unreachable!("struct type without struct value")
				};
				let r#struct = r#struct.as_ref();
				!r#struct.is_packed()
			},
			value::Type::Array(_) => true,
			value::Type::Int { .. }
			| value::Type::Anyint
			| value::Type::Anyfloat
			| value::Type::Usize
			| value::Type::Isize
			| value::Type::F16
			| value::Type::F32
			| value::Type::F64
			| value::Type::F128
			| value::Type::Bool
			| value::Type::Void
			| value::Type::Enum(_)
			| value::Type::Fn(_)
			| value::Type::Ptr(_)
			| value::Type::Slice(_)
			| value::Type::NullPtr
			| value::Type::Any
			| value::Type::Anyptr
			| value::Type::GenericPoison
			| value::Type::Type
			| value::Type::Never
			| value::Type::EnumLiteral => false,
		}
	}

	fn llvm_callconv_id(
		&self,
		callconv: CallingConvention,
	) -> llvm_sys::LLVMCallConv {
		match callconv {
			CallingConvention::Vif => llvm_sys::LLVMCallConv::LLVMFastCallConv,
			CallingConvention::C => llvm_sys::LLVMCallConv::LLVMCCallConv,
			CallingConvention::Fast => llvm_sys::LLVMCallConv::LLVMFastCallConv,
			CallingConvention::Cold => llvm_sys::LLVMCallConv::LLVMColdCallConv,
			CallingConvention::X86_64Windows => llvm_sys::LLVMCallConv::LLVMWin64CallConv,

			CallingConvention::Count => unreachable!(),
		}
	}

	pub fn new(
		ctx: &'ctx inkwell::context::Context,
		compilation_unit: &'ctx CompilationUnit,
		build_opts: &Build,
	) -> Self {
		let triple = TargetTriple::create(&compilation_unit.resolved_target.triple.to_string());
		let target = { Target::from_triple(&triple).unwrap() };

		let cpu = TargetMachine::get_host_cpu_name();
		let features = TargetMachine::get_host_cpu_features();

		let target_machine = target
			.create_target_machine(
				&triple,
				cpu.to_string_lossy().as_ref(),
				features.to_string_lossy().as_ref(),
				match build_opts.opt {
					0 => inkwell::OptimizationLevel::None,
					1 => inkwell::OptimizationLevel::Less,
					2 => inkwell::OptimizationLevel::Default,
					3 => inkwell::OptimizationLevel::Aggressive,
					_ => unreachable!(),
				},
				inkwell::targets::RelocMode::Default,
				inkwell::targets::CodeModel::Default,
			)
			.unwrap();

		let builder = ctx.create_builder();
		let module = ctx.create_module("vifmod");
		let data_layout = target_machine.get_target_data().get_data_layout();
		module.set_data_layout(&data_layout);
		module.set_triple(&triple);
		let type_info_ptr_ty = ctx.ptr_type(AddressSpace::default());
		let type_info_table = module.add_global(type_info_ptr_ty, None, "__vif_type_info_table");
		type_info_table.set_linkage(Linkage::Private);
		type_info_table.set_constant(true);
		type_info_table.set_unnamed_address(UnnamedAddress::Global);
		type_info_table.set_initializer(&type_info_ptr_ty.const_null());

		// On Windows we want to emit CodeView data for PDB-based debuggers
		let u32_one = ctx.i32_type().const_int(1, false);

		#[cfg(target_os = "windows")]
		module.add_metadata_flag(
			"CodeView",
			inkwell::module::FlagBehavior::Error,
			ctx.metadata_node(&[u32_one.into()]),
		);

		let di = if build_opts.debug_info {
			let modules = compilation_unit.modules.read();
			let root_mod_path = modules[compilation_unit.root_module].path.clone();
			let (builder, cu) = module.create_debug_info_builder(
				true,
				inkwell::debug_info::DWARFSourceLanguage::C,
				root_mod_path.file_name().unwrap(),
				root_mod_path.parent().unwrap().as_str(),
				"vifc",
				false,
				"",
				0,
				"",
				inkwell::debug_info::DWARFEmissionKind::Full,
				0,
				false,
				false,
				"",
				"",
			);

			let mut module_to_file = FxHashMap::default();
			for (id, module) in compilation_unit.modules.read().iter().enumerate() {
				let path = &module.path;
				let file = builder.create_file(path.file_name().unwrap(), path.parent().unwrap().as_str());
				module_to_file.insert(ModuleId::from(id), file);
			}

			Some(DebugInfoCtx {
				builder,
				cu,
				module_to_file,
			})
		} else {
			None
		};

		Self {
			compilation_unit,
			ctx,
			builder,
			module,
			target_machine,
			interned_value_to_llvm_type: Default::default(),
			interned_value_to_llvm_value: Default::default(),
			interned_value_to_llvm_storage: Default::default(),
			type_info_table,
			attributes: LlvmAttributes::new(ctx),
			intrins: LlvmIntrins::new(ctx),
			di,
		}
	}

	fn build_slice(
		&mut self,
		slice_ty: &StructType<'ctx>,
		ptr: BasicValueEnum<'ctx>,
		len: BasicValueEnum<'ctx>,
	) -> AnyValueEnum<'ctx> {
		let undef = slice_ty.get_undef();
		let with_ptr = self.builder.build_insert_value(undef, ptr, 0, "slice.ptr").unwrap();
		let with_len = self.builder.build_insert_value(with_ptr, len, 1, "slice.len").unwrap();
		with_len.as_any_value_enum()
	}

	fn lower_interned_value(
		&mut self,
		val: value::Index,
	) -> AnyValueEnum<'ctx> {
		if let Some(value) = self.interned_value_to_llvm_value.get(&val) {
			return *value;
		}

		let ty = self.compilation_unit.values.type_of_interned(val);
		let value = if self.vif_abi_type_is_by_ref(ty) {
			self.lower_interned_value_in_const_storage(val).as_any_value_enum()
		} else {
			self.lower_interned_value_as_llvm_value(val)
		};
		self.interned_value_to_llvm_value.insert(val, value);
		value
	}

	fn lower_interned_value_in_const_storage(
		&mut self,
		val: value::Index,
	) -> PointerValue<'ctx> {
		if let Some(storage) = self.interned_value_to_llvm_storage.get(&val) {
			return *storage;
		}

		let initializer: BasicValueEnum = self.lower_interned_value_as_llvm_value(val).try_into().unwrap();
		let global = self.module.add_global(initializer.get_type(), None, "interned");
		global.set_linkage(Linkage::Private);
		global.set_constant(true);
		global.set_unnamed_address(UnnamedAddress::Global);
		global.set_initializer(&initializer);
		let ptr = global.as_pointer_value();
		self.interned_value_to_llvm_storage.insert(val, ptr);
		ptr
	}

	fn lower_interned_value_as_llvm_value(
		&mut self,
		val: value::Index,
	) -> AnyValueEnum<'ctx> {
		let key = self.compilation_unit.values.index_to_key(val);
		match key {
			value::Key::Str { value, slice_ty } => {
				let slice_ty = self.lower_type(*slice_ty).into_struct_type();
				let ptr = self
					.builder
					.build_global_string_ptr(core::str::from_utf8(value).expect("todo change"), "")
					.unwrap()
					.as_basic_value_enum();
				let len = self
					.ctx
					.ptr_sized_int_type(&self.target_machine.get_target_data(), None)
					.const_int(value.len().try_into().unwrap(), false)
					.as_basic_value_enum();
				slice_ty.const_named_struct(&[ptr, len]).as_any_value_enum()
			},
			value::Key::Slice { ty, ptr, len } => {
				let slice_ty = self.lower_type(*ty).into_struct_type();
				let ptr = self
					.lower_interned_value_as_llvm_value(*ptr)
					.into_pointer_value()
					.as_basic_value_enum();
				let len = self.lower_interned_value_as_llvm_value(*len).into_int_value().as_basic_value_enum();
				slice_ty.const_named_struct(&[ptr, len]).as_any_value_enum()
			},
			value::Key::Int { ty, value } => {
				let ty = self.lower_type(*ty);
				let str = value.to_string();
				ty.into_int_type()
					.const_int_from_string(&str, inkwell::types::StringRadix::Decimal)
					.unwrap()
					.into()
			},
			value::Key::Float { ty, value } => {
				let ty = self.lower_type(*ty);
				let value = **value as f64; // TODO(zino): f128 casted to f64 not good
				ty.into_float_type().const_float(value).into()
			},
			value::Key::Bool(b) => self.ctx.bool_type().const_int(*b as u64, false).into(),
			value::Key::Ptr(p) => match p.kind {
				value::PtrKind::Value(v) => self.lower_interned_value(v).as_any_value_enum(),
				value::PtrKind::Decl(decl) => {
					let decl_value = self.compilation_unit.decls.with_mut(|decls| {
						let DeclAnalysisState::Analysed { value } = decls[decl].analysis_state else {
							unreachable!("decl-backed constant pointer references an unanalyzed decl")
						};
						value
					});
					let decl_storage = self.lower_interned_value_in_const_storage(decl_value);
					let pointee_ty = self.compilation_unit.values.index_to_key(p.ty).as_type_ptr().pointee_ty;
					let decl_value_ty = self.compilation_unit.values.type_of_interned(decl_value);
					if decl_value_ty == pointee_ty {
						decl_storage.as_any_value_enum()
					} else if let value::Key::Type(value::Type::Array(array)) = self.compilation_unit.values.index_to_key(decl_value_ty)
						&& array.elem_ty == pointee_ty
					{
						let array_ty = self.lower_type_basic(decl_value_ty);
						let zero = self.ctx.i32_type().const_zero();
						// SAFETY: the global stores an array constant of `array_ty`; indexing [0, 0]
						// yields the address of the first element used as the slice backing pointer.
						unsafe {
							PointerValue::new(llvm_sys::core::LLVMConstInBoundsGEP2(
								array_ty.as_type_ref(),
								decl_storage.as_value_ref(),
								[zero.as_value_ref(), zero.as_value_ref()].as_mut_ptr(),
								2,
							))
						}
						.as_any_value_enum()
					} else {
						unreachable!("decl-backed constant pointer type mismatch: {p:?}");
					}
				},
				value::PtrKind::ComptimeAlloc(_) => unreachable!("{p:?}"),
			},
			value::Key::EnumTag { val: v, .. } => self.lower_interned_value_as_llvm_value(*v),
			value::Key::Fn(fun) => self.lower_decl_fn(fun.owner_decl).as_any_value_enum(),
			value::Key::NullPtr => self.ctx.ptr_type(AddressSpace::default()).const_null().as_any_value_enum(),
			value::Key::Aggregate { ty, values } => {
				let values = values
					.iter()
					.map(|value| self.lower_interned_value_as_llvm_value(*value).try_into().unwrap())
					.collect::<Vec<BasicValueEnum>>();
				let (value::Key::Type(ty_key), value) = self.compilation_unit.values.index_to_key_value(*ty) else {
					unreachable!("aggregate value has a non-type type")
				};
				match ty_key {
					value::Type::Struct(_) => {
						let value::Value::Struct(r#struct) = value else {
							unreachable!("struct type without struct value")
						};
						if r#struct.as_ref().is_packed() {
							todo!()
						}
						let struct_ty_llvm = self.lower_type(*ty).into_struct_type();
						let nominal_fields = struct_ty_llvm.get_field_types();
						let actual_fields = values.iter().map(BasicValueEnum::get_type).collect::<Vec<_>>();
						let value = if nominal_fields == actual_fields {
							struct_ty_llvm.const_named_struct(&values)
						} else {
							self.ctx.struct_type(&actual_fields, false).const_named_struct(&values)
						};
						value.as_any_value_enum()
					},
					value::Type::Array(array) => {
						if self.vif_abi_type_is_by_ref(array.elem_ty) {
							let fields = values.iter().map(BasicValueEnum::get_type).collect::<Vec<_>>();
							self.ctx.struct_type(&fields, false).const_named_struct(&values).as_any_value_enum()
						} else {
							let elem_ty: BasicTypeEnum = self.lower_type_basic(array.elem_ty);
							let mut values = values.iter().map(|value| value.as_value_ref()).collect::<Vec<_>>();
							// SAFETY: every element was coerced to the array element type during sema.
							unsafe {
								BasicValueEnum::new(llvm_sys::core::LLVMConstArray2(
									elem_ty.as_type_ref(),
									values.as_mut_ptr(),
									values.len() as u64,
								))
								.as_any_value_enum()
							}
						}
					},
					value::Type::Int { .. }
					| value::Type::Anyint
					| value::Type::Anyfloat
					| value::Type::Usize
					| value::Type::Isize
					| value::Type::F16
					| value::Type::F32
					| value::Type::F64
					| value::Type::F128
					| value::Type::Bool
					| value::Type::Void
					| value::Type::Enum(_)
					| value::Type::Union(_)
					| value::Type::Fn(_)
					| value::Type::Ptr(_)
					| value::Type::Slice(_)
					| value::Type::NullPtr
					| value::Type::Any
					| value::Type::Anyptr
					| value::Type::GenericPoison
					| value::Type::Type
					| value::Type::Never
					| value::Type::EnumLiteral => unreachable!("aggregate value has a non-aggregate type"),
				}
			},
			value::Key::Union { ty, tag, payload } => {
				let union_ty = self.compilation_unit.values.index_to_value(*ty).as_union();
				let union_ty = union_ty.as_ref();
				let field_idx = if let Some(tag) = tag {
					let tag_ty = union_ty.tag_ty.expect("tagged union constant without tag type");
					union_ty
						.fields
						.iter()
						.enumerate()
						.find_map(|(field_idx, _)| {
							(self
								.compilation_unit
								.values
								.intern_enum_tag_from_field_idx(tag_ty, field_idx as u32)
								== *tag)
								.then_some(field_idx as u32)
						})
						.expect("union constant tag does not match any field")
				} else {
					let payload = payload.expect("bare union constant requires a payload");
					let payload_ty = self.compilation_unit.values.type_of_interned(payload);
					union_ty
						.fields
						.iter()
						.position(|field| field.ty == Some(payload_ty))
						.expect("union constant payload type does not match any field") as u32
				};
				match self.lower_union_repr_for_field(*ty, field_idx) {
					UnionRepr::TagOnly(tag) => tag.as_any_value_enum(),
					repr @ UnionRepr::Aggregate { ty: view_ty, .. } => {
						let payload = payload.map(|payload| self.lower_interned_value_as_llvm_value(payload).try_into().unwrap());
						let UnionReprValue {
							value: view_value,
							fields: field_values,
						} = repr.const_value(payload);
						let nominal_ty = self.lower_type(*ty).into_struct_type();
						if nominal_ty.get_field_types() == view_ty.get_field_types() {
							nominal_ty.const_named_struct(&field_values).as_any_value_enum()
						} else {
							view_value.as_any_value_enum()
						}
					},
				}
			},
			value::Key::Undefined { .. }
			| value::Key::Void
			| value::Key::Unreachable
			| value::Key::Type(_)
			| value::Key::GenericPoison { .. }
			| value::Key::DeclRef { .. }
			| value::Key::FnDecl(_)
			| value::Key::EnumLiteral(_) => {
				unreachable!("{:?} is not a value and therefore cannot be lowered into a LLVM value", key)
			},
		}
	}

	fn lower_type_basic(
		&mut self,
		index: value::Index,
	) -> inkwell::types::BasicTypeEnum<'ctx> {
		if let Some(ty) = self.interned_value_to_llvm_type.get(&index) {
			return (*ty).try_into().unwrap();
		}

		let (key, value) = self.compilation_unit.values.index_to_key_value(index);
		let ty_key = match key {
			value::Key::Type(ty) => ty,
			value::Key::Int { ty, .. } => return self.lower_type_basic(*ty),
			value::Key::Undefined { .. }
			| value::Key::Str { .. }
			| value::Key::Slice { .. }
			| value::Key::Float { .. }
			| value::Key::Bool(_)
			| value::Key::Ptr(_)
			| value::Key::Fn(_)
			| value::Key::EnumTag { .. }
			| value::Key::Aggregate { .. }
			| value::Key::NullPtr
			| value::Key::Void
			| value::Key::Unreachable
			| value::Key::Union { .. }
			| value::Key::GenericPoison { .. }
			| value::Key::DeclRef { .. }
			| value::Key::FnDecl(_)
			| value::Key::EnumLiteral(_) => unreachable!("{key:?} is not a type"),
		};
		let ty = match ty_key {
			value::Type::Int { bits, .. } => self.ctx.custom_width_int_type(*bits as u32).into(),
			value::Type::Usize | value::Type::Isize => self
				.ctx
				.ptr_sized_int_type(&self.target_machine.get_target_data(), None)
				.as_basic_type_enum(),
			value::Type::F16 => self.ctx.f16_type().as_basic_type_enum(),
			value::Type::F32 => self.ctx.f32_type().as_basic_type_enum(),
			value::Type::F64 => self.ctx.f64_type().as_basic_type_enum(),
			value::Type::F128 => self.ctx.f128_type().as_basic_type_enum(),
			value::Type::Bool => self.ctx.bool_type().as_basic_type_enum(),
			value::Type::Ptr(_) => self.ctx.ptr_type(inkwell::AddressSpace::default()).as_basic_type_enum(),
			value::Type::Slice(_) => {
				let ptr_type = self.ctx.ptr_type(inkwell::AddressSpace::default());
				let len_type = self.ctx.ptr_sized_int_type(&self.target_machine.get_target_data(), None);
				self.ctx
					.struct_type(&[ptr_type.into(), len_type.into()], false)
					.as_basic_type_enum()
			},
			value::Type::Anyptr => {
				let ptr_type = self.ctx.ptr_type(inkwell::AddressSpace::default());
				let type_id_type = self.ctx.ptr_sized_int_type(&self.target_machine.get_target_data(), None);
				self.ctx
					.struct_type(&[ptr_type.into(), type_id_type.into()], false)
					.as_basic_type_enum()
			},
			value::Type::Array(array) => {
				let elem_ty = self.lower_type_basic(array.elem_ty);
				elem_ty.array_type(array.len.try_into().unwrap()).as_basic_type_enum()
			},
			value::Type::Struct(_) => {
				let value::Value::Struct(struct_ty) = value else {
					unreachable!("struct type without struct value")
				};
				let struct_ty = struct_ty.as_ref();
				if let &value::StructLayout::Packed { storage_bits, .. } = &struct_ty.layout {
					self.ctx.custom_width_int_type(storage_bits).as_basic_type_enum()
				} else {
					let nominal_ty = self.ctx.opaque_struct_type(&struct_ty.name);
					self.interned_value_to_llvm_type.insert(index, nominal_ty.as_any_type_enum());
					let field_types = struct_ty
						.fields
						.iter()
						.map(|field| self.lower_type_basic(field.ty))
						.collect::<Vec<_>>();
					nominal_ty.set_body(&field_types, false);
					nominal_ty.as_basic_type_enum()
				}
			},
			value::Type::Enum(_) => {
				let value::Value::Enum(r#enum) = value else {
					unreachable!("enum type without enum value")
				};
				self.lower_type_basic(r#enum.tag_ty)
			},
			value::Type::Union(_) => {
				let value::Value::Union(union_ty) = value else {
					unreachable!("union type without union value")
				};
				let union_ty = union_ty.as_ref();
				let canonical_field = self
					.compilation_unit
					.values
					.type_union_layout(&self.compilation_unit.resolved_target, index)
					.most_aligned_field
					.0 as u32;
				match self.lower_union_repr_for_field(index, canonical_field) {
					UnionRepr::TagOnly(tag) => tag.get_type(),
					UnionRepr::Aggregate { ty: view_ty, .. } => {
						let nominal_ty = self.ctx.opaque_struct_type(&union_ty.name);
						self.interned_value_to_llvm_type.insert(index, nominal_ty.as_any_type_enum());
						nominal_ty.set_body(&view_ty.get_field_types(), false);
						nominal_ty.as_basic_type_enum()
					},
				}
			},
			value::Type::Anyint
			| value::Type::Anyfloat
			| value::Type::Void
			| value::Type::Fn(_)
			| value::Type::NullPtr
			| value::Type::Any
			| value::Type::GenericPoison
			| value::Type::Type
			| value::Type::Never
			| value::Type::EnumLiteral => unreachable!(
				"cannot lower type {:?} as an LLVM basic type",
				self.compilation_unit.values.index_to_key(index)
			),
		};
		self.interned_value_to_llvm_type.insert(index, ty.as_any_type_enum());
		ty
	}

	fn lower_type(
		&mut self,
		index: value::Index,
	) -> inkwell::types::AnyTypeEnum<'ctx> {
		if let Some(ty) = self.interned_value_to_llvm_type.get(&index) {
			*ty
		} else {
			let key = self.compilation_unit.values.index_to_key(index);
			let ty = match key {
				value::Key::Int { ty, .. } => self.lower_type(*ty),
				value::Key::Type(value::Type::Void | value::Type::Never) => self.ctx.void_type().into(),
				value::Key::Type(value::Type::Fn(_)) => {
					let fn_ty = self.compilation_unit.values.index_to_key(index).as_type_fn();
					let ret_repr = self.compute_fn_ret_ty_abi_repr(fn_ty);
					let sret = ret_repr == abi::Repr::ByRef;

					let mut params = Vec::<BasicMetadataTypeEnum>::with_capacity(if sret { 1 } else { 0 } + fn_ty.params.len());
					if sret {
						params.push(self.ctx.ptr_type(AddressSpace::default()).into());
					}

					for (_, &param_ty) in fn_ty.params.iter().enumerate().filter(|&(i, _)| !fn_ty.comptime_params[i]) {
						let param = match self.compute_fn_param_abi_repr(fn_ty, param_ty) {
							abi::Repr::ByValue => self.lower_type_basic(param_ty).into(),
							abi::Repr::ByRef => self.ctx.ptr_type(AddressSpace::default()).into(),
							abi::Repr::AsInteger => {
								let layout = self
									.compilation_unit
									.values
									.type_layout(&self.compilation_unit.resolved_target, param_ty);
								self.ctx.custom_width_int_type((layout.size * 8) as _).into()
							},
						};
						params.push(param);
					}

					match ret_repr {
						abi::Repr::ByRef => self.ctx.void_type().fn_type(&params, fn_ty.var_args).as_any_type_enum(),
						abi::Repr::AsInteger => {
							let layout = self
								.compilation_unit
								.values
								.type_layout(&self.compilation_unit.resolved_target, fn_ty.ret_ty);
							self.ctx
								.custom_width_int_type((layout.size * 8) as _)
								.fn_type(&params, fn_ty.var_args)
								.as_any_type_enum()
						},
						abi::Repr::ByValue => {
							let value::Key::Type(ret_ty) = self.compilation_unit.values.index_to_key(fn_ty.ret_ty) else {
								unreachable!("function return type is not a type")
							};

							// TODO(zino): i don't like this
							match ret_ty {
								value::Type::Void | value::Type::Never => {
									self.ctx.void_type().fn_type(&params, fn_ty.var_args).as_any_type_enum()
								},
								value::Type::Int { .. }
								| value::Type::Anyint
								| value::Type::Anyfloat
								| value::Type::Usize
								| value::Type::Isize
								| value::Type::F16
								| value::Type::F32
								| value::Type::F64
								| value::Type::F128
								| value::Type::Bool
								| value::Type::Struct(_)
								| value::Type::Enum(_)
								| value::Type::Union(_)
								| value::Type::Fn(_)
								| value::Type::Ptr(_)
								| value::Type::Slice(_)
								| value::Type::Array(_)
								| value::Type::NullPtr
								| value::Type::Any
								| value::Type::Anyptr
								| value::Type::GenericPoison
								| value::Type::Type
								| value::Type::EnumLiteral => self
									.lower_type_basic(fn_ty.ret_ty)
									.fn_type(&params, fn_ty.var_args)
									.as_any_type_enum(),
							}
						},
					}
				},
				value::Key::Type(
					value::Type::Int { .. }
					| value::Type::Anyint
					| value::Type::Anyfloat
					| value::Type::Usize
					| value::Type::Isize
					| value::Type::F16
					| value::Type::F32
					| value::Type::F64
					| value::Type::F128
					| value::Type::Bool
					| value::Type::Struct(_)
					| value::Type::Enum(_)
					| value::Type::Union(_)
					| value::Type::Ptr(_)
					| value::Type::Slice(_)
					| value::Type::Array(_)
					| value::Type::NullPtr
					| value::Type::Any
					| value::Type::Anyptr
					| value::Type::GenericPoison
					| value::Type::Type
					| value::Type::EnumLiteral,
				) => self.lower_type_basic(index).as_any_type_enum(),

				// not types
				value::Key::Undefined { .. }
				| value::Key::Str { .. }
				| value::Key::Slice { .. }
				| value::Key::Float { .. }
				| value::Key::Bool(_)
				| value::Key::Ptr(_)
				| value::Key::Fn(_)
				| value::Key::EnumTag { .. }
				| value::Key::Aggregate { .. }
				| value::Key::NullPtr
				| value::Key::Void
				| value::Key::Unreachable
				| value::Key::Union { .. }
				| value::Key::GenericPoison { .. }
				| value::Key::DeclRef { .. }
				| value::Key::FnDecl(_)
				| value::Key::EnumLiteral(_) => unreachable!("cannot lower {key:?} as an LLVM type"),
			};
			self.interned_value_to_llvm_type.insert(index, ty);
			ty
		}
	}

	fn lower_decl_fn(
		&mut self,
		decl: DeclId,
	) -> AnyValueEnum<'ctx> {
		let (fn_ty_idx, name, module) = self.compilation_unit.decls.with_mut(|decls| {
			let decl = &decls[decl];
			let ty = match &decl.analysis_state {
				DeclAnalysisState::TypeKnown(ty) => *ty,
				DeclAnalysisState::Analysed { value } => self.compilation_unit.values.type_of_interned(*value),
				_ => {
					unreachable!("encountered a invalid decl in codegen: {decl:?}");
				},
			};

			(ty, decl.name, decl.module)
		});

		let fn_ty = self.compilation_unit.values.index_to_key(fn_ty_idx).as_type_fn();
		let ty = self.lower_type(fn_ty_idx).into_function_type();

		// TODO(zino): handle extern declarations more explicitly.
		let is_main = &*name == "main";
		let is_runtime_main = is_main && self.compilation_unit.is_std_rt_module(module);

		// TODO(ldubos): generate a stable unique name for generic function instantiations.
		let mangled_name: String = if is_main && !is_runtime_main {
			format!("__vif_mod{}_main", usize::from(module))
		} else {
			name.to_string()
		};

		let llvm_fn_value = if fn_ty.external {
			if let Some(existing) = self.module.get_function(&mangled_name) {
				existing
			} else {
				self.module.add_function(&mangled_name, ty, Some(Linkage::External))
			}
		} else if false {
			/// todo was is generic
			if let Some(existing) = self.module.get_function(&mangled_name) {
				existing
			} else {
				self.module.add_function(&mangled_name, ty, Some(Linkage::Private))
			}
		} else {
			self.module.add_function(
				&mangled_name,
				ty,
				if is_runtime_main {
					Some(Linkage::External)
				} else {
					Some(Linkage::Private)
				},
			)
		};

		llvm_fn_value.set_call_conventions(self.llvm_callconv_id(fn_ty.callconv) as u32);

		// attributes

		// fn ret attrs
		let ret_ty = self.lower_type(fn_ty.ret_ty);
		let sret = self.fn_use_sret(fn_ty);
		if sret {
			llvm_fn_value.add_attribute(AttributeLoc::Param(0), self.attributes.sret);
			llvm_fn_value.add_attribute(AttributeLoc::Param(0), self.attributes.noalias);
			llvm_fn_value.add_attribute(AttributeLoc::Param(0), self.attributes.nonnull);
		}

		// params attrs
		for (i, &param_ty) in fn_ty.params.iter().enumerate().filter(|&(i, _)| !fn_ty.comptime_params[i]) {
			let llvm_param_idx = if sret { 1 } else { 0 } + i as u32;
			match self.compute_fn_param_abi_repr(fn_ty, param_ty) {
				abi::Repr::ByRef => {
					let param_ty_llvm = self.lower_type(param_ty);
					let byval_attr = self
						.ctx
						.create_type_attribute(self.attributes.byval.get_enum_kind_id(), param_ty_llvm);
					llvm_fn_value.add_attribute(AttributeLoc::Param(llvm_param_idx), byval_attr);
					llvm_fn_value.add_attribute(AttributeLoc::Param(llvm_param_idx), self.attributes.noalias);
					llvm_fn_value.add_attribute(AttributeLoc::Param(llvm_param_idx), self.attributes.nonnull);
				},
				abi::Repr::ByValue | abi::Repr::AsInteger => {},
			}
		}

		// fn attrs
		llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.nounwind);
		if fn_ty.ret_ty == self.compilation_unit.values.common.never_t {
			llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.noreturn);
		}
		match fn_ty.inline {
			ast::Inline::Always => {
				llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.alwaysinline);
			},
			ast::Inline::Never => {
				llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.noinline);
			},
			ast::Inline::None => {},
		}

		// for windows debugging we need the unwind table
		llvm_fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, self.attributes.uwtable_sync);

		llvm_fn_value.into()
	}

	pub fn lower_function(
		&mut self,
		compilation_unit: &CompilationUnit,
		fun: value::Index,
		vtir: &Vtir,
		build_opts: &Build,
	) {
		let module = {
			let fun = self.compilation_unit.values.index_to_key(fun).as_fn();
			self.compilation_unit.decls.lock()[fun.owner_decl].module
		};

		let di = self.di.take();

		let mut fn_lower_ctx = FnLowerCtx {
			compilation_unit,
			module,
			vtir_inst_to_llvm_value: Default::default(),
			cur_fn: None,
			cur_llvm_fn_param_idx: 0,
			di_gen: di.map(|di| DebugInfoGen {
				di_file: di.module_to_file[&module],
				di_ctx: di,
				di_lexical_block_stack: vec![],
			}),
			vtir_block_to_break_list: Default::default(),
			cur_fn_ty: self.compilation_unit.values.common.void_t,
			cur_fn_args: Vec::default(),
			lowerer: self,
		};
		fn_lower_ctx.lower_fn_body(fun, vtir).unwrap();

		self.di = fn_lower_ctx.di_gen.map(|di| di.di_ctx);
	}

	fn lower_union_repr_for_field(
		&mut self,
		union_ty: value::Index,
		field_idx: u32,
	) -> UnionRepr<'ctx> {
		let union_val_ty = self.compilation_unit.values.index_to_value(union_ty).as_union();
		let union_val_ty = union_val_ty.as_ref();

		let layout = self
			.compilation_unit
			.values
			.type_union_layout(&self.compilation_unit.resolved_target, union_ty);

		let tag = union_val_ty.tag_ty.map(|tag_ty| {
			let tag = self.compilation_unit.values.intern_enum_tag_from_field_idx(tag_ty, field_idx);
			let tag: BasicValueEnum = self.lower_interned_value(tag).try_into().unwrap();
			tag
		});
		if layout.payload.size == 0 {
			return UnionRepr::TagOnly(tag.expect("union without payload storage must have a tag"));
		}

		let active_payload_ty = union_val_ty.fields[field_idx as usize].ty;
		let payload_ty = active_payload_ty.unwrap_or_else(|| union_val_ty.fields[layout.most_aligned_field.0].ty.unwrap());
		let payload_llvm_ty: BasicTypeEnum = self.lower_type_basic(payload_ty);
		let payload_layout = self
			.compilation_unit
			.values
			.type_layout(&self.compilation_unit.resolved_target, payload_ty);

		let payload_wrapper_ty = if payload_layout.size == layout.payload.size {
			None
		} else {
			let padding = self
				.ctx
				.i8_type()
				.array_type((layout.payload.size - payload_layout.size).try_into().unwrap());
			Some(self.ctx.struct_type(&[payload_llvm_ty, padding.into()], true))
		};
		let payload_view_ty = payload_wrapper_ty.map(BasicTypeEnum::from).unwrap_or(payload_llvm_ty);

		let tag_first = layout.tag.size != 0 && layout.tag.align >= layout.payload.align;
		let tag_field_idx = (layout.tag.size != 0).then_some(u32::from(!tag_first));
		let payload_field_idx = u32::from(tag_first);

		let mut view_fields = Vec::with_capacity(4);
		if tag_first {
			view_fields.push(self.lower_type_basic(union_val_ty.tag_ty.unwrap()));
		}
		view_fields.push(payload_view_ty);
		if layout.tag.size != 0 && !tag_first {
			view_fields.push(self.lower_type_basic(union_val_ty.tag_ty.unwrap()));
		}
		if layout.trailing_padding != 0 {
			view_fields.push(
				self.ctx
					.i8_type()
					.array_type(layout.trailing_padding.try_into().unwrap())
					.as_basic_type_enum(),
			);
		}
		if payload_wrapper_ty.is_some() {
			let aligned_field_ty = union_val_ty.fields[layout.most_aligned_field.0]
				.ty
				.expect("most-aligned union field");
			let aligned_field_ty: BasicTypeEnum = self.lower_type_basic(aligned_field_ty);
			view_fields.push(aligned_field_ty.array_type(0).as_basic_type_enum());
		}

		let ty = self
			.ctx
			.opaque_struct_type(&format!("{}.{}", union_val_ty.name, union_val_ty.fields[field_idx as usize].name));
		ty.set_body(&view_fields, false);

		UnionRepr::Aggregate {
			ty,
			tag: tag.zip(tag_field_idx),
			payload: active_payload_ty.map(|_| (payload_field_idx, payload_wrapper_ty)),
		}
	}

	pub fn finish(
		mut self,
		build_opts: &Build,
	) -> Result<inkwell::memory_buffer::MemoryBuffer, ()> {
		let ptr_ty = self.ctx.ptr_type(AddressSpace::default());
		let mut type_info_ptrs = Vec::with_capacity(self.compilation_unit.type_info_entries.len());
		for index in 0..self.compilation_unit.type_info_entries.len() {
			let type_info = *self.compilation_unit.type_info_entries.get(TypeInfoId(index));
			type_info_ptrs.push(self.lower_interned_value_in_const_storage(type_info));
		}
		let type_info_entries = self
			.module
			.add_global(ptr_ty.array_type(type_info_ptrs.len() as u32), None, "__vif_type_info_entries");
		type_info_entries.set_linkage(Linkage::Private);
		type_info_entries.set_constant(true);
		type_info_entries.set_unnamed_address(UnnamedAddress::Global);
		type_info_entries.set_initializer(&ptr_ty.const_array(&type_info_ptrs));
		self.type_info_table.set_initializer(&type_info_entries.as_pointer_value());

		if let Some(di) = self.di {
			di.builder.finalize();
		}

		// validate and dump any LLVM IR errors
		{
			if let Err(err) = self.module.verify() {
				for err in err.to_string_lossy().split('\n') {
					eprintln!("{}", err);
				}
				self.module.print_to_stderr();
				return Err(());
			}
		}

		// Run LLVM passes to transform IR depending on opt levels
		let pass_opts = PassBuilderOptions::create();
		pass_opts.set_loop_unrolling(true);
		pass_opts.set_loop_vectorization(true);
		pass_opts.set_loop_slp_vectorization(true);

		self.module
			.run_passes(
				match build_opts.opt {
					0 => "default<O0>",
					1 => "default<O1>",
					2 => "default<O2>",
					3 => "default<O3>",
					_ => unreachable!(),
				},
				&self.target_machine,
				PassBuilderOptions::create(),
			)
			.unwrap();

		if build_opts.dump_llvm_ir {
			println!("--- START OF LLVM IR DUMP ---");
			self.module.print_to_stderr();
			println!("--- END OF LLVM IR DUMP ---");
		}

		if build_opts.dump_asm {
			println!("--- START OF ASM DUMP ---");
			let mb = self
				.target_machine
				.write_to_memory_buffer(&self.module, FileType::Assembly)
				.unwrap();
			println!("{}", String::from_utf8_lossy(mb.as_slice()).into_owned());
			println!("--- END OF ASM DUMP ---");
		}

		Ok(self.target_machine.write_to_memory_buffer(&self.module, FileType::Object).unwrap())
	}

	fn compute_fn_param_abi_repr(
		&self,
		fn_ty: &value::TypeFn,
		param_ty: value::Index,
	) -> abi::Repr {
		match fn_ty.callconv {
			// TODO(zino): cold, fast
			CallingConvention::Vif | CallingConvention::Cold | CallingConvention::Fast => {
				if self.vif_abi_type_is_by_ref(param_ty) {
					abi::Repr::ByRef
				} else {
					abi::Repr::ByValue
				}
			},
			CallingConvention::C | CallingConvention::X86_64Windows => abi::compute_type_abi_win64(self.compilation_unit, param_ty),

			CallingConvention::Count => unreachable!(),
		}
	}

	fn compute_fn_ret_ty_abi_repr(
		&self,
		fn_ty: &value::TypeFn,
	) -> abi::Repr {
		match fn_ty.callconv {
			// TODO(zino): cold, fast
			CallingConvention::Vif | CallingConvention::Cold | CallingConvention::Fast => {
				if self.vif_abi_type_is_by_ref(fn_ty.ret_ty) {
					abi::Repr::ByRef
				} else {
					abi::Repr::ByValue
				}
			},
			CallingConvention::C | CallingConvention::X86_64Windows => abi::compute_type_abi_win64(self.compilation_unit, fn_ty.ret_ty),
			CallingConvention::Cold | CallingConvention::Fast => todo!(),

			CallingConvention::Count => unreachable!(),
		}
	}

	fn fn_use_sret(
		&self,
		fn_ty: &value::TypeFn,
	) -> bool {
		// TODO(zino): instead of the underlying type check we should have a fn that checks a type has a repr
		fn_ty.ret_ty != self.compilation_unit.values.common.void_t && matches!(self.compute_fn_ret_ty_abi_repr(fn_ty), abi::Repr::ByRef)
	}
}

enum UnionRepr<'ctx> {
	/// Only store a tag
	TagOnly(BasicValueEnum<'ctx>),

	/// An aggregate of an optional tag and optional payload depending on the field and union type (if it is tagged or not)
	Aggregate {
		ty: StructType<'ctx>,
		tag: Option<(BasicValueEnum<'ctx>, u32)>,
		payload: Option<(u32, Option<StructType<'ctx>>)>,
	},
}

impl<'ctx> UnionRepr<'ctx> {
	fn const_value(
		&self,
		payload_value: Option<BasicValueEnum<'ctx>>,
	) -> UnionReprValue<'ctx> {
		let UnionRepr::Aggregate { ty, tag, payload } = self else {
			unreachable!("tag-only union is already a constant");
		};
		let mut values = ty.get_field_types().into_iter().map(|ty| ty.const_zero()).collect::<Vec<_>>();

		if let Some((payload_field_idx, payload_wrapper_ty)) = payload {
			let payload_value = payload_value.expect("union payload constant is missing its value");
			values[*payload_field_idx as usize] = if let Some(wrapper_ty) = payload_wrapper_ty {
				let padding = wrapper_ty.get_field_types()[1].const_zero();
				wrapper_ty.const_named_struct(&[payload_value, padding]).as_basic_value_enum()
			} else {
				payload_value
			};
		} else {
			assert!(payload_value.is_none(), "payload-less union constant has a payload");
		}

		if let Some((tag, tag_field_idx)) = tag {
			values[*tag_field_idx as usize] = *tag;
		}

		UnionReprValue {
			value: ty.const_named_struct(&values),
			fields: values,
		}
	}
}

struct UnionReprValue<'ctx> {
	value: inkwell::values::StructValue<'ctx>,
	fields: Vec<BasicValueEnum<'ctx>>,
}
