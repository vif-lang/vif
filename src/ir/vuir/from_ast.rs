use std::{
	hash::DefaultHasher,
	ops::{
		ControlFlow,
		Deref,
	},
	pin::Pin,
	ptr::NonNull,
};

use bitvec::vec::BitVec;
use bumpalo::Bump;
use hashbrown::DefaultHashBuilder;
use internment::Intern;
use rustc_hash::{
	FxBuildHasher,
	FxHashMap,
	FxHashSet,
};

use crate::{
	common::{
		COMMON_INTERNS,
		IndexVec,
		RcLinearAllocator,
		Span,
		diagnostic::*,
		index_map::IndexMap,
	},
	compile_unit::{
		CompilationUnit,
		Namespace,
		NamespaceId,
		module::ModuleId,
	},
	frontend::{
		Radix,
		ast,
	},
	ir::vuir::{
		self,
		NamingKind,
	},
	value::{
		self,
		Anyfloat,
		Anyint,
		Key,
		TypePtr,
		TypeSlice,
	},
};

type BumpVec<'bump, T> = bumpalo::collections::Vec<'bump, T>;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct ScopeId(usize);
impl From<ScopeId> for usize {
	fn from(value: ScopeId) -> Self {
		value.0
	}
}
impl From<usize> for ScopeId {
	fn from(value: usize) -> Self {
		Self(value)
	}
}

#[repr(transparent)]
struct StackedBlock(ScopeId);
impl StackedBlock {
	fn block<'a>(
		&self,
		lowerer: &'a Lowerer,
	) -> &'a Block {
		lowerer.scopes[self.0].as_block()
	}
}
impl Deref for StackedBlock {
	type Target = ScopeId;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
impl Drop for StackedBlock {
	fn drop(&mut self) {
		if !std::thread::panicking() {
			panic!("a StackedBlock must be dropped with unstack_block()")
		}
	}
}

struct ScopeNamespace {
	parent: Option<ScopeId>,
	decl_to_ast_node: FxHashMap<Intern<str>, (ast::NodeId, Span)>,
	/// captured values for this namespace
	captures: IndexMap<vuir::Capture, (), FxBuildHasher>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ResolveResult {
	FoundAndIsLocalStackAllocated(vuir::InstructionRef),
	FoundAndIsLocalValue(vuir::InstructionRef),
	FoundAndIsDecl,
	/// Found in an outer scope, tunnelled through at least one struct namespace.
	/// The inner `InstructionRef` is the original outer instruction.
	Capture {
		inst: vuir::InstructionRef,
		traversed_namespace_count: usize,
	},
}

/// A (G)UIR scope represent a single, distinct scope of UIR instructions. It may itself contains subscopes.
/// The most nested scope instructions starts at inst_start and finish at the end of the instruction index vec inside the Lowerer.
struct Block {
	parent: Option<ScopeId>,
	inst_start: usize,
	kind: BlockKind,
	/// cache of resolved symbols inside the block
	resolve_cache: FxHashMap<Intern<str>, Option<ResolveResult>>,
}

impl Block {
	fn instructions<'a>(
		&self,
		instructions: &'a [vuir::InstructionId],
	) -> &'a [vuir::InstructionId] {
		&instructions[self.inst_start..]
	}

	pub fn last_instruction<'a>(
		&self,
		ast2vuir: &'a Lowerer,
	) -> Option<&'a vuir::Opcode> {
		self.instructions(&ast2vuir.uir_scopes_instructions)
			.last()
			.map(|inst| &ast2vuir.instructions[*inst])
	}

	pub fn ends_with_never(
		&self,
		ast2vuir: &Lowerer,
	) -> bool {
		self.instructions(&ast2vuir.uir_scopes_instructions)
			.last()
			.is_some_and(|inst| ast2vuir.instructions[*inst].returns_never())
	}
}

enum Scope {
	/// A namespace contains a list of named declarations. This can be a module,
	/// a struct, enum ...
	Namespace(ScopeNamespace),

	/// A local stack allocated variable
	LocalStackAllocated {
		parent: ScopeId,
		name: Intern<str>,
		node: ast::NodeId,
		vuir_inst: vuir::InstructionRef,
	},

	/// A local value such as a function parameter
	LocalValue {
		parent: ScopeId,
		name: Intern<str>,
		node: ast::NodeId,
		vuir_inst: vuir::InstructionRef,

		/// If Some and this local value is resolved at least once, it'll write true
		/// SAFETY: if non null, the pointer must live as long as the scope
		unsafe was_resolved_atleast_once: *mut bool,
	},

	Block(Block),
	Defer {
		parent: ScopeId,
		body: &'static [vuir::InstructionId],
		span: Span,
	},
}
impl Scope {
	fn parent(&self) -> Option<ScopeId> {
		match self {
			Scope::LocalStackAllocated { parent, .. } => Some(*parent),
			Scope::LocalValue { parent, .. } => Some(*parent),
			Scope::Namespace(ns) => ns.parent,
			Scope::Block(block) => block.parent,
			Scope::Defer { parent, .. } => Some(*parent),
		}
	}

	fn as_block(&self) -> &Block {
		match self {
			Scope::Block(block) => block,
			_ => unreachable!(),
		}
	}

	fn as_block_mut(&mut self) -> &mut Block {
		match self {
			Scope::Block(block) => block,
			_ => unreachable!(),
		}
	}

	fn as_namespace_mut(&mut self) -> &mut ScopeNamespace {
		match self {
			Scope::Namespace(ns) => ns,
			_ => unreachable!(),
		}
	}
}

#[derive(Copy, Clone, Debug)]
enum ExprResultLocation {
	None,

	CoerceToTy(vuir::InstructionRef),

	/// The RHS expression must store its result to this typed pointer
	StoreToPtr {
		ptr: vuir::InstructionRef,
		span: Span,
	},
	StoreToInferredPtr {
		ptr: vuir::InstructionRef,
		span: Span,
	},

	/// LHS want to get the address of the RHS
	GetAddressOf,
}
impl ExprResultLocation {
	pub fn into_inst(self) -> Option<vuir::InstructionRef> {
		match self {
			Self::CoerceToTy(ty) => Some(ty),
			Self::StoreToPtr { ptr, .. } => Some(ptr),
			Self::StoreToInferredPtr { .. } => None,
			Self::GetAddressOf | Self::None => None,
			_ => unreachable!("{:?}", self),
		}
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
enum BlockKind {
	/// A block without any control flow
	Body,
	Block {
		inst: vuir::InstructionRef,
		label: Option<ast::Ident>,
	},
	Branch(vuir::InstructionRef),
	Loop {
		block_inst: vuir::InstructionRef,
		label: Option<ast::Ident>,
	},
}

struct Lowerer<'ast> {
	cu: &'ast CompilationUnit,
	src: &'ast str,
	module_id: ModuleId,
	ast: &'ast ast::Module,
	/// Flatten array of instructions per in stack UIR scopes
	uir_scopes_instructions: Vec<vuir::InstructionId>,
	instructions: IndexVec<vuir::InstructionId, vuir::Opcode>,
	instructions_payload_alloc: &'static Bump,
	scopes: IndexVec<ScopeId, Scope>,
	errors: Vec<Diagnostic>,
	// Precomputed line starts for fast (line, col) mapping
	line_starts: Vec<usize>,
	imports: BumpVec<'static, vuir::Import>,

	/// If in a function, points to the root scope
	fn_body_root_scope: Option<ScopeId>,
}

impl<'ast> Lowerer<'ast> {
	#[inline(always)]
	fn diag_span(
		&self,
		span: Span,
	) -> DiagSpan {
		DiagSpan {
			module: self.module_id,
			span,
		}
	}

	#[inline(always)]
	fn start_line_col(
		&self,
		span: Span,
	) -> (usize, usize) {
		let offset = span.start();
		let idx = match self.line_starts.binary_search(&offset) {
			Ok(i) => i,
			Err(i) => i.saturating_sub(1),
		};
		let line = idx + 1;
		let col = offset - self.line_starts[idx];
		(line, col)
	}

	#[inline(always)]
	fn inst_invalid(
		&mut self,
		block_scope: ScopeId,
	) -> vuir::InstructionRef {
		self.inst(block_scope, vuir::Opcode::Invalid)
	}

	#[inline(always)]
	fn inst(
		&mut self,
		block_scope: ScopeId,
		inst: vuir::Opcode,
	) -> vuir::InstructionRef {
		self.inst_id(block_scope, inst).as_ref()
	}

	#[inline(always)]
	fn inst_id(
		&mut self,
		_block_scope: ScopeId,
		inst: vuir::Opcode,
	) -> vuir::InstructionId {
		self.instructions.push(inst);
		let id = vuir::InstructionId::from_usize(self.instructions.len() - 1);
		self.uir_scopes_instructions.push(id);
		id
	}

	fn stack_block(
		&mut self,
		parent: ScopeId,
		kind: BlockKind,
	) -> StackedBlock {
		self.scopes.push(Scope::Block(Block {
			parent: Some(parent),
			inst_start: self.uir_scopes_instructions.len(),
			kind,
			resolve_cache: FxHashMap::with_capacity_and_hasher(16, Default::default()),
		}));
		StackedBlock(ScopeId(self.scopes.len() - 1))
	}

	/// Close the scope
	fn unstack_block(
		&mut self,
		block: StackedBlock,
	) {
		let block = {
			let id = block.0;
			core::mem::forget(block);
			id
		};

		// unstack scopes until we encounter the block, if we encounter a different block this is instant ICE
		let block = loop {
			let Some(scope) = self.scopes.pop() else {
				unreachable!();
			};
			if let Scope::Block(unstacked_block) = scope {
				if ScopeId(self.scopes.len()) == block {
					break unstacked_block;
				} else {
					panic!("while unstacking a block we encountered a different block")
				}
			}
		};

		// and remove range from block insts
		debug_assert!(block.inst_start <= self.uir_scopes_instructions.len());
		// SAFETY: inst_start is smaller or equal than the current length
		unsafe { self.uir_scopes_instructions.set_len(block.inst_start) };
	}

	/// Helper to map over the instructions and then unstack the scope
	fn map_and_unstack_block<R>(
		&mut self,
		block: StackedBlock,
		f: impl FnOnce(&[vuir::InstructionId]) -> R,
	) -> R {
		let res = {
			let block = self.scopes[*block].as_block();
			f(block.instructions(&self.uir_scopes_instructions))
		};
		self.unstack_block(block);
		res
	}

	fn collect_instructions_and_unstack_block(
		&mut self,
		block: StackedBlock,
	) -> &'static [vuir::InstructionId] {
		let instructions = {
			let block = self.scopes[*block].as_block();
			let mut instructions = BumpVec::new_in(self.instructions_payload_alloc);
			block
				.instructions(&self.uir_scopes_instructions)
				.iter()
				.collect_into(&mut instructions);
			instructions
		};
		self.unstack_block(block);
		instructions.into_bump_slice()
	}

	/// Append all defers we found throughout the scope stack until we reach `stop_scope`
	fn append_defers(
		&mut self,
		start_scope: ScopeId,
		stop_scope: ScopeId,
	) {
		self.traverse_scopes_from_mut(start_scope, |this, scope| match &this.scopes[scope] {
			Scope::Defer { body, span, .. } => {
				this.inst(scope, vuir::Opcode::Defer { body, span: *span });
				ControlFlow::Continue(())
			},
			_ => {
				if scope == stop_scope {
					ControlFlow::Break(())
				} else {
					ControlFlow::Continue(())
				}
			},
		});
	}
}

impl<'ast> Lowerer<'ast> {
	/// Like Zig AstGen rvalue() this apply r-value semantics to an expression in the case it was present in the RHS of a binary
	/// expression or a statement. This is to be called in every expression lowering that doesn't explicitly handle it.
	#[inline(always)]
	fn rvalue(
		&mut self,
		block_scope: ScopeId,
		inst: vuir::InstructionRef,
		rhs_ctx: ExprResultLocation,
		span: Span,
	) -> vuir::InstructionRef {
		match rhs_ctx {
			ExprResultLocation::StoreToPtr { ptr, span: ptr_span } => self.inst(block_scope, vuir::Opcode::Store {
				dst: ptr,
				src: inst,
				span: (span, ptr_span).into(),
			}),
			ExprResultLocation::StoreToInferredPtr { ptr, span: ptr_span } => {
				let _ = self.inst(block_scope, vuir::Opcode::StoreToInferredAlloc {
					dst: ptr,
					src: inst,
					span: (span, ptr_span).into(),
				});
				vuir::InstructionRef::Interned(self.cu.values.common.void_value)
			},
			ExprResultLocation::GetAddressOf => self.inst(block_scope, vuir::Opcode::RvalueToLvalue { rvalue: inst }),
			ExprResultLocation::CoerceToTy(ty) => self.inst(block_scope, vuir::Opcode::Coerce {
				value: inst,
				into: ty,
				span,
			}),
			_ => inst,
		}
	}

	fn resolve_symbol(
		&mut self,
		block_scope: ScopeId,
		symbol: Intern<str>,
	) -> Option<ResolveResult> {
		// does it begin with a @ ? then it is a builtin from the prelude
		if symbol.starts_with('@') {
			return Some(ResolveResult::FoundAndIsDecl);
		}

		let mut traversed_namespace_count = 0;

		// SAFETY: `traverse_scopes_from_mut` yields valid scope ids from `self.scopes`.
		self.traverse_scopes_from_mut(block_scope, |this, scope| unsafe {
			match &mut this.scopes[scope] {
				Scope::LocalValue {
					name,
					vuir_inst,
					was_resolved_atleast_once,
					..
				} if *name == symbol => {
					if !was_resolved_atleast_once.is_null() {
						// SAFETY: caller must ensure ptr is valid throughout the scope duration
						unsafe { was_resolved_atleast_once.write(true) }
					}

					let r = if traversed_namespace_count > 0 {
						ResolveResult::Capture {
							inst: *vuir_inst,
							traversed_namespace_count,
						}
					} else {
						ResolveResult::FoundAndIsLocalValue(*vuir_inst)
					};
					ControlFlow::Break(r)
				},
				Scope::LocalStackAllocated { name, vuir_inst, .. } if *name == symbol => {
					let r = if traversed_namespace_count > 0 {
						ResolveResult::Capture {
							inst: *vuir_inst,
							traversed_namespace_count,
						}
					} else {
						ResolveResult::FoundAndIsLocalStackAllocated(*vuir_inst)
					};
					ControlFlow::Break(r)
				},
				Scope::Namespace(ns) => {
					traversed_namespace_count += 1;
					if symbol == COMMON_INTERNS.self_ty_symbol || ns.decl_to_ast_node.contains_key(&symbol) {
						ControlFlow::Break(ResolveResult::FoundAndIsDecl)
					} else {
						ControlFlow::Continue(())
					}
				},
				_ => ControlFlow::Continue(()),
			}
		})
	}

	fn capture_outer_scope_inst(
		&mut self,
		block_scope: ScopeId,
		outer_ref: vuir::InstructionRef,
		traversed_namespace_count: usize,
		span: Span,
	) -> vuir::InstructionRef {
		let outer_ref = match outer_ref {
			vuir::InstructionRef::Instruction(id) => id,
			other => return other, // no need to capture, we already know at vuir time the value
		};

		// find the namespace of the outer and all namespaces we traversed (in top -> bottom order)
		// we don't do that in resolve_symbol because in the majority of cases we don't have a captured value so we don't need to
		// pay the cost of maintaining the stack of traversed namespace
		let (root_namespace, intermediate_namespaces) = {
			let mut namespaces = std::iter::successors(self.scopes[block_scope].parent(), |scope| self.scopes[scope].parent())
				.filter(|scope| matches!(self.scopes[scope], Scope::Namespace(_)));

			let mut intermediate_namespaces: Vec<ScopeId> = namespaces.by_ref().take(traversed_namespace_count - 1).collect::<Vec<_>>();

			// reverse to top -> bottom order
			intermediate_namespaces.reverse();

			// root is the next namespace past the intermediates
			let outer_namespace = namespaces.next().expect("scope chain must contain root namespace");

			(outer_namespace, intermediate_namespaces)
		};

		// start inserting value capture from the root until we get to our namespace
		let mut capture_idx = self.scopes[root_namespace]
			.as_namespace_mut()
			.captures
			.entry(&vuir::Capture::Id(outer_ref))
			.or_insert_with(|| ());

		for intermediate in intermediate_namespaces {
			capture_idx = self.scopes[intermediate]
				.as_namespace_mut()
				.captures
				.entry(&vuir::Capture::FromParent(capture_idx))
				.or_insert_with(|| ());
		}

		self.inst(block_scope, vuir::Opcode::CaptureGet { idx: capture_idx, span })
	}

	/// CHeck if `label` is already defined, if that's the case it'll push a diagnostic and return true
	fn check_label_already_defined(
		&mut self,
		scope: ScopeId,
		label: ast::Ident,
	) -> bool {
		let existing = self.traverse_scopes_from(scope, |scope| match &self.scopes[scope] {
			Scope::Block(block) => {
				let block_label = match block.kind {
					BlockKind::Loop { label, .. } => label,
					BlockKind::Block { label, .. } => label,
					BlockKind::Branch(..) | BlockKind::Body => None,
				};

				if let Some(block_label) = block_label
					&& block_label.symbol == label.symbol
				{
					ControlFlow::Break(block_label)
				} else {
					ControlFlow::Continue(())
				}
			},
			_ => ControlFlow::Continue(()),
		});

		if let Some(existing) = existing {
			self.errors.push(
				Diagnostic::error()
					.with_message(format!("label `:{}` already defined", label.symbol))
					.with_label(
						Label::primary()
							.with_span(self.diag_span(label.span))
							.with_message("redefinition here"),
					)
					.with_label(
						Label::primary()
							.with_span(self.diag_span(existing.span))
							.with_message("first defined here"),
					),
			);
			true
		} else {
			false
		}
	}

	fn lower_field_access(
		&mut self,
		block_scope: ScopeId,
		field_access: &ast::FieldAccess,
		rhs_ctx: ExprResultLocation,
	) -> vuir::InstructionRef {
		match rhs_ctx {
			ExprResultLocation::GetAddressOf => {
				self.lower_field_access_no_rhs_ctx_handling(block_scope, field_access, ExprResultLocation::GetAddressOf, false)
			},
			rhs_ctx => {
				// every other case, assume we want the value therefore perform a field ptr load
				let field_val =
					self.lower_field_access_no_rhs_ctx_handling(block_scope, field_access, ExprResultLocation::GetAddressOf, true);
				self.rvalue(block_scope, field_val, rhs_ctx, field_access.span)
			},
		}
	}

	#[inline]
	fn lower_field_access_no_rhs_ctx_handling(
		&mut self,
		block_scope: ScopeId,
		field_access: &ast::FieldAccess,
		rhs_ctx: ExprResultLocation,
		loads_field: bool,
	) -> vuir::InstructionRef {
		let lhs = self.lower_expr(block_scope, field_access.lhs, rhs_ctx);
		self.inst(
			block_scope,
			if loads_field {
				vuir::Opcode::FieldValFromPtr {
					lhs,
					field: field_access.field.symbol,
					span: field_access.field.span,
				}
			} else {
				vuir::Opcode::FieldPtrFromPtr {
					lhs,
					field: field_access.field.symbol,
					span: field_access.field.span,
				}
			},
		)
	}

	fn lower_ident(
		&mut self,
		block_scope: ScopeId,
		ident: &ast::Ident,
		rhs_ctx: ExprResultLocation,
		span: Span,
	) -> Option<vuir::InstructionRef> {
		let symbol = ident.symbol;
		let resolved = self.resolve_symbol(block_scope, symbol);

		if let Some(result) = resolved {
			let inst = match result {
				ResolveResult::FoundAndIsLocalStackAllocated(vuir) => match rhs_ctx {
					ExprResultLocation::GetAddressOf => vuir,
					_ => {
						let loaded = self.inst(block_scope, vuir::Opcode::Load { src: vuir, span });
						self.rvalue(block_scope, loaded, rhs_ctx, span)
					},
				},
				ResolveResult::FoundAndIsLocalValue(vuir) => self.rvalue(block_scope, vuir, rhs_ctx, span),
				ResolveResult::FoundAndIsDecl => {
					let val = match rhs_ctx {
						ExprResultLocation::GetAddressOf => return Some(self.inst(block_scope, vuir::Opcode::DeclRef(*ident))),
						_ => self.inst(block_scope, vuir::Opcode::DeclVal(*ident)),
					};
					self.rvalue(block_scope, val, rhs_ctx, span)
				},
				ResolveResult::Capture {
					inst,
					traversed_namespace_count,
				} => self.capture_outer_scope_inst(block_scope, inst, traversed_namespace_count, span),
			};
			Some(inst)
		} else {
			self.errors.push(
				Diagnostic::error()
					.with_message(format!("unknown name '{}'", symbol))
					.with_label(Label::primary().with_span(self.diag_span(span)))
					.with_note("ensure name is visible in the current scope"),
			);
			None
		}
	}

	fn lower_type(
		&mut self,
		block_scope: ScopeId,
		ty: &ast::Type,
		naming: vuir::NamingKind,
		span: Span,
	) -> vuir::InstructionRef {
		match ty {
			ast::Type::Void => self.cu.values.common.void_t.into(),
			ast::Type::Int(i) => vuir::InstructionRef::Interned(match i {
				ast::IntSuffix::U(bits) => self.cu.values.intern_trivial(&value::Key::TypeInt {
					signed: false,
					bits: *bits,
				}),
				ast::IntSuffix::I(bits) => self.cu.values.intern_trivial(&value::Key::TypeInt { signed: true, bits: *bits }),
				ast::IntSuffix::Usize => self.cu.values.common.usize_t,
				ast::IntSuffix::Isize => self.cu.values.common.isize_t,
			}),
			ast::Type::Float(f) => vuir::InstructionRef::Interned(match f {
				ast::FloatSuffix::F16 => self.cu.values.common.f16_t,
				ast::FloatSuffix::F32 => self.cu.values.common.f32_t,
				ast::FloatSuffix::F64 => self.cu.values.common.f64_t,
				ast::FloatSuffix::F128 => self.cu.values.common.f128_t,
			}),
			ast::Type::Bool => self.cu.values.common.bool_t.into(),
			ast::Type::Any => self.cu.values.common.any_t.into(),
			ast::Type::Type => self.cu.values.common.type_t.into(),
			ast::Type::Anyint => self.cu.values.common.anyint_t.into(),
			ast::Type::Anyfloat => self.cu.values.common.anyfloat_t.into(),
			ast::Type::Never => self.cu.values.common.never_t.into(),
			ast::Type::Ptr { ty, modifiers } => {
				let pointee = self.lower_expr(block_scope, ty, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::TypePtr {
					pointee,
					is_const: modifiers.is_const,
					is_volatile: modifiers.is_volatile,
					span,
				})
			},
			ast::Type::ManyPtr { ty, modifiers, sentinel } => {
				todo!()
			},
			ast::Type::Slice { ty, modifiers, sentinel } => {
				let elem = self.lower_expr(block_scope, ty, ExprResultLocation::None);
				let sentinel = sentinel.map(|s| self.lower_expr(block_scope, s, ExprResultLocation::None));
				self.inst(block_scope, vuir::Opcode::TypeSlice {
					elem,
					is_const: modifiers.is_const,
					sentinel,
				})
			},
			ast::Type::Array {
				ty,
				is_const,
				size,
				sentinel,
			} => {
				let elem = self.lower_expr(block_scope, ty, ExprResultLocation::None);
				let len = size
					.map(|size| self.lower_expr(block_scope, size, ExprResultLocation::None))
					.unwrap();

				let sentinel = sentinel.map(|s| self.lower_expr(block_scope, s, ExprResultLocation::None));
				self.inst(block_scope, vuir::Opcode::TypeArray {
					elem,
					is_const: *is_const,
					len,
					sentinel,
					elem_span: ty.span,
					len_span: size.unwrap().span,
					span,
				})
			},
			ast::Type::Nullable(inner) => {
				todo!()
			},
			ast::Type::ErrorUnion { err_ty, ok_ty } => {
				todo!()
			},
			ast::Type::Fn(fn_sig) => {
				todo!()
			},
			ast::Type::Generic => self.cu.values.common.any_t.into(),
			ast::Type::Struct(r#struct) => self.lower_struct(block_scope, r#struct, naming).into_ref(),
			ast::Type::Enum(r#enum) => self.lower_enum(block_scope, r#enum, naming).into_ref(),
			ast::Type::Union(u) => self.lower_union(block_scope, u, naming).into_ref(),
			ast::Type::Error(_) => todo!(),
			ast::Type::Anyerror => todo!(),
		}
	}

	fn lower_var_binding(
		&mut self,
		block_scope: ScopeId,
		binding: &'ast ast::VarBinding,
		is_comptime: bool,
		is_mutable: bool,
	) -> (vuir::InstructionRef, ScopeId) {
		let (var, rhs_ctx, is_comptime) = if let Some(ty) = binding.ty {
			let ty = self.wrap_in_comptime_block(block_scope, |this, block| {
				this.lower_expr(block_scope, ty, ExprResultLocation::None)
			});
			let (var, is_comptime) = match (is_comptime, is_mutable) {
				(false, false) => (
					self.inst_id(block_scope, vuir::Opcode::StackAlloc {
						name: binding.name,
						ty,
						span: binding.name.span,
					}),
					false,
				),
				(false, true) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocMut {
						name: binding.name,
						ty,
						span: binding.name.span,
					}),
					false,
				),
				(true, false) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocComptime {
						name: binding.name,
						ty,
						span: binding.name.span,
					}),
					true,
				),
				(true, true) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocComptimeMut {
						name: binding.name,
						ty,
						span: binding.name.span,
					}),
					true,
				),
			};
			(
				var,
				ExprResultLocation::StoreToPtr {
					ptr: var.as_ref(),
					span: binding.name.span,
				},
				is_comptime,
			)
		} else {
			let (var, is_comptime) = match (is_comptime, is_mutable) {
				(false, false) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocInferred {
						name: binding.name,
						span: binding.name.span,
					}),
					false,
				),
				(false, true) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocInferredMut {
						name: binding.name,
						span: binding.name.span,
					}),
					false,
				),
				(true, false) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocInferredComptime {
						name: binding.name,
						span: binding.name.span,
					}),
					true,
				),
				(true, true) => (
					self.inst_id(block_scope, vuir::Opcode::StackAllocInferredComptimeMut {
						name: binding.name,
						span: binding.name.span,
					}),
					true,
				),
			};
			(
				var,
				ExprResultLocation::StoreToInferredPtr {
					ptr: var.as_ref(),
					span: binding.name.span,
				},
				is_comptime,
			)
		};

		// init
		if is_comptime {
			let name = binding.name.symbol;
			let _ = self.wrap_in_comptime_block(block_scope, |this, block| {
				this.lower_expr_named(block_scope, binding.val, rhs_ctx, NamingKind::Named(name))
			});
		} else {
			let _ = self.lower_expr_named(block_scope, binding.val, rhs_ctx, NamingKind::FromPreviousStackAlloc);
		}

		// if inferred, directly reify the allocation
		let var = if let ExprResultLocation::StoreToInferredPtr { ptr, .. } = rhs_ctx {
			self.inst_id(block_scope, vuir::Opcode::ReifyInferredAlloc {
				alloc: ptr,
				span: binding.name.span,
			})
		} else if is_mutable {
			self.inst_id(block_scope, vuir::Opcode::FreezeStackAlloc {
				alloc: var,
				span: binding.name.span,
			})
		} else {
			var
		};

		let scope = self.scopes.push(Scope::LocalStackAllocated {
			parent: block_scope,
			name: binding.name.symbol,
			node: binding.id,
			vuir_inst: var.as_ref(),
		});

		(var.into_ref(), scope)
	}

	fn lower_item_var_binding(
		&mut self,
		block_scope: ScopeId,
		_item: &'ast ast::AssociatedItem,
		binding: &'ast ast::VarBinding,
	) -> vuir::InstructionId {
		let decl = self.inst_id(block_scope, vuir::Opcode::Invalid);

		// Decl value
		let value = {
			let scope = self.stack_block(block_scope, BlockKind::Body);
			let init_rhs_ctx = if let Some(ty) = binding.ty {
				let ty = self.lower_expr(block_scope, ty, ExprResultLocation::None);
				ExprResultLocation::CoerceToTy(ty)
			} else {
				ExprResultLocation::None
			};
			let init = self.lower_decl_init_expr(*scope, binding.val, init_rhs_ctx);
			self.inst(block_scope, vuir::Opcode::BreakComptime { block: decl, value: init });
			self.collect_instructions_and_unstack_block(scope)
		};
		self.instructions[decl] = vuir::Opcode::Declaration(vuir::Decl {
			name: binding.name.symbol,
			value,
			span: binding.name.span,
		});
		decl
	}

	/// Wraps a expression in a comptime block
	fn wrap_in_comptime_block(
		&mut self,
		block_scope: ScopeId,
		f: impl FnOnce(&mut Self, ScopeId) -> vuir::InstructionRef,
	) -> vuir::InstructionRef {
		let inst = self.inst_id(block_scope, vuir::Opcode::Invalid);

		let instructions = {
			let block = self.stack_block(block_scope, BlockKind::Body);
			let expr = f(self, *block);
			if !block.block(self).ends_with_never(self) {
				self.inst(*block, vuir::Opcode::BreakComptime { block: inst, value: expr });
			}
			self.collect_instructions_and_unstack_block(block)
		};

		self.instructions[inst] = vuir::Opcode::BlockComptime { instructions };

		inst.as_ref()
	}

	fn lower_expr_named(
		&mut self,
		block_scope: ScopeId,
		expr: &'ast ast::Expr,
		rhs_ctx: ExprResultLocation,
		naming: vuir::NamingKind,
	) -> vuir::InstructionRef {
		match expr.kind {
			ast::ExprKind::Type(r#type) => {
				let inst = self.lower_type(block_scope, r#type, naming, expr.span);
				self.rvalue(block_scope, inst, rhs_ctx, expr.span)
			},
			_ => self.lower_expr(block_scope, expr, rhs_ctx),
		}
	}

	fn lower_expr(
		&mut self,
		block_scope: ScopeId,
		expr: &'ast ast::Expr,
		rhs_ctx: ExprResultLocation,
	) -> vuir::InstructionRef {
		if let Some(inst) = self.try_lower_linear_unary_expr(block_scope, expr, rhs_ctx) {
			return inst;
		}

		let mut expr = expr;
		while let ast::ExprKind::Group(group) = expr.kind {
			expr = group;
		}

		let inst = match expr.kind {
			ast::ExprKind::FnCall(&ast::FnCall { callee, generics, args }) => {
				// if the fn call has a field access, we must emit a FnCallField since we must first access the receiver
				// then dereference the receiver ptr to get the namepace type to then get the proper function by the field name
				enum CallKind {
					Direct(vuir::InstructionRef),
					Field {
						receiver: vuir::InstructionRef,
						field: &'static ast::Ident,
					},
				}
				let call_kind = match callee.kind {
					ast::ExprKind::FieldAccess(field_access) => {
						let receiver = self.lower_expr(block_scope, field_access.lhs, ExprResultLocation::GetAddressOf);
						CallKind::Field {
							receiver,
							field: field_access.field,
						}
					},
					_ => CallKind::Direct(self.lower_expr(block_scope, callee, ExprResultLocation::None)),
				};

				// get return type before call instruction as it directly depend on its value
				let ret_ty = match rhs_ctx {
					ExprResultLocation::StoreToPtr { ptr, .. } | ExprResultLocation::StoreToInferredPtr { ptr, .. } => {
						Some(self.inst(block_scope, vuir::Opcode::TypeOfPtrPointee { ptr }))
					},
					c => c.into_inst(),
				};

				// now emit the future call instruction and lower arguments
				let call_inst = self.inst_id(block_scope, vuir::Opcode::Invalid);

				// Lower explicit generic arguments
				let mut generic_args = BumpVec::with_capacity_in(generics.len(), self.instructions_payload_alloc);

				let mut fn_args = BumpVec::with_capacity_in(args.len(), self.instructions_payload_alloc);

				for generic in generics {
					// For now, we don't have the type info to coerce
					let arg_block_scope = self.stack_block(block_scope, BlockKind::Body);
					let lowered_arg = self.lower_expr(*arg_block_scope, generic.value, ExprResultLocation::None);
					self.inst(*arg_block_scope, vuir::Opcode::BreakComptime {
						block: call_inst,
						value: lowered_arg,
					});
					let arg_body = self.collect_instructions_and_unstack_block(arg_block_scope);

					fn_args.push(vuir::FnCallArg {
						name: Some(generic.ident.symbol),
						body: arg_body,
						span: expr.span,
					});
				}

				// Collect all arguments preserving their names/positions
				for (i, arg) in args.iter().enumerate() {
					let (arg_name, arg_expr, span, arg_rhs_ctx) = match arg {
						ast::Arg::Named { name, value } => (
							Some(name.symbol),
							*value,
							name.span,
							ExprResultLocation::CoerceToTy(call_inst.as_ref()), // in sema call analysis we will properly resolve the type
						),
						ast::Arg::Positional(value) => {
							(
								None,
								*value,
								value.span,
								ExprResultLocation::CoerceToTy(call_inst.as_ref()), // in sema call analysis we will properly resolve the type
							)
						},
					};

					// For now, we don't have the type info to coerce
					let arg_block_scope = self.stack_block(block_scope, BlockKind::Body);
					let lowered_arg = self.lower_expr(*arg_block_scope, arg_expr, arg_rhs_ctx);
					self.inst(*arg_block_scope, vuir::Opcode::BreakComptime {
						block: call_inst,
						value: lowered_arg,
					});
					let arg_body = self.collect_instructions_and_unstack_block(arg_block_scope);

					fn_args.push(vuir::FnCallArg {
						name: arg_name,
						body: arg_body,
						span,
					});
				}

				// special handling for @import, we want to discover imports early on before semantic analysis for performances
				if let ast::ExprKind::Ident(ident) = callee.kind
					&& ident.is_builtin()
					&& ident.symbol.as_ref() == "@import"
				{
					if !args.is_empty()
						&& let ast::Arg::Positional(path) = args[0]
						&& let ast::ExprKind::Lit(ast::Lit::Str(import_path)) = path.kind
					{
						self.imports.push(vuir::Import {
							path: *import_path,
							span: path.span,
						});
					} else {
						self.errors.push(
							Diagnostic::error()
								.with_message("@import takes a positional string path as its sole argument: `@import(\"module.vif\")`")
								.with_label(Label::primary().with_span(self.diag_span(expr.span))),
						)
					};
				}

				// finally set the fn call inst
				match call_kind {
					CallKind::Direct(fun) => {
						self.instructions[call_inst] = vuir::Opcode::FnCall {
							fun,
							generic_args: generic_args.into_bump_slice(),
							args: fn_args.into_bump_slice(),
							ret_ty,
							span: expr.span,
						};
					},
					CallKind::Field { receiver, field } => {
						self.instructions[call_inst] = vuir::Opcode::FnCallWithFieldPtrReceiver {
							field_ptr: receiver,
							field_name: *field,
							generic_args: generic_args.into_bump_slice(),
							args: fn_args.into_bump_slice(),
							ret_ty,
							span: expr.span,
						};
					},
				}

				call_inst.as_ref()
			},
			ast::ExprKind::FieldAccess(path) => return self.lower_field_access(block_scope, path, rhs_ctx), /* lower_path handles rvalue */
			ast::ExprKind::Ident(ident) => {
				return self
					.lower_ident(block_scope, ident, rhs_ctx, expr.span)
					.unwrap_or_else(|| self.inst_invalid(block_scope));
			},
			ast::ExprKind::Lit(lit) => match lit {
				int @ ast::Lit::Integer { .. } => self.lower_lit_int(int, true),
				ast::Lit::Str(s) => {
					// TODO(ldubos): better handle pointers, and properly set str ptr type based on LHS
					// String literals are typed as [*:0]const u8 (pointer to null-terminated u8 array)
					let u8_ty = self.cu.values.intern_trivial(&value::Key::TypeInt { signed: false, bits: 8 });

					let slice_ty = self
						.cu
						.values
						.intern_trivial(&value::Key::TypeSlice(TypeSlice { pointee_ty: u8_ty }));

					let ptr_ty = self.cu.values.intern_trivial(&value::Key::TypePtr(TypePtr {
						pointee_ty: u8_ty,
						packed: None,
						is_const: true,
					}));
					vuir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Str { slice_ty, value: *s }))
				},
				ast::Lit::Float { symbol, suffix } => vuir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Float {
					ty: match suffix {
						Some(ast::FloatSuffix::F16) => self.cu.values.common.f16_t,
						Some(ast::FloatSuffix::F32) => self.cu.values.common.f32_t,
						Some(ast::FloatSuffix::F64) => self.cu.values.common.f64_t,
						Some(ast::FloatSuffix::F128) => self.cu.values.common.f128_t,
						_ => self.cu.values.common.anyfloat_t,
					},
					// TODO(zino): rust nightly doesn't support f128 parsing
					value: value::Anyfloat(symbol.parse::<f64>().unwrap() as f128),
				})),
				ast::Lit::Bool(b) => vuir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(*b))),
				ast::Lit::EnumVariant(variant) => {
					// try to be more smart with a enum variant, in cases of a expr such as
					// var a: ... = .variant we can construct a field access instruction by fetching the type
					// of the ptr
					match rhs_ctx {
						ExprResultLocation::StoreToPtr { ptr, .. } => {
							let pointee_ty = self.inst(block_scope, vuir::Opcode::TypeOfPtrPointee { ptr });
							self.inst(block_scope, vuir::Opcode::FieldValFromVal {
								lhs: pointee_ty,
								field: *variant,
								span: expr.span,
							})
						},
						ExprResultLocation::CoerceToTy(ty) => self.inst(block_scope, vuir::Opcode::FieldValFromVal {
							lhs: ty,
							field: *variant,
							span: expr.span,
						}),
						c => {
							self.errors.push(
								Diagnostic::error()
									.with_message("cannot infer type of field")
									.with_label(Label::primary().with_span(self.diag_span(expr.span))),
							);
							self.inst(block_scope, vuir::Opcode::Invalid)
						},
					}
				},
				ast::Lit::Null | ast::Lit::Char(..) => todo!(),
			},
			ast::ExprKind::Undefined => self.inst(block_scope, vuir::Opcode::Undefined {
				ty: rhs_ctx.into_inst(),
				span: expr.span,
			}),

			// arithmetic
			ast::ExprKind::Add(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Add { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::AddSat(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::AddSat { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Sub(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Sub { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::SubSat(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::SubSat { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Mul(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Mul { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::MulSat(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::MulSat { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Div(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Div { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Rem(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Rem { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Lt(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Lt { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Lte(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Lte { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Gt(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Gt { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Gte(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Gte { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BoolAnd(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BoolAnd { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BoolOr(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BoolOr { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Eq(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Eq { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Neq(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Neq { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Type(ty) => self.lower_type(block_scope, ty, NamingKind::Anonymous, expr.span),
			ast::ExprKind::StructInit(init) => {
				let struct_ty = match rhs_ctx {
					ExprResultLocation::StoreToPtr { ptr, .. } => self.inst(block_scope, vuir::Opcode::TypeOfPtrPointee { ptr }),
					ExprResultLocation::StoreToInferredPtr { ptr, .. } => {
						if let Some(explicit_ty) = init.ty {
							self.lower_expr(block_scope, explicit_ty, ExprResultLocation::None)
						} else {
							self.inst(block_scope, vuir::Opcode::TypeOfPtrPointee { ptr })
						}
					},
					_ => {
						if let Some(ty) = rhs_ctx.into_inst() {
							ty
						} else if let Some(explicit_ty) = init.ty {
							self.lower_expr(block_scope, explicit_ty, ExprResultLocation::None)
						} else {
							self.errors.push(
								Diagnostic::error()
									.with_message("cannot infer type of struct initializer")
									.with_label(Label::primary().with_span(self.diag_span(expr.span))),
							);
							self.inst(block_scope, vuir::Opcode::Invalid)
						}
					},
				};

				let mut fields = BumpVec::with_capacity_in(init.fields.len(), self.instructions_payload_alloc);
				init.fields
					.iter()
					.map(|field| {
						// TODO(zino): struct field ty inference
						let ty = self.inst(block_scope, vuir::Opcode::StructInitTypeOfField {
							r#struct: struct_ty,
							field: field.ident.symbol,
						});
						let value = self.lower_expr(block_scope, field.value, ExprResultLocation::CoerceToTy(ty));
						vuir::StructInitField {
							name: field.ident,
							value,
							span: field.value.span,
						}
					})
					.collect_into(&mut fields);
				self.inst(block_scope, vuir::Opcode::AggregateInit {
					ty: Some(struct_ty),
					fields: fields.into_bump_slice(),
					span: expr.span,
				})
			},
			ast::ExprKind::If(r#if) => self.lower_expr_if(block_scope, r#if, rhs_ctx),
			kind @ (ast::ExprKind::Block(b) | ast::ExprKind::Loop(b)) => {
				if let Some(label) = b.label {
					self.check_label_already_defined(block_scope, label);
				}

				let block = self.inst_id(block_scope, vuir::Opcode::Invalid);
				let parent_scope = block_scope;
				let block_scope = self.stack_block(block_scope, match kind {
					ast::ExprKind::Block(_) => BlockKind::Block {
						inst: block.as_ref(),
						label: b.label,
					},
					ast::ExprKind::Loop(_) => BlockKind::Loop {
						block_inst: block.as_ref(),
						label: b.label,
					},
					_ => unreachable!(),
				});
				self.lower_block(*block_scope, b);

				// if loop and we don't end with a break or continue insert the implicit repeat
				if matches!(kind, ast::ExprKind::Loop(_)) && !block_scope.block(self).ends_with_never(self) {
					self.inst(*block_scope, vuir::Opcode::Repeat { r#loop: block });
				}

				// finalize block
				let instructions = self.collect_instructions_and_unstack_block(block_scope);

				self.instructions[block] = match kind {
					ast::ExprKind::Block(b) => vuir::Opcode::Block {
						instructions,
						span: b.span,
					},
					ast::ExprKind::Loop(b) => vuir::Opcode::Loop {
						instructions,
						span: b.span,
					},
					_ => unreachable!(),
				};

				block.as_ref()
			},
			ast::ExprKind::While(w) => {
				// a while loop is desugared to loop { if !cond { break; } body; repeat; }
				if let Some(label) = w.body.label {
					self.check_label_already_defined(block_scope, label);
				}

				let loop_block = self.inst_id(block_scope, vuir::Opcode::Invalid);
				let loop_scope = self.stack_block(block_scope, BlockKind::Loop {
					block_inst: loop_block.as_ref(),
					label: w.body.label,
				});

				// branch(!cond, then=[break from loop], else=[])
				{
					let cond = (self.lower_expr(*loop_scope, w.cond, ExprResultLocation::None), w.cond.span);
					let cond = (
						self.inst(*loop_scope, vuir::Opcode::BitNot {
							op: cond.0,
							span: w.cond.span,
						}),
						cond.1,
					); // TODO(zino): use BoolNot?
					let branch_block = self.inst_id(*loop_scope, vuir::Opcode::Invalid);
					let then_body = {
						let scope = self.stack_block(block_scope, BlockKind::Branch(branch_block.as_ref()));
						// break from loop
						self.inst(*scope, vuir::Opcode::Break {
							block: loop_block,
							value: vuir::InstructionRef::Interned(self.cu.values.common.void_value),
							value_span: w.span,
						});
						self.collect_instructions_and_unstack_block(scope)
					};
					let else_body = {
						let scope = self.stack_block(block_scope, BlockKind::Branch(branch_block.as_ref()));
						// nothing
						self.inst(*scope, vuir::Opcode::Break {
							block: branch_block,
							value: vuir::InstructionRef::Interned(self.cu.values.common.void_value),
							value_span: w.span,
						});
						self.collect_instructions_and_unstack_block(scope)
					};

					// create scope for block, it'll only contain the branch instruction
					let branch_block_scope = self.stack_block(block_scope, BlockKind::Body);
					self.inst(*branch_block_scope, vuir::Opcode::Branch {
						cond,
						then_body,
						else_body,
						span: w.span,
					});

					// finalize block
					let instructions = self.collect_instructions_and_unstack_block(branch_block_scope);
					self.instructions[branch_block] = vuir::Opcode::Block {
						instructions,
						span: w.cond.span,
					};
				}

				// Lower the actual body statements
				self.lower_block(*loop_scope, w.body);

				// Implicit repeat at end of loop if body doesn't end with never
				if !loop_scope.block(self).ends_with_never(self) {
					self.inst(*loop_scope, vuir::Opcode::Repeat { r#loop: loop_block });
				}

				let instructions = self.collect_instructions_and_unstack_block(loop_scope);
				self.instructions[loop_block] = vuir::Opcode::Loop {
					instructions,
					span: w.span,
				};

				loop_block.as_ref()
			},

			ast::ExprKind::AddressOf(addr_of) => become self.lower_expr(block_scope, addr_of, ExprResultLocation::GetAddressOf),
			ast::ExprKind::Deref(deref) => {
				// If the context require the address of the deref,
				// we just lower the inner expression without generating a load
				if matches!(rhs_ctx, ExprResultLocation::GetAddressOf) {
					become self.lower_expr(block_scope, deref, ExprResultLocation::None)
				} else {
					let expr = self.lower_expr(block_scope, deref, ExprResultLocation::None);
					self.inst(block_scope, vuir::Opcode::Load {
						src: expr,
						span: deref.span,
					})
				}
			},
			// bitwise
			ast::ExprKind::Shl(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Shl { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::ShlSat(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::ShlSat { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::ShlWrap(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::ShlWrap { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::Shr(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::Shr { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::ShrSat(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::ShrSat { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::ShrWrap(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::ShrWrap { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BitAnd(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BitAnd { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BitOr(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BitOr { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BitXor(&ast::BinOp { lhs, rhs }) => {
				let lhs = self.lower_expr(block_scope, lhs, ExprResultLocation::None);
				let rhs = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BitXor { lhs, rhs, span: expr.span })
			},
			ast::ExprKind::BitNot(op) => {
				let op = self.lower_expr(block_scope, op, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BitNot { op, span: expr.span })
			},

			ast::ExprKind::Neg(op) => {
				// if op is a integer, apply negation now
				if let ast::ExprKind::Lit(integer @ ast::Lit::Integer { .. }) = op.kind {
					self.lower_lit_int(integer, false)
				} else {
					let op = self.lower_expr(block_scope, op, ExprResultLocation::None);
					self.inst(block_scope, vuir::Opcode::Negate { op, span: expr.span })
				}
			},
			ast::ExprKind::Not(op) => {
				let op = self.lower_expr(block_scope, op, ExprResultLocation::None);
				self.inst(block_scope, vuir::Opcode::BoolNot { op, span: expr.span })
			},
			ast::ExprKind::Switch(sw) => {
				if let Some(label) = sw.label {
					self.check_label_already_defined(block_scope, label);
				}

				let switch_block = self.inst_id(block_scope, vuir::Opcode::Invalid);
				let switch_scope = self.stack_block(block_scope, BlockKind::Block {
					inst: switch_block.as_ref(),
					label: sw.label,
				});

				// Lower the switch expression as the operand
				let operand_ref = self.lower_expr(*switch_scope, sw.expr, ExprResultLocation::None);
				let operand = operand_ref.as_id().expect("switch operand must be an instruction");

				// Get the type of the operand for enum variant inference in case patterns
				let operand_ty = self.inst(*switch_scope, vuir::Opcode::TypeOf { value: operand_ref });
				let case_rhs_ctx = ExprResultLocation::CoerceToTy(operand_ty);

				let mut single_cases = BumpVec::new_in(self.instructions_payload_alloc);
				let mut multi_cases = BumpVec::new_in(self.instructions_payload_alloc);

				for case in sw.cases {
					assert!(!case.patterns.is_empty());

					if case.patterns.len() == 1 {
						// Single-pattern case: lower pattern first (needed for capture)
						let item = self.lower_expr(*switch_scope, &case.patterns[0], case_rhs_ctx);

						let body = {
							let scope = self.stack_block(block_scope, BlockKind::Branch(switch_block.as_ref()));

							// If there's a capture, create a SwitchCapture instruction
							// and bind it as a local value so the body can reference it
							let body_scope = if let Some(capture) = case.capture {
								let capture_ref = self.inst(*scope, vuir::Opcode::SwitchCapture {
									switch_operand: operand,
									case_item: item,
									span: capture.span,
								});

								// SAFETY: was_resolved_atleast_once is null

								unsafe {
									self.scopes.push(Scope::LocalValue {
										parent: *scope,
										name: capture.symbol,
										node: case.id,
										vuir_inst: capture_ref,
										was_resolved_atleast_once: std::ptr::null_mut(),
									})
								}
							} else {
								*scope
							};

							let val = match &case.stmt.kind {
								ast::StatementKind::ImplicitReturn(expr) => self.lower_expr(body_scope, expr, ExprResultLocation::None),
								_ => self.lower_stmt(body_scope, case.stmt).0,
							};
							if !scope.block(self).ends_with_never(self) {
								self.inst(*scope, vuir::Opcode::Break {
									block: switch_block,
									value: val,
									value_span: case.stmt.span,
								});
							}
							self.collect_instructions_and_unstack_block(scope)
						};

						single_cases.push(vuir::SwitchSingleCase {
							item,
							capture: case.capture,
							body,
						});
					} else {
						// Multi-pattern case
						if let Some(capture) = case.capture {
							self.errors.push(
								Diagnostic::error()
									.with_message("captures are not supported on multi-pattern switch cases")
									.with_label(Label::primary().with_span(self.diag_span(capture.span))),
							);
						}

						let body = {
							let scope = self.stack_block(block_scope, BlockKind::Branch(switch_block.as_ref()));
							let val = match &case.stmt.kind {
								ast::StatementKind::ImplicitReturn(expr) => self.lower_expr(*scope, expr, ExprResultLocation::None),
								_ => self.lower_stmt(*scope, case.stmt).0,
							};
							if !scope.block(self).ends_with_never(self) {
								self.inst(*scope, vuir::Opcode::Break {
									block: switch_block,
									value: val,
									value_span: case.stmt.span,
								});
							}
							self.collect_instructions_and_unstack_block(scope)
						};

						let mut items = BumpVec::new_in(self.instructions_payload_alloc);
						for pattern in case.patterns {
							let item = self.lower_expr(*switch_scope, pattern, case_rhs_ctx);
							items.push(item);
						}
						multi_cases.push(vuir::SwitchMultiCase {
							items: items.into_bump_slice(),
							body,
						});
					}
				}

				// Lower else body
				let else_body = if let Some(else_stmt) = sw.else_stmt {
					let scope = self.stack_block(block_scope, BlockKind::Branch(switch_block.as_ref()));
					let val = match &else_stmt.kind {
						ast::StatementKind::ImplicitReturn(expr) => self.lower_expr(*scope, expr, ExprResultLocation::None),
						_ => self.lower_stmt(*scope, else_stmt).0,
					};
					if !scope.block(self).ends_with_never(self) {
						self.inst(*scope, vuir::Opcode::Break {
							block: switch_block,
							value: val,
							value_span: else_stmt.span,
						});
					}
					Some(self.collect_instructions_and_unstack_block(scope))
				} else {
					None
				};

				// Emit the Switch instruction inside the block scope
				self.inst(*switch_scope, vuir::Opcode::Switch {
					operand,
					single_cases: single_cases.into_bump_slice(),
					multi_cases: multi_cases.into_bump_slice(),
					else_body,
					span: sw.span,
				});

				// If no else, add implicit void break after the switch
				if sw.else_stmt.is_none() && !switch_scope.block(self).ends_with_never(self) {
					self.inst(*switch_scope, vuir::Opcode::Break {
						block: switch_block,
						value: vuir::InstructionRef::Interned(self.cu.values.common.void_value),
						value_span: sw.span,
					});
				}

				let instructions = self.collect_instructions_and_unstack_block(switch_scope);
				self.instructions[switch_block] = vuir::Opcode::Block {
					instructions,
					span: sw.span,
				};

				switch_block.as_ref()
			},
			ast::ExprKind::Index(index) => {
				let array = self.lower_expr(block_scope, index.collection, ExprResultLocation::GetAddressOf);
				let index = match index.kind {
					ast::IndexKind::Index(index) => self.lower_expr(
						block_scope,
						index,
						ExprResultLocation::CoerceToTy(self.cu.values.common.usize_t.into()),
					),
					ast::IndexKind::Range { .. } | ast::IndexKind::RangeInclusive { .. } => todo!(),
				};
				return match rhs_ctx {
					ExprResultLocation::GetAddressOf => self.inst(block_scope, vuir::Opcode::ArrayIndexElemPtr {
						array_ptr: array,
						index,
						span: expr.span,
					}),
					_ => {
						let inst = self.inst(block_scope, vuir::Opcode::ArrayIndexElemVal {
							array_ptr: array,
							index,
							span: expr.span,
						});
						self.rvalue(block_scope, inst, rhs_ctx, expr.span)
					},
				};
			},
			kind => todo!("expr {:?} not supported yet", kind),
		};

		// default expression rvalue handling
		self.rvalue(block_scope, inst, rhs_ctx, expr.span)
	}

	fn try_lower_linear_unary_expr(
		&mut self,
		block_scope: ScopeId,
		expr: &'ast ast::Expr,
		rhs_ctx: ExprResultLocation,
	) -> Option<vuir::InstructionRef> {
		#[derive(Copy, Clone)]
		enum UnaryLayer {
			Pos(Span),
			Neg(Span),
			Not(Span),
			BitNot(Span),
		}

		let mut current = expr;
		let mut layers = Vec::with_capacity(16);
		loop {
			match current.kind {
				ast::ExprKind::Group(group) => current = group,
				ast::ExprKind::Pos(op) => {
					layers.push(UnaryLayer::Pos(current.span));
					current = op;
				},
				ast::ExprKind::Neg(op) => {
					layers.push(UnaryLayer::Neg(current.span));
					current = op;
				},
				ast::ExprKind::Not(op) => {
					layers.push(UnaryLayer::Not(current.span));
					current = op;
				},
				ast::ExprKind::BitNot(op) => {
					layers.push(UnaryLayer::BitNot(current.span));
					current = op;
				},
				_ => break,
			}
		}

		if layers.is_empty() {
			return None;
		}

		let mut inst = self.lower_expr(block_scope, current, ExprResultLocation::None);
		while let Some(layer) = layers.pop() {
			inst = match layer {
				UnaryLayer::Pos(_) => inst,
				UnaryLayer::Neg(span) => self.inst(block_scope, vuir::Opcode::Negate { op: inst, span }),
				UnaryLayer::Not(span) => self.inst(block_scope, vuir::Opcode::BoolNot { op: inst, span }),
				UnaryLayer::BitNot(span) => self.inst(block_scope, vuir::Opcode::BitNot { op: inst, span }),
			};
		}

		Some(self.rvalue(block_scope, inst, rhs_ctx, expr.span))
	}

	fn lower_lit_int(
		&mut self,
		lit: &'static ast::Lit,
		is_positive: bool,
	) -> vuir::InstructionRef {
		let ast::Lit::Integer { symbol, radix, suffix } = lit else {
			unreachable!();
		};

		let ty = match suffix.as_ref() {
			Some(suffix) => match suffix {
				&ast::IntSuffix::U(bits) => self.cu.values.intern_trivial(&value::Key::TypeInt { signed: false, bits }),
				&ast::IntSuffix::I(bits) => self.cu.values.intern_trivial(&value::Key::TypeInt { signed: true, bits }),
				ast::IntSuffix::Usize => self.cu.values.common.usize_t,
				ast::IntSuffix::Isize => self.cu.values.common.isize_t,
			},
			None => self.cu.values.common.anyint_t,
		};
		let value = {
			let value = Anyint::parse_radix(radix.base().into(), match radix {
				Radix::Hexadecimal | Radix::Binary | Radix::Octal => &symbol[2..],
				Radix::Decimal => symbol,
			})
			.unwrap();

			if is_positive { value } else { -value }
		};
		let value = self.cu.values.intern_trivial(&value::Key::Int { ty, value: value.into() });
		vuir::InstructionRef::Interned(value)
	}

	fn lower_expr_if(
		&mut self,
		block_scope: ScopeId,
		r#if: &'static ast::If,
		_rhs_ctx: ExprResultLocation,
	) -> vuir::InstructionRef {
		let cond = (self.lower_expr(block_scope, r#if.cond, ExprResultLocation::None), r#if.cond.span);

		let branch_block = self.inst_id(block_scope, vuir::Opcode::Invalid);

		let then_body = {
			let scope = self.stack_block(block_scope, BlockKind::Branch(branch_block.as_ref()));
			self.lower_block(*scope, r#if.then_block);
			if !scope.block(self).ends_with_never(self) {
				self.inst(block_scope, vuir::Opcode::Break {
					block: branch_block,
					value: vuir::InstructionRef::Interned(self.cu.values.common.void_value),
					value_span: r#if.then_block.span,
				});
			}

			self.collect_instructions_and_unstack_block(scope)
		};
		let else_body = {
			let scope = self.stack_block(block_scope, BlockKind::Branch(branch_block.as_ref()));

			if let Some(r#else) = r#if.else_block {
				match &r#else {
					ast::ElseBlock::If(r#if) => {
						let value = self.lower_expr_if(*scope, r#if, _rhs_ctx);
						self.inst(*scope, vuir::Opcode::Break {
							block: branch_block,
							value,
							value_span: r#if.span,
						});
					},
					ast::ElseBlock::Block(block) => {
						self.lower_block(*scope, block);
					},
				}
			}

			if !scope.block(self).ends_with_never(self) {
				self.inst(*scope, vuir::Opcode::Break {
					block: branch_block,
					value: vuir::InstructionRef::Interned(self.cu.values.common.void_value),
					value_span: r#if.span,
				});
			}
			self.collect_instructions_and_unstack_block(scope)
		};

		// create scope for block, it'll only contain the branch instruction
		let branch_block_scope = self.stack_block(block_scope, BlockKind::Body);
		let _branch = self.inst_id(*branch_block_scope, vuir::Opcode::Branch {
			cond,
			then_body,
			else_body,
			span: r#if.span,
		});

		// finalize block
		let instructions = self.collect_instructions_and_unstack_block(branch_block_scope);
		self.instructions[branch_block] = vuir::Opcode::Block {
			instructions,
			span: r#if.span,
		};

		branch_block.as_ref()
	}

	/// Lower an expression being part of the init part of a declaration.
	/// The main difference with `lower_expr` is that it'll enforce a proper naming for structs and such
	fn lower_decl_init_expr(
		&mut self,
		block_scope: ScopeId,
		expr: &'ast ast::Expr,
		rhs_ctx: ExprResultLocation,
	) -> vuir::InstructionRef {
		match expr.kind {
			ast::ExprKind::Type(&ast::Type::Struct(r#struct)) => {
				// TODO(zino): rhs_ctx propagate here
				self.lower_struct(block_scope, r#struct, vuir::NamingKind::FromDecl).into_ref()
			},
			// TODO(zino): rhs_ctx propagate here
			ast::ExprKind::Type(&ast::Type::Enum(r#enum)) => self.lower_enum(block_scope, r#enum, vuir::NamingKind::FromDecl).into_ref(),
			_ => become self.lower_expr(block_scope, expr, rhs_ctx),
		}
	}

	/// Lower a statement.
	///
	/// Returns the instruction reference alongside the scope created by the stmt
	fn lower_stmt(
		&mut self,
		block_scope: ScopeId,
		stmt: &'ast ast::Statement,
	) -> (vuir::InstructionRef, ScopeId) {
		let (line, col) = self.start_line_col(stmt.span);
		self.inst(block_scope, vuir::Opcode::DbgSrcLoc { line, col });

		match &stmt.kind {
			kind @ (ast::StatementKind::Const(binding) | ast::StatementKind::Var(binding)) => {
				self.lower_var_binding(block_scope, binding, false, matches!(kind, ast::StatementKind::Var(_)))
			},
			ast::StatementKind::ComptimeVarBinding(binding) => self.lower_var_binding(block_scope, binding, true, true),
			ast::StatementKind::Expr(expr) => (self.lower_expr(block_scope, expr, ExprResultLocation::None), block_scope),
			ast::StatementKind::Return(value) => {
				let value = value.as_ref().map(|expr| {
					// Create the return type reference first so we can use it as a hint
					let ret_type = self.inst(block_scope, vuir::Opcode::TypeOfCurFnRet);
					let value = self.lower_expr(block_scope, expr, ExprResultLocation::CoerceToTy(ret_type));
					self.inst(block_scope, vuir::Opcode::Coerce {
						value,
						into: ret_type,
						span: expr.span,
					})
				});

				self.append_defers(block_scope, self.fn_body_root_scope.expect("return outside function"));

				let inst = self.inst(block_scope, vuir::Opcode::Return { value, span: stmt.span });
				(inst, block_scope)
			},
			ast::StatementKind::Break { label, value } => {
				// try find the block to break
				// if we don't have any label we'll try to find the first loop we encounter
				let break_block_scope = self.traverse_scopes_from(block_scope, |scope| {
					match &self.scopes[scope] {
						Scope::Block(block) => match &block.kind {
							// if a block, only break from it if we happend to have same the label
							BlockKind::Block { inst, label: block_label } if matches!((label, block_label), (Some(a), Some(b)) if a.symbol == b.symbol) => {
								ControlFlow::Break((scope, *inst))
							},
							// for a loop, either the break has no labe then we just hit the first loop or else
							// we do like with blocks
							BlockKind::Loop {
								block_inst,
								label: block_label,
							} if label.is_none() || matches!((label, block_label), (Some(a), Some(b)) if a.symbol == b.symbol) => {
								ControlFlow::Break((scope, *block_inst))
							},
							_ => ControlFlow::Continue(()),
						},
						_ => ControlFlow::Continue(()),
					}
				});

				if let Some((_break_block_scope, break_block_inst)) = break_block_scope {
					let (value, value_span) = value
						.as_ref()
						.map(|expr| (self.lower_expr(block_scope, expr, ExprResultLocation::None), expr.span))
						.unwrap_or((vuir::InstructionRef::Interned(self.cu.values.common.void_value), stmt.span));

					let inst = self.inst(block_scope, vuir::Opcode::Break {
						block: break_block_inst.as_id().unwrap(),
						value,
						value_span,
					});
					(inst, block_scope)
				} else {
					if let Some(label) = label {
						self.errors.push(
							Diagnostic::error()
								.with_message(format!("cannot find labeled block or loop `:{}`", label))
								.with_label(Label::primary().with_span(self.diag_span(stmt.span))),
						);
					} else {
						self.errors.push(
							Diagnostic::error()
								.with_message("cannot break outside of loop or labeled block")
								.with_label(Label::primary().with_span(self.diag_span(stmt.span))),
						);
					}

					let inst = self.inst(block_scope, vuir::Opcode::Invalid);
					(inst, block_scope)
				}
			},
			ast::StatementKind::Continue { label, .. } => {
				// find the loop that we should repeat
				enum FindLoopResult {
					Loop {
						continue_block_scope: ScopeId,
						block_inst: vuir::InstructionRef,
					},
					LabeledBlock(ast::Ident),
				}
				let continue_block_scope = self.traverse_scopes_from(block_scope, |scope| {
					match &self.scopes[scope] {
						Scope::Block(block) => match block.kind {
							// for loops, either we have no label and therefore hit the first loop we encounter
							// or find the one with our label
							BlockKind::Loop {
								block_inst,
								label: block_label,
							} if label.is_none() || matches!((label, block_label), (Some(a), Some(b)) if a.symbol == b.symbol) => {
								ControlFlow::Break(FindLoopResult::Loop {
									continue_block_scope: scope,
									block_inst,
								})
							},
							// for blocks break if we found one with our label, we'll emit a diagnostic later on
							BlockKind::Block {
								inst,
								label: Some(block_label),
								..
							} if matches!((label, block_label), (Some(a), b) if a.symbol == b.symbol) => {
								ControlFlow::Break(FindLoopResult::LabeledBlock(block_label))
							},
							_ => ControlFlow::Continue(()),
						},
						_ => ControlFlow::Continue(()),
					}
				});

				let inst = match continue_block_scope {
					Some(FindLoopResult::Loop {
						continue_block_scope,
						block_inst,
					}) => {
						let block = block_inst.as_id().unwrap();
						self.inst(continue_block_scope, vuir::Opcode::Repeat { r#loop: block })
					},
					Some(FindLoopResult::LabeledBlock(block_label)) => {
						self.errors.push(
							Diagnostic::error()
								.with_message(format!("cannot continue label `:{}` which is not a loop", block_label))
								.with_label(Label::primary().with_span(self.diag_span(stmt.span)))
								.with_label(
									Label::primary()
										.with_span(self.diag_span(block_label.span))
										.with_message("label defined here"),
								),
						);
						self.inst(block_scope, vuir::Opcode::Invalid)
					},
					None => {
						if let Some(label) = label {
							self.errors.push(
								Diagnostic::error()
									.with_message(format!("cannot find labeled loop `:{}`", label))
									.with_label(Label::primary().with_span(self.diag_span(stmt.span))),
							);
						} else {
							self.errors.push(
								Diagnostic::error()
									.with_message("cannot continue outside of a loop")
									.with_label(Label::primary().with_span(self.diag_span(stmt.span))),
							);
						}
						self.inst(block_scope, vuir::Opcode::Invalid)
					},
				};

				(inst, block_scope)
			},
			ast::StatementKind::ImplicitReturn(expr) => {
				// implicit returns stop at the first block found, even if it's a loop
				// we'll error out later
				let break_block_scope = self
					.traverse_scopes_from(block_scope, |scope| match &self.scopes[scope] {
						Scope::Block(block) => match block.kind {
							BlockKind::Loop { .. } | BlockKind::Branch(_) | BlockKind::Block { .. } => {
								ControlFlow::Break(Some((scope, block)))
							},
							// completely ignore implicit returns in body blocks
							BlockKind::Body => ControlFlow::Break(None),
						},
						_ => ControlFlow::Continue(()),
					})
					.flatten();

				let inst = if let Some((break_block_scope, break_block)) = break_block_scope {
					let block_inst = match break_block.kind {
						BlockKind::Loop { block_inst, .. } => {
							self.errors.push(
								Diagnostic::error()
									.with_message("implicit return illegal in loop expressions")
									.with_label(Label::primary().with_span(self.diag_span(expr.span)))
									.with_note("use the `break ...;` syntax to exit the loop with a value"),
							);
							block_inst
						},
						BlockKind::Branch(inst) | BlockKind::Block { inst, .. } => inst,
						BlockKind::Body => unreachable!(),
					}
					.as_id()
					.unwrap();

					// Breaking to an enclosing block or branch. Do not coerce to the function return type.
					let value = self.lower_expr(block_scope, expr, ExprResultLocation::None);
					self.append_defers(block_scope, break_block_scope);
					self.inst(block_scope, vuir::Opcode::Break {
						block: block_inst,
						value,
						value_span: expr.span,
					})
				} else {
					// No block, assume we return from the enclosing function
					let ret_type = self.inst(block_scope, vuir::Opcode::TypeOfCurFnRet);
					let value = self.lower_expr(block_scope, expr, ExprResultLocation::CoerceToTy(ret_type));
					let value = self.inst(block_scope, vuir::Opcode::Coerce {
						value,
						into: ret_type,
						span: expr.span,
					});

					// since we assume to return from the function, append defers until function root scope
					self.append_defers(block_scope, self.fn_body_root_scope.expect("return outside function"));

					self.inst(block_scope, vuir::Opcode::Return {
						value: Some(value),
						span: stmt.span,
					})
				};
				(inst, block_scope)
			},
			ast::StatementKind::Assign { lhs, op, rhs } => {
				let lhs_addr = self.lower_expr(block_scope, lhs, ExprResultLocation::GetAddressOf);
				let inst = match op {
					ast::AssignOp::Assign => self.lower_expr(block_scope, rhs, ExprResultLocation::StoreToPtr {
						ptr: lhs_addr,
						span: lhs.span,
					}),
					compound_op => {
						let lhs_val = self.inst(block_scope, vuir::Opcode::Load {
							src: lhs_addr,
							span: lhs.span,
						});
						let rhs_val = self.lower_expr(block_scope, rhs, ExprResultLocation::None);
						let result = match compound_op {
							ast::AssignOp::Add => self.inst(block_scope, vuir::Opcode::Add {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::AddSat => self.inst(block_scope, vuir::Opcode::AddSat {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Sub => self.inst(block_scope, vuir::Opcode::Sub {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::SubSat => self.inst(block_scope, vuir::Opcode::SubSat {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Mul => self.inst(block_scope, vuir::Opcode::Mul {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::MulSat => self.inst(block_scope, vuir::Opcode::MulSat {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Div => self.inst(block_scope, vuir::Opcode::Div {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Rem => self.inst(block_scope, vuir::Opcode::Rem {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::BoolAnd => self.inst(block_scope, vuir::Opcode::BoolAnd {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::BoolOr => self.inst(block_scope, vuir::Opcode::BoolOr {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Shl => self.inst(block_scope, vuir::Opcode::Shl {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::ShlSat => self.inst(block_scope, vuir::Opcode::ShlSat {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::ShlWrap => self.inst(block_scope, vuir::Opcode::ShlWrap {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::Shr => self.inst(block_scope, vuir::Opcode::Shr {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::ShrSat => self.inst(block_scope, vuir::Opcode::ShrSat {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::ShrWrap => self.inst(block_scope, vuir::Opcode::ShrWrap {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::BitAnd => self.inst(block_scope, vuir::Opcode::BitAnd {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::BitOr => self.inst(block_scope, vuir::Opcode::BitOr {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							ast::AssignOp::BitXor => self.inst(block_scope, vuir::Opcode::BitXor {
								lhs: lhs_val,
								rhs: rhs_val,
								span: stmt.span,
							}),
							op => {
								self.errors.push(
									Diagnostic::error()
										.with_message(format!("compound assignment operator '{op}' not yet supported"))
										.with_label(Label::primary().with_span(self.diag_span(stmt.span))),
								);
								self.inst(block_scope, vuir::Opcode::Invalid)
							},
						};
						self.inst(block_scope, vuir::Opcode::Store {
							dst: lhs_addr,
							src: result,
							span: stmt.span,
						})
					},
				};
				(inst, block_scope)
			},
			ast::StatementKind::Defer(expr) => {
				let defer_body_scope = self.stack_block(block_scope, BlockKind::Body);
				let _ = self.lower_expr(*defer_body_scope, expr, ExprResultLocation::None);
				let defer_body = self.collect_instructions_and_unstack_block(defer_body_scope);
				let defer_scope = self.scopes.push(Scope::Defer {
					parent: block_scope,
					body: defer_body,
					span: stmt.span,
				});
				let inst = self.cu.values.common.void_value.into();
				(inst, defer_scope)
			},
			ast::StatementKind::Errdefer(..) => unreachable!(),
		}
	}

	fn lower_block(
		&mut self,
		block_scope: ScopeId,
		block: &'ast ast::Block,
	) {
		let mut scope = block_scope;
		for stmt in block.stmts {
			let (_, stmt_scope) = self.lower_stmt(scope, stmt);
			scope = stmt_scope;
		}

		self.append_defers(scope, block_scope);
	}

	fn lower_associated_fn(
		&mut self,
		block_scope: ScopeId,
		fun: &'ast ast::Fn,
	) -> vuir::InstructionId {
		let decl = self.inst_id(block_scope, vuir::Opcode::Invalid);
		let value = {
			let (params, body, ret_ty, ret_ty_is_generic, callconv) = {
				let fn_scope = self.stack_block(block_scope, BlockKind::Body);

				let mut params_scope = *fn_scope;
				let params_start = self.uir_scopes_instructions.len();

				// if true it means some expression resolved to a parameter, so basically the expression is generic
				// we replace later on in sema every generic parameters or return type by the generic poison type
				let mut any_param_resolved_atleast_once = false;

				// lower generic types as regular fn params as they are a shortcut to
				// `const {name}: type` must be done before fn params
				for generic in fun.generics {
					let inst = self.inst_id(*fn_scope, vuir::Opcode::Invalid);

					let type_body = {
						let type_scope = self.stack_block(block_scope, BlockKind::Body);
						let type_start = self.uir_scopes_instructions.len();
						self.inst(*type_scope, vuir::Opcode::BreakComptime {
							block: inst,
							value: vuir::InstructionRef::Interned(self.cu.values.common.type_t),
						});
						let type_end = self.uir_scopes_instructions.len();
						let mut type_body = BumpVec::new_in(self.instructions_payload_alloc);
						self.uir_scopes_instructions[type_start..type_end]
							.iter()
							.collect_into(&mut type_body);
						self.unstack_block(type_scope);
						type_body
					};

					self.instructions[inst] = vuir::Opcode::DeclFnParam {
						name: generic.ident,
						type_body: type_body.into_bump_slice(),
						comptime: true,
						generic: true,
						span: generic.ident.span,
					};

					// SAFETY: any_param_resolved_atleast_once lives as long as params_scope
					params_scope = unsafe {
						self.scopes.push(Scope::LocalValue {
							parent: params_scope,
							name: generic.ident.symbol,
							node: generic.id,
							vuir_inst: inst.as_ref(),
							was_resolved_atleast_once: &raw mut any_param_resolved_atleast_once,
						})
					};
				}

				for param in fun.params {
					any_param_resolved_atleast_once = false;

					let inst = self.inst_id(params_scope, vuir::Opcode::Invalid);
					let type_body = {
						let type_scope = self.stack_block(params_scope, BlockKind::Body);
						let ty = self.lower_expr(*type_scope, param.ty, ExprResultLocation::None);
						self.inst(*type_scope, vuir::Opcode::BreakComptime { block: inst, value: ty });
						self.collect_instructions_and_unstack_block(type_scope)
					};
					self.instructions[inst] = vuir::Opcode::DeclFnParam {
						name: param.ident,
						type_body,
						comptime: param.comptime,
						span: param.ty.span,
						generic: any_param_resolved_atleast_once,
					};

					// SAFETY: any_param_resolved_atleast_once lives as long as params_scope
					params_scope = unsafe {
						self.scopes.push(Scope::LocalValue {
							parent: params_scope,
							name: param.ident.symbol,
							node: param.id,
							vuir_inst: inst.as_ref(),
							was_resolved_atleast_once: &raw mut any_param_resolved_atleast_once,
						})
					};
				}
				let params_end = self.uir_scopes_instructions.len();

				let (ret_ty, ret_ty_is_generic) = {
					any_param_resolved_atleast_once = false;

					let block_inst = self.inst_id(params_scope, vuir::Opcode::Invalid);
					let ret_ty_block_body = {
						let block_scope = self.stack_block(params_scope, BlockKind::Block {
							inst: block_inst.as_ref(),
							label: None,
						});

						let ret_ty_start = self.uir_scopes_instructions.len();
						let ret_ty = self.lower_expr(
							*block_scope,
							fun.ret_ty,
							ExprResultLocation::CoerceToTy(self.cu.values.common.type_t.into()), // for ret ty we want the type of the expr, therefore coerce to type
						);
						self.inst(*block_scope, vuir::Opcode::BreakComptime {
							block: block_inst,
							value: ret_ty,
						});
						let ret_ty_end = self.uir_scopes_instructions.len();

						self.collect_instructions_and_unstack_block(block_scope)
					};

					self.instructions[block_inst] = vuir::Opcode::BlockComptime {
						instructions: ret_ty_block_body,
					};
					(block_inst, any_param_resolved_atleast_once)
				};

				let callconv = if let Some(callconv_expr) = fun.callconv {
					let block_inst = self.inst_id(params_scope, vuir::Opcode::Invalid);
					let callconv_block_body = {
						let block_scope = self.stack_block(params_scope, BlockKind::Block {
							inst: block_inst.as_ref(),
							label: None,
						});
						let builtin_callconv_ty = self.inst(*block_scope, vuir::Opcode::TypeBuiltinCallingConvention);
						let callconv_inst =
							self.lower_expr(*block_scope, callconv_expr, ExprResultLocation::CoerceToTy(builtin_callconv_ty));
						self.inst(*block_scope, vuir::Opcode::BreakComptime {
							block: block_inst,
							value: callconv_inst,
						});
						self.collect_instructions_and_unstack_block(block_scope)
					};

					self.instructions[block_inst] = vuir::Opcode::BlockComptime {
						instructions: callconv_block_body,
					};
					Some(block_inst)
				} else {
					None
				};

				let body_start = self.uir_scopes_instructions.len();
				if let Some(block) = &fun.block {
					let prev_fn_body_root_scope = self.fn_body_root_scope.replace(params_scope);
					self.lower_block(params_scope, block);
					self.fn_body_root_scope = prev_fn_body_root_scope;
				}
				let body_end = self.uir_scopes_instructions.len();

				let mut params = BumpVec::new_in(self.instructions_payload_alloc);
				let mut body = BumpVec::new_in(self.instructions_payload_alloc);

				self.uir_scopes_instructions[params_start..params_end]
					.iter()
					.collect_into(&mut params);
				self.uir_scopes_instructions[body_start..body_end].iter().collect_into(&mut body);

				self.unstack_block(fn_scope);

				(params, body, ret_ty, ret_ty_is_generic, callconv)
			};

			let builtin = match &*fun.ident.symbol {
				"@unsafe_int_cast" => Some(vuir::BuiltinKind::UnsafeIntCast),
				"@size_of" => Some(vuir::BuiltinKind::SizeOf),
				"@bit_size_of" => Some(vuir::BuiltinKind::BitSizeOf),
				"@zeroed" => Some(vuir::BuiltinKind::Zeroed),
				"@int_from_enum" => Some(vuir::BuiltinKind::IntFromEnum),
				"@int_to_float" => Some(vuir::BuiltinKind::IntToFloat),
				"@import" => Some(vuir::BuiltinKind::Import),
				"@nullptr" => Some(vuir::BuiltinKind::Nullptr),
				"@slice_from_raw_parts" => Some(vuir::BuiltinKind::SliceFromRawParts),
				"@slice_ptr" => Some(vuir::BuiltinKind::SlicePtr),
				"@slice_len" => Some(vuir::BuiltinKind::SliceLen),
				"@abort" => Some(vuir::BuiltinKind::Abort),
				"@unreachable" => Some(vuir::BuiltinKind::Unreachable),
				"@ptr_to_int" => Some(vuir::BuiltinKind::PtrToInt),
				"@int_to_ptr" => Some(vuir::BuiltinKind::IntToPtr),
				"@forget" => Some(vuir::BuiltinKind::Forget),
				"@bitcast" => Some(vuir::BuiltinKind::Bitcast),
				"@slice_copy_nonoverlapping" => Some(vuir::BuiltinKind::SliceCopyNonoverlapping),
				_ if fun.ident.symbol.starts_with("@") => {
					self.errors.push(
						Diagnostic::error()
							.with_message(format!("unknown builtin `{}`", fun.ident.symbol))
							.with_label(Label::primary().with_span(self.diag_span(fun.ident.span)))
							.with_note("@ prefix is reserved for compiler builtins"),
					);
					None
				},
				_ => None,
			};

			// Decl value
			let scope = self.stack_block(block_scope, BlockKind::Body);
			let fun_inst = self.inst(*scope, vuir::Opcode::DeclFn {
				external: fun.block.is_none() && builtin.is_none(),
				callconv,
				first_positional_arg_index: if params.is_empty() { None } else { Some(fun.generics.len() as u16) },
				params: params.into_bump_slice(),
				var_args: fun.variadic,
				body: body.into_bump_slice(),
				ret_ty,
				ret_ty_is_generic,
				builtin,
				inline: if builtin.is_some() { ast::Inline::Always } else { fun.inline },
				span: fun.ident.span,
			});
			self.inst(*scope, vuir::Opcode::BreakComptime {
				block: decl,
				value: fun_inst,
			});

			self.collect_instructions_and_unstack_block(scope)
		};

		// Finalize declaration
		self.instructions[decl] = vuir::Opcode::Declaration(vuir::Decl {
			name: fun.ident.symbol,
			value,
			span: fun.ident.span,
		});

		decl
	}

	fn lower_struct(
		&mut self,
		block_scope: ScopeId,
		struct_decl: &'ast ast::StructTy,
		naming: vuir::NamingKind,
	) -> vuir::InstructionId {
		// Insert the struct inst now, we want it to be before inst related to it.
		let struct_inst = self.inst_id(block_scope, vuir::Opcode::Invalid);

		// Scan all declarations before lowering them for name resolution and insert a new namespace scope
		let struct_scope = {
			let mut top_scope = ScopeNamespace {
				parent: Some(block_scope),
				decl_to_ast_node: Default::default(),
				captures: IndexMap::new(FxBuildHasher),
			};
			self.collect_namespace_items(&mut top_scope, struct_decl.associated_items);
			self.scopes.push(Scope::Namespace(top_scope));
			ScopeId(self.scopes.len() - 1)
		};

		let (fields, decls) = {
			let mut fields = vec![];
			for field in struct_decl.fields {
				let ty_block = self.stack_block(struct_scope, BlockKind::Body);
				let ty_ref = self.lower_expr(*ty_block, field.ty, ExprResultLocation::None);
				if !ty_block.block(self).ends_with_never(self) {
					self.inst(*ty_block, vuir::Opcode::BreakComptime {
						block: struct_inst,
						value: ty_ref,
					});
				}
				let ty_instructions = self.collect_instructions_and_unstack_block(ty_block);

				let field_ty = if ty_instructions.is_empty() {
					vuir::FieldTy::Ref(ty_ref)
				} else {
					vuir::FieldTy::Body(ty_instructions)
				};

				fields.push(vuir::Field {
					name: field.ident,
					ty: field_ty,
				});
			}

			let decls = {
				let decls_scope = self.stack_block(struct_scope, BlockKind::Body);
				let mut decls = vec![];
				for item in struct_decl.associated_items {
					decls.push(self.lower_associated_item(*decls_scope, item));
				}
				self.unstack_block(decls_scope);
				decls
			};

			(fields, decls)
		};

		let captures = {
			let mut keys = BumpVec::new_in(self.instructions_payload_alloc);
			self.scopes[struct_scope]
				.as_namespace_mut()
				.captures
				.kvs()
				.iter()
				.map(|(k, _)| *k)
				.collect_into(&mut keys);
			keys
		};

		self.instructions[struct_inst] = vuir::Opcode::DeclStruct {
			naming,
			fields,
			packed: struct_decl.packed,
			linear: struct_decl.linear,
			decls,
			captures: captures.into_bump_slice(),
		};

		// Pop the struct namespace scope
		self.scopes.pop();

		struct_inst
	}

	fn lower_enum(
		&mut self,
		block_scope: ScopeId,
		enum_decl: &'static ast::EnumTy,
		naming: vuir::NamingKind,
	) -> vuir::InstructionId {
		let tag_ty = enum_decl
			.tag_ty
			.map(|expr| (self.lower_expr(block_scope, expr, ExprResultLocation::None), expr.span));

		let enum_inst = self.inst_id(block_scope, vuir::Opcode::Invalid);

		// Scan all declarations before lowering them for name resolution and insert a new namespace scope
		let enum_scope = {
			let mut top_scope = ScopeNamespace {
				parent: if self.scopes.is_empty() { None } else { Some(block_scope) },
				decl_to_ast_node: Default::default(),
				captures: IndexMap::new(FxBuildHasher),
			};
			self.collect_namespace_items(&mut top_scope, enum_decl.associated_items);
			self.scopes.push(Scope::Namespace(top_scope));
			ScopeId(self.scopes.len() - 1)
		};
		let (variants, decls) = {
			let mut variants = vec![];
			// hashmap used to report error on redefinition of variant
			let mut variant_name_to_span = FxHashMap::default();
			for variant in enum_decl.variants {
				let value = variant
					.value
					.map(|expr| (self.lower_expr(block_scope, expr, ExprResultLocation::None), expr.span));

				// ensure variant is unique
				if let Some(existing_variant_span) = variant_name_to_span.get(&variant.ident.symbol) {
					self.errors.push(
						Diagnostic::error()
							.with_message(format!("enum variant `{}` already defined", variant.ident.symbol))
							.with_label(
								Label::primary()
									.with_span(self.diag_span(variant.ident.span))
									.with_message("redefinition here"),
							)
							.with_label(
								Label::secondary()
									.with_span(self.diag_span(*existing_variant_span))
									.with_message("first defined here"),
							),
					);
				} else {
					variant_name_to_span.insert(variant.ident.symbol, variant.span);
				}

				variants.push(vuir::EnumVariant {
					ident: variant.ident,
					value,
					span: variant.span,
				});
			}

			let decls = {
				let associated_items_scope = self.stack_block(block_scope, BlockKind::Body);
				let mut decls = vec![];
				for item in enum_decl.associated_items {
					decls.push(self.lower_associated_item(*associated_items_scope, item));
				}
				self.unstack_block(associated_items_scope);
				decls
			};

			(variants, decls)
		};

		let captures = {
			let mut keys = BumpVec::new_in(self.instructions_payload_alloc);
			self.scopes[enum_scope]
				.as_namespace_mut()
				.captures
				.kvs()
				.iter()
				.map(|(k, _)| *k)
				.collect_into(&mut keys);
			keys
		};

		self.instructions[enum_inst] = vuir::Opcode::DeclEnum {
			tag_ty,
			naming,
			linear: enum_decl.linear,
			variants,
			decls,
			captures: captures.into_bump_slice(),
		};

		// Pop the struct namespace scope
		self.scopes.pop();

		enum_inst
	}

	fn lower_union(
		&mut self,
		block_scope: ScopeId,
		union_decl: &'ast ast::UnionTy,
		naming: vuir::NamingKind,
	) -> vuir::InstructionId {
		let union_inst = self.inst_id(block_scope, vuir::Opcode::Invalid);

		let union_scope = {
			let mut top_scope = ScopeNamespace {
				parent: Some(block_scope),
				decl_to_ast_node: Default::default(),
				captures: IndexMap::new(FxBuildHasher),
			};
			self.collect_namespace_items(&mut top_scope, union_decl.associated_items);
			self.scopes.push(Scope::Namespace(top_scope));
			ScopeId(self.scopes.len() - 1)
		};

		let (fields, decls) = {
			let mut fields = vec![];
			let mut field_name_to_span = FxHashMap::default();

			for field in union_decl.fields {
				// ensure field is unique
				if let Some(existing_span) = field_name_to_span.get(&field.ident.symbol) {
					self.errors.push(
						Diagnostic::error()
							.with_message(format!("union field `{}` already defined", field.ident.symbol))
							.with_label(
								Label::primary()
									.with_span(self.diag_span(field.ident.span))
									.with_message("redefinition here"),
							)
							.with_label(
								Label::secondary()
									.with_span(self.diag_span(*existing_span))
									.with_message("first defined here"),
							),
					);
				} else {
					field_name_to_span.insert(field.ident.symbol, field.span);
				}

				let ty = field.ty.map(|ty_expr| {
					let ty_block = self.stack_block(union_scope, BlockKind::Body);
					let ty_ref = self.lower_expr(*ty_block, ty_expr, ExprResultLocation::None);
					if !ty_block.block(self).ends_with_never(self) {
						self.inst(*ty_block, vuir::Opcode::BreakComptime {
							block: union_inst,
							value: ty_ref,
						});
					}
					let ty_instructions = self.collect_instructions_and_unstack_block(ty_block);

					if ty_instructions.is_empty() {
						vuir::FieldTy::Ref(ty_ref)
					} else {
						vuir::FieldTy::Body(ty_instructions)
					}
				});

				fields.push(vuir::UnionField {
					name: field.ident,
					ty,
					span: field.span,
				});
			}

			let decls = {
				let decls_scope = self.stack_block(union_scope, BlockKind::Body);
				let mut decls = vec![];
				for item in union_decl.associated_items {
					decls.push(self.lower_associated_item(*decls_scope, item));
				}
				self.unstack_block(decls_scope);
				decls
			};

			(fields, decls)
		};

		let captures = {
			let mut keys = BumpVec::new_in(self.instructions_payload_alloc);
			self.scopes[union_scope]
				.as_namespace_mut()
				.captures
				.kvs()
				.iter()
				.map(|(k, _)| *k)
				.collect_into(&mut keys);
			keys
		};

		let tag = match union_decl.tag {
			ast::UnionTagKind::Bare => None,
			ast::UnionTagKind::AutoEnum => Some(None),
			ast::UnionTagKind::Enum(expr) => {
				let ref_val = self.lower_expr(block_scope, expr, ExprResultLocation::None);
				Some(Some((ref_val, expr.span)))
			},
		};

		self.instructions[union_inst] = vuir::Opcode::DeclUnion {
			tag,
			naming,
			linear: union_decl.linear,
			fields,
			decls,
			captures: captures.into_bump_slice(),
		};

		self.scopes.pop();

		union_inst
	}

	fn lower_associated_item(
		&mut self,
		block_scope: ScopeId,
		item: &'ast ast::AssociatedItem,
	) -> vuir::InstructionId {
		match &item.kind {
			ast::AssociatedItemKind::Fn(fun) => self.lower_associated_fn(block_scope, fun),
			ast::AssociatedItemKind::Const(binding) => self.lower_item_var_binding(block_scope, item, binding),
		}
	}

	fn lower(mut self) -> Result<vuir::Vuir, Vec<Diagnostic>> {
		let scope = {
			self.scopes.push(Scope::Block(Block {
				parent: None,
				inst_start: self.uir_scopes_instructions.len(),
				kind: BlockKind::Body,
				resolve_cache: FxHashMap::with_capacity_and_hasher(16, Default::default()),
			}));
			StackedBlock(ScopeId(self.scopes.len() - 1))
		};
		let _module = match &self.ast.kind {
			ast::ModuleKind::StructDecl(s) => {
				// naming is from the module decl
				self.lower_struct(*scope, s, vuir::NamingKind::FromDecl)
			},
			_ => todo!(),
		};
		self.unstack_block(scope);

		if self.errors.is_empty() {
			Ok(vuir::Vuir {
				imports: self.imports.into_bump_slice(),
				instructions: self.instructions,
				// SAFETY: tkt
				instructions_payload_allocator: unsafe { Box::from_raw(std::ptr::from_ref(self.instructions_payload_alloc) as *mut _) },
			})
		} else {
			Err(self.errors)
		}
	}
}

impl<'ast> Lowerer<'ast> {
	fn collect_namespace_items(
		&mut self,
		scope: &mut ScopeNamespace,
		items: &[ast::AssociatedItem],
	) {
		for item in items {
			let ident = match &item.kind {
				ast::AssociatedItemKind::Fn(fun) => fun.ident,
				ast::AssociatedItemKind::Const(c) => c.name,
			};

			if let Some((_, existing_span)) = scope.decl_to_ast_node.insert(ident.symbol, (item.id, ident.span)) {
				self.errors.push(
					Diagnostic::error()
						.with_message(format!("name `{}` already defined", ident.symbol))
						.with_label(
							Label::primary()
								.with_span(self.diag_span(ident.span))
								.with_message("redefinition here"),
						)
						.with_label(
							Label::secondary()
								.with_span(self.diag_span(existing_span))
								.with_message("first defined here"),
						)
						.with_note("a name can only appear once in a namespace"),
				);
			}
		}
	}

	fn traverse_scopes_from<B>(
		&self,
		scope: ScopeId,
		mut f: impl FnMut(ScopeId) -> ControlFlow<B>,
	) -> Option<B> {
		let mut scope_id = scope;
		loop {
			let flow = f(scope_id);
			match flow {
				ControlFlow::Break(b) => break Some(b),
				ControlFlow::Continue(_) => {
					let scope = &self.scopes[scope_id];
					if let Some(parent) = scope.parent() {
						scope_id = parent;
					} else {
						break None;
					}
				},
			}
		}
	}

	fn traverse_scopes_from_mut<B>(
		&mut self,
		scope: ScopeId,
		mut f: impl FnMut(&mut Self, ScopeId) -> ControlFlow<B>,
	) -> Option<B> {
		let mut scope_id = scope;
		loop {
			let flow = f(self, scope_id);
			match flow {
				ControlFlow::Break(b) => break Some(b),
				ControlFlow::Continue(_) => {
					let scope = &self.scopes[scope_id];
					if let Some(parent) = scope.parent() {
						scope_id = parent;
					} else {
						break None;
					}
				},
			}
		}
	}
}

pub fn to_vuir(
	cu: &CompilationUnit,
	src: &str,
	module_id: ModuleId,
	ast: &ast::Module,
) -> Result<vuir::Vuir, Vec<Diagnostic>> {
	// Pre-allocate based on source size
	// Heuristic: ~1 VUIR instruction per ~4 keywords and ~1 scope per 16 bytes of source
	let estimated_instructions = (src.len() / crate::frontend::lexer::AVG_KEYWORD_LEN / 4).max(256);
	let estimated_scopes = (src.len() / 16).max(32);
	let instructions_payload_alloc = Box::leak(Box::new(Bump::new()));

	Lowerer {
		cu,
		src,
		module_id,
		ast,
		instructions: IndexVec::with_capacity(estimated_instructions),
		uir_scopes_instructions: Vec::with_capacity(estimated_instructions),
		instructions_payload_alloc,
		scopes: IndexVec::with_capacity(estimated_scopes),
		errors: vec![],
		line_starts: compute_line_starts(src),
		imports: BumpVec::new_in(instructions_payload_alloc),
		fn_body_root_scope: None,
	}
	.lower()
}

#[inline(always)]
fn compute_line_starts(src: &str) -> Vec<usize> {
	let src = src.as_bytes();

	let mut starts = Vec::with_capacity(128);
	let mut offset = 0;
	starts.push(0);

	while offset < src.len() {
		if let Some(pos) = memx::memchr(&src[offset..], b'\n') {
			offset += pos + 1;
			starts.push(offset);
		} else {
			break;
		}
	}

	starts
}
