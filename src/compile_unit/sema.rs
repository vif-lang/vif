//! Conversion from VUIR to VTIR

use std::{
	collections::HashMap,
	hint::{
		unlikely,
		unreachable_unchecked,
	},
	ops::Deref,
	ptr,
	sync::Arc,
};

use bitvec::vec::BitVec;
use bumpalo::Bump;
use internment::Intern;
use rustc_hash::{
	FxHashMap,
	FxHashSet,
};

mod call;
use call::AnalyzedCallee;

pub type BumpVec<'bump, T> = bumpalo::collections::Vec<'bump, T>;

use crate::{
	common::{
		COMMON_INTERNS,
		IndexVec,
		RcLinearAllocator,
		Span,
		diagnostic::*,
	},
	compile_unit::{
		CompilationUnit,
		Decl,
		DeclAnalysisState,
		DeclId,
		Namespace,
		NamespaceId,
		TypeInfoId,
		module::{
			ModuleAnalyzeState,
			ModuleId,
		},
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
		Anyfloat,
		Anyint,
		CallingConvention,
		ComptimeAllocId,
		EnumField,
		FnDecl,
		FnKey,
		FnValue,
		GlobalVuirInstructionId,
		PackedStructFieldInfo,
		Ptr,
		PtrKind,
		StructField,
		StructLayout,
		TypeEnum,
		TypeFn,
		TypePtr,
		TypePtrPacked,
		TypeSlice,
		TypeStruct,
		TypeUnion,
		UnionField,
	},
};

#[derive(Copy, Clone, Debug)]
pub enum AnalyzeError {
	/// The provided VUIR is semantically wrong.
	AnalysisFailed,
	ComptimeBreak {
		block: vuir::InstructionId,
		value: vtir::InstructionRef,
	},
	InlineReturn {
		value: Option<vtir::InstructionRef>,
	},
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ControlFlow {
	Always,
	May,
	Never,
}

impl ControlFlow {
	fn join(
		&self,
		b: ControlFlow,
	) -> ControlFlow {
		use ControlFlow::*;
		match (self, b) {
			(Always, Always) => Always,
			(Never, Never) => Never,
			_ => May,
		}
	}
}

#[derive(Clone, Debug)]
pub struct DeclFnParam {
	pub name: Intern<str>,
	ty: value::Index,
	pub vuir_id: vuir::InstructionId,
	comptime: bool,
	pub span: Span,
}

#[derive(Debug)]
pub struct VuirBlockAnalysisData {
	block_inst: vtir::InstructionRef,
	/// List of breaked values
	breaks: BumpVec<'static, vtir::InstructionRef>,
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BlockId(pub usize);
impl From<BlockId> for usize {
	fn from(value: BlockId) -> Self {
		value.0
	}
}
impl From<usize> for BlockId {
	fn from(value: usize) -> Self {
		Self(value)
	}
}

#[derive(Clone, Debug)]
pub struct RuntimeCaptureEnv {
	pub ptr: vtir::InstructionRef,
	pub ty: value::Index,
	pub fields: FxHashMap<GlobalVuirInstructionId, usize>,
}

#[derive(Clone, Debug, Default)]
pub struct BlockCaptureContext {
	pub runtime_env: Option<RuntimeCaptureEnv>,
}

#[derive(Debug)]
pub struct Block {
	pub namespace: NamespaceId,
	pub parent: Option<BlockId>,
	pub instructions: BumpVec<'static, vtir::InstructionRef>,
	/// if we are currently inside a vuir block, this is Some
	pub vuir_block: Option<VuirBlockAnalysisData>,
	pub comptime: bool,

	/// we are a block representing some inlined operation
	pub inlined: bool,

	/// The base type name we use to generate other type names if needed
	pub base_type_name: Intern<str>,

	/// Function parameters filled for DeclFn analysis
	pub decl_fn_params: Vec<DeclFnParam>,

	/// Effect handlers in scope: (effect_type_idx, handler_value_ref)
	pub handler_stack: Vec<(value::Index, vtir::InstructionRef)>,

	pub capture_context: BlockCaptureContext,
}

#[repr(transparent)]
struct ScopedBlock(BlockId);
impl Deref for ScopedBlock {
	type Target = BlockId;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
impl Drop for ScopedBlock {
	fn drop(&mut self) {
		// HACK(zino): LE COMPILO EST BOURRE ET MET DES DROP POUR RIEN APRES DES MOVE
		// panic!("a ScopedBlock must be dropped with unstack_block()")
	}
}

#[derive(Debug)]
struct ComptimeAlloc {
	ty: value::Index,
	value: ComptimeAllocValue,
	span: Span,
}

#[derive(Debug)]
pub enum ComptimeAllocValue {
	Interned(value::Index),
}

struct ComptimeMemory<'cu> {
	cu: &'cu Arc<CompilationUnit>,
	allocs: IndexVec<ComptimeAllocId, ComptimeAlloc>,
}
impl<'cu> ComptimeMemory<'cu> {
	pub fn allocate(
		&mut self,
		ty: value::Index,
		span: Span,
	) -> ComptimeAllocId {
		let value = ComptimeAllocValue::Interned(self.cu.values.intern_trivial(&value::Key::Undefined { ty }));
		let id = ComptimeAllocId(self.allocs.len());
		self.allocs.push(ComptimeAlloc { ty, value, span });
		id
	}
}

/// =========== ALLOCS
#[derive(Debug, Default)]
struct PotentialComptimeAlloc {
	stores: Vec<(vtir::InstructionRef, Span)>,
}

/// Allocs related data structures
struct Allocs {
	potential_comptime_allocs: FxHashMap<vtir::InstructionRef, PotentialComptimeAlloc>,
}

// ============ LINEAR TYPE TRACKING

#[derive(Clone, Debug)]
struct LinearSlot {
	/// The pointee type of the linear value
	ty: value::Index,
	/// Where the linear value was created (for error reporting)
	span: Span,
	/// Whether this linear value has been consumed (moved out)
	consumed: bool,
	/// Where the linear value was consumed (for error reporting)
	consumed_at: Option<Span>,
	/// The name of the variable (for error messages)
	name: Intern<str>,
}

// ============ SEMA

pub struct Sema<'a> {
	cu: &'a Arc<CompilationUnit>,
	vuir: &'a Vuir,
	module: ModuleId,
	owner_decl: DeclId,
	fun: Option<value::Index>,
	instructions: IndexVec<vtir::InstructionId, vtir::Opcode>,
	pub instructions_payload_alloc: &'static Bump,
	pub vuir_map: FxHashMap<vuir::InstructionId, vtir::InstructionRef>,
	pending_inferred_alloc_to_ty: FxHashMap<vtir::InstructionRef, Option<value::Index>>,
	pub blocks: IndexVec<BlockId, Block>,
	allocs: Allocs,
	comptime_memory: ComptimeMemory<'a>,
	/// Tracks linear values by their stack allocation instruction.
	/// Key is the VTIR InstructionRef for the StackAlloc.
	linear_slots: FxHashMap<vtir::InstructionRef, LinearSlot>,
	/// When analyzing an effect handler body, the expected return type of the operation.
	/// Used by EffectResume to coerce the resume value.
	effect_handler_ret_ty: Option<value::Index>,
}

impl<'a> Sema<'a> {
	fn diag_span(
		&self,
		span: Span,
	) -> DiagSpan {
		DiagSpan { module: self.module, span }
	}

	pub fn new(
		cu: &'a Arc<CompilationUnit>,
		vuir: &'a Vuir,
		module: ModuleId,
		owner_decl: DeclId,
		fun: Option<value::Index>,
	) -> Self {
		Self {
			cu,
			vuir,
			module,
			owner_decl,
			fun,
			instructions: IndexVec::new(),
			instructions_payload_alloc: Box::leak(Box::new(Bump::new())),
			vuir_map: FxHashMap::default(),
			pending_inferred_alloc_to_ty: FxHashMap::default(),
			blocks: IndexVec::new(),
			comptime_memory: ComptimeMemory {
				cu,
				allocs: IndexVec::default(),
			},
			allocs: Allocs {
				potential_comptime_allocs: FxHashMap::default(),
			},
			linear_slots: FxHashMap::default(),
			effect_handler_ret_ty: None,
		}
	}

	fn namespace_captures(
		&self,
		owner_type: value::Index,
	) -> Option<&'static [value::Capture]> {
		let value::Key::Type(ty) = self.cu.values.index_to_key(owner_type) else {
			return None;
		};
		match ty {
			value::Type::Struct(ns) | value::Type::Enum(ns) | value::Type::Union(ns) => Some(ns.captures),
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
			| value::Type::EnumLiteral => None,
		}
	}

	fn resolve_vuir_capture(
		&mut self,
		block: BlockId,
		capture: &vuir::Capture,
	) -> Result<value::Capture, AnalyzeError> {
		Ok(match capture {
			vuir::Capture::Id(id) => match self.resolve_inst(&id.as_ref()).as_interned_opt() {
				Some(index) => value::Capture::Comptime(index),
				None => value::Capture::Runtime(GlobalVuirInstructionId {
					module: self.module,
					inst: *id,
				}),
			},
			vuir::Capture::FromParent(parent) => {
				let owner_type = self
					.cu
					.namespaces
					.with(|namespaces| namespaces[self.blocks[block].namespace].owner_type);
				let captures = self.namespace_captures(owner_type).ok_or_else(|| {
					self.push_error(Diagnostic::error().with_message("capture forwarding requires a namespace owner with captures"));
					AnalyzeError::AnalysisFailed
				})?;
				captures[*parent]
			},
		})
	}

	fn resolve_vuir_captures(
		&mut self,
		block: BlockId,
		captures: &[vuir::Capture],
	) -> Result<&'static [value::Capture], AnalyzeError> {
		let mut resolved = Vec::with_capacity(captures.len());
		for capture in captures {
			resolved.push(self.resolve_vuir_capture(block, capture)?);
		}
		Ok(self.cu.values.alloc_slice(&resolved))
	}

	fn effect_handler_physical_fn_ty(
		&self,
		source_fn_ty: value::Index,
	) -> value::Index {
		let source_fn_ty = self.cu.values.index_to_key(source_fn_ty).as_type_fn();
		let env_ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
			pointee_ty: self.cu.values.common.void_t,
			packed: None,
			is_const: false,
		})));
		let mut params = Vec::with_capacity(source_fn_ty.params.len() + 1);
		params.push(env_ptr_ty);
		params.extend_from_slice(source_fn_ty.params);
		let mut comptime_params = BitVec::<u8>::with_capacity(source_fn_ty.comptime_params.len() + 1);
		comptime_params.push(false);
		comptime_params.extend(source_fn_ty.comptime_params.iter().by_vals());
		self.cu.values.intern_trivial(&value::Key::Type(value::Type::Fn(TypeFn {
			params: self.cu.values.alloc_slice(&params),
			comptime_params: self.cu.values.alloc_bitslice(&comptime_params),
			first_positional_param: source_fn_ty.first_positional_param.map(|idx| idx + 1),
			var_args: source_fn_ty.var_args,
			ret_ty: source_fn_ty.ret_ty,
			external: source_fn_ty.external,
			callconv: source_fn_ty.callconv,
			inline: source_fn_ty.inline,
		})))
	}

	pub fn finish(
		mut self,
		main_block: BlockId,
	) -> Vtir {
		let main_block = self.blocks.remove(main_block);
		Vtir {
			main_body: main_block.instructions.into_bump_slice(),
			instructions: self.instructions,
			// SAFETY: tkt
			instructions_payload_allocator: unsafe { Box::from_raw(std::ptr::from_ref(self.instructions_payload_alloc) as *mut _) },
		}
	}

	/// Emits a VTIR instruction into the current body
	pub fn inst(
		&mut self,
		block: BlockId,
		opcode: vtir::Opcode,
	) -> vtir::InstructionRef {
		#[allow(clippy::overly_complex_bool_expr)]
		{
			assert!(
				true || !self.blocks[block].comptime,
				"tried adding a VTIR instruction in a comptime block"
			);
		}
		self.instructions.push(opcode);
		let inst = vtir::InstructionRef::Instruction(vtir::InstructionId::from_usize(self.instructions.len() - 1));
		self.blocks[block].instructions.push(inst);
		inst
	}

	fn inst_id(
		&mut self,
		block: BlockId,
		opcode: vtir::Opcode,
	) -> vtir::InstructionId {
		#[allow(clippy::overly_complex_bool_expr)]
		{
			assert!(
				true || !self.blocks[block].comptime,
				"tried adding a VTIR instruction in a comptime block"
			);
		}
		self.instructions.push(opcode);
		let inst = vtir::InstructionId::from_usize(self.instructions.len() - 1);
		self.blocks[block].instructions.push(vtir::InstructionRef::Instruction(inst));
		inst
	}

	fn child_block(
		&mut self,
		parent: BlockId,
	) -> ScopedBlock {
		let namespace = self.blocks[parent].namespace;
		let capture_context = self.blocks[parent].capture_context.clone();
		self.blocks.push(Block {
			namespace,
			parent: Some(parent),
			instructions: BumpVec::new_in(self.instructions_payload_alloc),
			vuir_block: None,
			comptime: false,
			inlined: self.blocks[parent].inlined,
			base_type_name: self.blocks[parent].base_type_name,
			decl_fn_params: vec![],
			handler_stack: vec![],
			capture_context,
		});
		ScopedBlock(BlockId(self.blocks.len() - 1))
	}

	fn child_block_from_vuir_block(
		&mut self,
		parent: BlockId,
		vuir_block: VuirBlockAnalysisData,
	) -> ScopedBlock {
		let is_const = self.blocks[parent].comptime;
		let namespace = self.blocks[parent].namespace;
		let capture_context = self.blocks[parent].capture_context.clone();
		self.blocks.push(Block {
			namespace,
			parent: Some(parent),
			instructions: BumpVec::new_in(self.instructions_payload_alloc),
			vuir_block: Some(vuir_block),
			comptime: is_const,
			inlined: self.blocks[parent].inlined,
			base_type_name: self.blocks[parent].base_type_name,
			decl_fn_params: Default::default(),
			handler_stack: vec![],
			capture_context,
		});
		ScopedBlock(BlockId(self.blocks.len() - 1))
	}

	fn unstack_block(
		&mut self,
		block: ScopedBlock,
	) -> Block {
		let block = {
			let id = block.0;
			core::mem::forget(block);
			id
		};
		assert_eq!(
			&self.blocks[block] as *const _,
			self.blocks.last().unwrap() as *const _,
			"cannot unstack a block that isn't at the top of the stack"
		);
		self.blocks.remove(block)
	}

	fn lookup_decl_in_namespace_recursively(
		&self,
		namespace: NamespaceId,
		name: Intern<str>,
	) -> Option<DeclId> {
		let namespaces = self.cu.namespaces.read();
		let mut namespace = namespace;
		loop {
			if let Some(decl) = namespaces[namespace].decls.get(&name) {
				break Some(*decl);
			}
			let Some(parent) = namespaces[namespace].parent else {
				break None;
			};
			namespace = parent;
		}
	}

	fn lookup_decl_in_namespace(
		&self,
		namespace: NamespaceId,
		name: Intern<str>,
	) -> Option<DeclId> {
		let namespaces = self.cu.namespaces.read();
		let mut namespace = namespace;
		namespaces[namespace].decls.get(&name).copied()
	}

	/// Walk block stack looking for a handler for `effect_ty`. Returns the InstructionRef to the
	/// handler (EffectHandler interned value or runtime FnParam instruction).
	pub(super) fn find_handler_in_block_recursively(
		&self,
		block: BlockId,
		effect_ty: value::Index,
	) -> Option<vtir::InstructionRef> {
		let mut cur = Some(block);
		while let Some(b) = cur {
			if let Some((_, ref_)) = self.blocks[b].handler_stack.iter().rev().find(|(ety, _)| *ety == effect_ty) {
				return Some(*ref_);
			}
			cur = self.blocks[b].parent;
		}
		None
	}

	fn resolve_builtin_calling_convention_type(&mut self) -> Result<value::Index, AnalyzeError> {
		// TODO(ldubos): maybe cache this?

		let Some(builtin_namespace) = self
			.cu
			.modules
			.with(|modules| modules[self.cu.builtin_module].namespace.get().copied())
		else {
			unreachable!("internal compiler error: `builtin` module does not have a namespace");
		};

		let calling_convention_symbol = COMMON_INTERNS.calling_convention_symbol;
		let decl = self
			.cu
			.namespaces
			.with(|namespaces| namespaces[builtin_namespace].decls.get(&calling_convention_symbol).copied());

		let Some(decl) = decl else {
			unreachable!("internal compiler error: `builtin` module does not have a declaration for CallingConvention");
		};

		let Some(ty) = self.cu.get_or_analyze_decl_value(decl)? else {
			return Err(AnalyzeError::AnalysisFailed);
		};

		let value::Key::Type(value::Type::Enum(_)) = self.cu.values.index_to_key(ty) else {
			unreachable!("internal compiler error: `CallingConvention` declaration in `builtin` module is not an enum");
		};

		Ok(ty)
	}

	/// Analyze a body of instruction
	fn analyze_instruction_block(
		&mut self,
		block: BlockId,
		instructions: &[vuir::InstructionId],
	) -> Result<ControlFlow, AnalyzeError> {
		let mut body_flow = ControlFlow::May;
		for &inst in instructions {
			match self.analyze_inst(block, inst)? {
				(vtir::InstructionRef::Interned(value), _) if value == self.cu.values.common.unreachable_value => {
					body_flow = ControlFlow::Always;
					break;
				},
				(_, ControlFlow::Always) => {
					body_flow = ControlFlow::Always;
					break;
				},
				(_, inst_flow) => {
					body_flow = body_flow.join(inst_flow);
				},
			}
		}
		Ok(body_flow)
	}

	pub fn analyze_fn_body(
		&mut self,
		block: BlockId,
		body: &[vuir::InstructionId],
		ret_ty: value::Index,
	) -> Result<(), AnalyzeError> {
		let body_flow = self.analyze_instruction_block(block, body)?;

		// after body analysis, the control flow MUST not be always returns
		// or else it means we are missing a return
		if body_flow != ControlFlow::Always {
			// implicit return for void ret ty
			if ret_ty == self.cu.values.common.void_t {
				self.inst(block, vtir::Opcode::Return { value: None });
			} else {
				self.push_error(Diagnostic::error().with_message("function with non-void return type does not return on all paths"));
			}
		}

		// Check that all linear values have been consumed
		let unconsumed: Vec<_> = self
			.linear_slots
			.values()
			.filter(|slot| !slot.consumed)
			.map(|slot| (slot.name, slot.ty, slot.span))
			.collect();
		for (name, ty, span) in unconsumed {
			self.push_error(
				Diagnostic::error()
					.with_message(format!("linear value `{name}` must be consumed before going out of scope",))
					.with_label(
						Label::primary()
							.with_span(self.diag_span(span))
							.with_message(format!("`{name}` has type `{}` which is linear", self.cu.values.display_index(ty),)),
					)
					.with_note("linear values cannot be implicitly dropped; pass to a consuming function or use @forget"),
			);
		}

		Ok(())
	}

	pub fn analyze_fn_body_at_comptime(
		&mut self,
		block: BlockId,
		body: &[vuir::InstructionId],
		ret_ty: value::Index,
		caller_span: DiagSpan,
	) -> Result<Option<vtir::InstructionRef>, AnalyzeError> {
		let value = match self.analyze_instruction_block(block, body) {
			Ok(_) => {
				// No explicit return was hit. Insert the implicit void return.
				if ret_ty == self.cu.values.common.void_t {
					Ok(Some(self.cu.values.common.void_t.into()))
				} else {
					self.push_error(
						Diagnostic::error()
							.with_message("function with non-void return type does not return on all paths")
							.with_label(Label::primary().with_span(caller_span)),
					);
					Err(AnalyzeError::AnalysisFailed)
				}
			},
			Err(AnalyzeError::InlineReturn { value }) => Ok(value),
			Err(e) => Err(e),
		};

		// Check that all linear values have been consumed
		let unconsumed: Vec<_> = self
			.linear_slots
			.values()
			.filter(|slot| !slot.consumed)
			.map(|slot| (slot.name, slot.ty, slot.span))
			.collect();
		for (name, ty, span) in unconsumed {
			self.push_error(
				Diagnostic::error()
					.with_message(format!("linear value `{name}` must be consumed before going out of scope",))
					.with_label(
						Label::primary()
							.with_span(self.diag_span(span))
							.with_message(format!("`{name}` has type `{}` which is linear", self.cu.values.display_index(ty),)),
					)
					.with_note("linear values cannot be implicitly dropped; pass to a consuming function or use @forget"),
			);
		}

		value
	}

	/// Analyze a builtin function body.
	/// Builtins are special, they are always inlined and therefore does not need any return instruction
	pub fn analyze_fn_builtin_body(
		&mut self,
		block: BlockId,
		fun: &FnKey,
		builtin_kind: vuir::BuiltinKind,
		caller_span: DiagSpan,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let fun_ty = self.cu.values.index_to_key(fun.ty).as_type_fn();
		let fun_decl = self.cu.values.index_to_key(fun.decl).as_fn_decl();
		let func_vuir_info = self.get_vuir_fn_info(fun_decl);

		match builtin_kind {
			vuir::BuiltinKind::UnsafeIntCast => {
				let src = self.resolve_inst(&func_vuir_info.params[2].as_ref());
				let result = self.inst(block, vtir::Opcode::UnsafeIntCast {
					src,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::SizeOf => {
				let value_ty = self.resolve_type(block, &func_vuir_info.params[0].as_ref(), &caller_span.span)?;
				let result = self.inst(block, vtir::Opcode::SizeOf {
					of: InstructionRef::Interned(value_ty),
				});
				Ok(result)
			},
			vuir::BuiltinKind::BitSizeOf => {
				let value_ty = self.resolve_type(block, &func_vuir_info.params[0].as_ref(), &caller_span.span)?;

				let known_bits = self.known_bit_size(value_ty);
				let result = if let Some(bits) = known_bits {
					let val = self.cu.values.intern_trivial(&value::Key::Int {
						ty: self.cu.values.common.usize_t,
						value: Intern::new(Anyint::from(bits)),
					});
					vtir::InstructionRef::Interned(val)
				} else {
					// Fallback to @size_of(T) * 8
					let size_of = self.inst(block, vtir::Opcode::SizeOf {
						of: InstructionRef::Interned(value_ty),
					});
					let eight = InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int {
						ty: self.cu.values.common.usize_t,
						value: Intern::new(Anyint::from(8u64)),
					}));
					self.inst(block, vtir::Opcode::Mul { lhs: size_of, rhs: eight })
				};

				Ok(result)
			},
			vuir::BuiltinKind::Zeroed => {
				let result = self.inst(block, vtir::Opcode::Zeroed { ty: fun_ty.ret_ty });
				Ok(result)
			},
			vuir::BuiltinKind::IntFromEnum => {
				let value = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let value_ty = self.type_of(&value);
				let value_ty = self.cu.values.index_to_key(value_ty);
				let value::Value::Enum(type_enum) = self.cu.values.index_to_value(self.type_of(&value)) else {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected an enum, found `{value_ty:?}`"))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				};
				let type_enum = type_enum.as_ref();

				// ensure return type can be coerced to tag
				self.coerce(
					block,
					type_enum.tag_ty,
					vtir::InstructionRef::Interned(fun_ty.ret_ty),
					&caller_span.span,
				)?;

				let result = self.inst(block, vtir::Opcode::BitCast {
					src: value,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::IntToFloat => {
				let src = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let result = self.inst(block, vtir::Opcode::IntToFloat {
					src,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::Import => {
				let path = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let path = self.cu.values.index_to_key(path.as_interned());
				let path = match path {
					value::Key::Str { slice_ty: ty, value } => str::from_utf8(value.as_ref()).unwrap(),
					_ => unreachable!(
						"encountered a @import with a non-lit str arg, should be checked by ast -> vuir pass: {path:?} vuir id",
					),
				};

				let Some(module) = self.cu.modules.with(|modules| match path {
					"root" => Some(self.cu.root_module),
					"std" => Some(self.cu.std_module),
					"builtin" => Some(self.cu.builtin_module),
					_ => {
						let src_module_path = &modules[caller_span.module].path;
						let module_path = src_module_path.parent().unwrap().join(path).normalize();
						self.cu
							.module_path_to_id
							.with_mut(|module_path_to_id| module_path_to_id.get(&module_path).copied())
					},
				}) else {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("module `{}` not found", path))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				};

				let value = self.cu.get_or_analyze_module_sync(module)?;
				Ok(value.into())
			},
			vuir::BuiltinKind::Nullptr => {
				let ptr_ty = fun_ty.ret_ty;
				if !matches!(self.cu.values.index_to_key(ptr_ty), value::Key::Type(value::Type::Ptr(_))) {
					self.push_error(
						Diagnostic::error()
							.with_message(format!(
								"return type must be a pointer, is {}",
								self.cu.values.display_index(ptr_ty)
							))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}

				let null = self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
					ty: ptr_ty,
					kind: PtrKind::Value(self.cu.values.common.nullptr),
				}));
				Ok(null.into())
			},
			vuir::BuiltinKind::SliceFromRawParts => {
				let pointee_ty = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let ptr = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let len = self.resolve_inst(&func_vuir_info.params[2].as_ref());
				let slice_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Slice(TypeSlice {
					pointee_ty: pointee_ty.as_interned(),
				})));
				let slice = self.inst(block, vtir::Opcode::SliceFromRawParts { slice_ty, ptr, len });
				Ok(slice)
			},
			vuir::BuiltinKind::SlicePtr => {
				let pointee_ty = self.resolve_inst(&func_vuir_info.params[0].as_ref()).as_interned();
				let slice = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
					pointee_ty,
					packed: None,
					is_const: true,
				})));
				let ptr = self.inst(block, vtir::Opcode::SlicePtr { slice, ptr_ty });
				Ok(ptr)
			},
			vuir::BuiltinKind::SliceLen => {
				let slice = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let len = self.inst(block, vtir::Opcode::SliceLen { slice });
				Ok(len)
			},
			vuir::BuiltinKind::Abort => {
				if self.blocks[block].comptime {
					self.push_error(
						Diagnostic::error()
							.with_message("cannot @abort in comptime")
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				self.inst(block, vtir::Opcode::Abort);
				self.inst(block, vtir::Opcode::Unreachable);
				Ok(self.cu.values.common.unreachable_value.into())
			},
			vuir::BuiltinKind::Unreachable => {
				if self.blocks[block].comptime {
					self.push_error(
						Diagnostic::error()
							.with_message("reached a @unreachable in comptime")
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				self.inst(block, vtir::Opcode::Unreachable);
				Ok(self.cu.values.common.unreachable_value.into())
			},
			vuir::BuiltinKind::PtrToInt => {
				let src = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let result = self.inst(block, vtir::Opcode::PtrToInt {
					src,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::IntToPtr => {
				let src = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let result = self.inst(block, vtir::Opcode::IntToPtr {
					src,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::Forget => {
				// @forget consumes a linear value and does nothing with it.
				// The value is already consumed at the call site via the Load.
				// Just return void.
				Ok(self.cu.values.common.void_value.into())
			},
			vuir::BuiltinKind::Bitcast => {
				// ensure preconditions
				let value = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let src_ty = self.type_of(&value);
				let src_byte_size = self.cu.values.type_layout(&self.cu.resolved_target, src_ty).size;
				let dst_byte_size = self.cu.values.type_layout(&self.cu.resolved_target, fun_ty.ret_ty).size;
				if src_byte_size != dst_byte_size {
					self.push_error(
						Diagnostic::error()
							.with_message("cannot bitcast to a differently sized type")
							.with_label(Label::primary().with_span(caller_span))
							.with_note(format!(
								"source type `{}` takes {} bytes",
								self.cu.values.display_index(src_ty),
								src_byte_size
							))
							.with_note(format!(
								"destination type `{}` takes {} bytes",
								self.cu.values.display_index(fun_ty.ret_ty),
								dst_byte_size
							)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}

				let result = self.inst(block, vtir::Opcode::BitCast {
					src: value,
					dst_ty: fun_ty.ret_ty,
				});
				Ok(result)
			},
			vuir::BuiltinKind::SliceCopyNonoverlapping => {
				let src = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let dst = self.resolve_inst(&func_vuir_info.params[2].as_ref());
				let result = self.inst(block, vtir::Opcode::SliceCopyNonoverlapping {
					slice_ty: self.type_of(&src),
					src,
					dst,
				});
				Ok(result)
			},
			vuir::BuiltinKind::AnyptrIs => {
				let target_ty = self.resolve_type(block, &func_vuir_info.params[0].as_ref(), &caller_span.span)?;
				self.analyze_type_info(target_ty)?;
				let value = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let value_ty = self.type_of(&value);
				if value_ty != self.cu.values.common.anyptr_t {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected `anyptr`, found `{}`", self.cu.values.display_index(value_ty)))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				Ok(self.inst(block, vtir::Opcode::AnyptrIs { value, target_ty }))
			},
			vuir::BuiltinKind::AnyptrAs => {
				let target_ty = self.resolve_type(block, &func_vuir_info.params[0].as_ref(), &caller_span.span)?;
				let value = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				let value_ty = self.type_of(&value);
				if value_ty != self.cu.values.common.anyptr_t {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected `anyptr`, found `{}`", self.cu.values.display_index(value_ty)))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				Ok(self.inst(block, vtir::Opcode::AnyptrAs { value, target_ty }))
			},
			vuir::BuiltinKind::AnyptrPtr => {
				let value = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let value_ty = self.type_of(&value);
				if value_ty != self.cu.values.common.anyptr_t {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected `anyptr`, found `{}`", self.cu.values.display_index(value_ty)))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				Ok(self.inst(block, vtir::Opcode::AnyptrPtr {
					value,
					ptr_ty: fun_ty.ret_ty,
				}))
			},
			vuir::BuiltinKind::AnyptrFromRaw => {
				let ptr = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let type_id = self.resolve_inst(&func_vuir_info.params[1].as_ref());
				Ok(self.inst(block, vtir::Opcode::AnyptrFromRaw { ptr, type_id }))
			},
			vuir::BuiltinKind::AnyptrTypeInfo => {
				let value = self.resolve_inst(&func_vuir_info.params[0].as_ref());
				let value_ty = self.type_of(&value);
				if value_ty != self.cu.values.common.anyptr_t {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected `anyptr`, found `{}`", self.cu.values.display_index(value_ty)))
							.with_label(Label::primary().with_span(caller_span)),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				let ty = self.cu.builtin_type_info()?;
				Ok(self.inst(block, vtir::Opcode::AnyptrTypeInfo { value, ty }))
			},
			vuir::BuiltinKind::TypeInfo => {
				let ty = self.resolve_type(block, &func_vuir_info.params[0].as_ref(), &caller_span.span)?;
				let type_info_id = self.analyze_type_info(ty)?;
				Ok(self.cu.type_info_entries[type_info_id].into())
			},
		}
	}

	pub fn analyze_comptime_block(
		&mut self,
		block: BlockId,
		instructions: &[vuir::InstructionId],
	) -> Result<Option<vtir::InstructionRef>, AnalyzeError> {
		if let Err(err) = self.analyze_instruction_block(block, instructions) {
			match err {
				AnalyzeError::ComptimeBreak { value, .. } => Ok(Some(value)),
				AnalyzeError::InlineReturn { value } => Ok(value),
				e => Err(e),
			}
		} else {
			Ok(None)
		}
	}

	#[track_caller]
	fn analyze_load(
		&mut self,
		block: BlockId,
		src: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let src_ty = self.type_of(&src);

		// Linear type move tracking: loading from a linear-typed pointer is a move
		if let value::Key::Type(value::Type::Ptr(ptr)) = self.cu.values.index_to_key(src_ty)
			&& self.cu.values.type_is_linear(ptr.pointee_ty)
			&& let Some(slot) = self.linear_slots.get(&src)
		{
			if slot.consumed {
				let name = slot.name;
				let decl_span = slot.span;
				let consumed_at = slot.consumed_at;
				let mut diag = Diagnostic::error()
					.with_message(format!("use of moved linear value `{name}`",))
					.with_label(Label::primary().with_span(self.diag_span(*span)))
					.with_label(
						Label::secondary()
							.with_span(self.diag_span(decl_span))
							.with_message("linear value declared here"),
					);
				if let Some(consumed_at) = consumed_at {
					diag = diag.with_label(
						Label::secondary()
							.with_span(self.diag_span(consumed_at))
							.with_message("value consumed here"),
					);
				}
				self.push_error(diag);
				return Err(AnalyzeError::AnalysisFailed);
			}
			let slot = self.linear_slots.get_mut(&src).unwrap();
			slot.consumed = true;
			slot.consumed_at = Some(*span);
		}

		// ensure src pointee is a pointer, else the dereference syntax does not make sense
		if !matches!(self.cu.values.index_to_key(src_ty), value::Key::Type(value::Type::Ptr(..))) {
			self.push_error(
				Diagnostic::error()
					.with_message(format!("load expected a ptr but found `{}`", self.cu.values.display_index(src_ty)))
					.with_label(Label::primary().with_span(self.diag_span(*span))),
			);
			return Err(AnalyzeError::AnalysisFailed);
		}

		// if we load from a comptime pointer, do not emit any load
		let inst = if let Some(value) = self.try_resolve_comptime_value(&src) {
			let value = match self.cu.values.index_to_key(value).as_ptr().kind {
				PtrKind::Value(value) => value,
				PtrKind::Decl(decl) => self.cu.decls.with_mut(|decls| {
					let DeclAnalysisState::Analysed { value } = &decls[decl].analysis_state else {
						unreachable!("tried dereferencing a decl pointer to {decl:?} that is not analysed");
					};
					*value
				}),
				PtrKind::ComptimeAlloc(alloc) => match self.comptime_memory.allocs[alloc].value {
					ComptimeAllocValue::Interned(value) => value,
				},
			};
			vtir::InstructionRef::Interned(value)
		} else {
			self.inst(block, vtir::Opcode::Load { ptr: src })
		};
		Ok(inst)
	}

	/// Register a function parameter as a linear slot for tracking.
	pub fn register_linear_param(
		&mut self,
		vtir_inst: vtir::InstructionRef,
		name: Intern<str>,
		ty: value::Index,
		span: Span,
	) {
		self.linear_slots.insert(vtir_inst, LinearSlot {
			ty,
			span,
			consumed: false,
			consumed_at: None,
			name,
		});
	}

	/// Try to consume a linear value (for value-type linear params).
	/// Returns Err if the value has already been consumed (use-after-move).
	fn try_consume_linear_value(
		&mut self,
		value: vtir::InstructionRef,
		span: &Span,
	) -> Result<(), AnalyzeError> {
		let ty = self.type_of(&value);
		if self.cu.values.type_is_linear(ty)
			&& let Some(slot) = self.linear_slots.get(&value)
		{
			if slot.consumed {
				let name = slot.name;
				let decl_span = slot.span;
				let consumed_at = slot.consumed_at;
				let mut diag = Diagnostic::error()
					.with_message(format!("use of moved linear value `{name}`",))
					.with_label(Label::primary().with_span(self.diag_span(*span)))
					.with_label(
						Label::secondary()
							.with_span(self.diag_span(decl_span))
							.with_message("linear value declared here"),
					);
				if let Some(consumed_at) = consumed_at {
					diag = diag.with_label(
						Label::secondary()
							.with_span(self.diag_span(consumed_at))
							.with_message("value consumed here"),
					);
				}
				self.push_error(diag);
				return Err(AnalyzeError::AnalysisFailed);
			}
			let slot = self.linear_slots.get_mut(&value).unwrap();
			slot.consumed = true;
			slot.consumed_at = Some(*span);
		}
		Ok(())
	}

	fn analyze_inst(
		&mut self,
		block: BlockId,
		id: vuir::InstructionId,
	) -> Result<(vtir::InstructionRef, ControlFlow), AnalyzeError> {
		match &self.vuir.instructions[id] {
			// Items
			vuir::Opcode::DeclStruct {
				naming,
				fields,
				packed,
				linear,
				decls,
				captures,
			} => {
				// insert the struct key with a none value so we have an index to give for our namespaces
				let (struct_key, struct_idx) = {
					let captures = self.resolve_vuir_captures(block, captures)?;
					let struct_key = value::Key::Type(value::Type::Struct(value::NamespaceType {
						inst: GlobalVuirInstructionId {
							module: self.module,
							inst: id,
						},
						captures,
					}));
					let struct_idx = self.cu.values.intern_non_trivial(&struct_key, value::Value::none());
					(struct_key, struct_idx)
				};

				// parse struct fields & decls
				let namespace = if id == vuir::InstructionId::FILE_MODULE {
					self.blocks[block].namespace
				} else {
					self.cu
						.namespaces
						.with_mut(|namespaces| namespaces.push(Namespace::with_parent(self.blocks[block].namespace, struct_idx)))
				};

				let block = self.child_block(block);
				self.blocks[*block].namespace = namespace;

				if !decls.is_empty() {
					let mut namespace_decls = FxHashMap::default();
					let mut cu_decls = self.cu.decls.lock();
					{
						for decl_id in decls {
							let vuir::Opcode::Declaration(decl) = self.vuir.instructions[*decl_id].clone() else {
								unreachable!()
							};

							let decl_id = cu_decls.push(Decl {
								name: decl.name,
								module: self.module,
								namespace,
								analysis_state: DeclAnalysisState::Unanalysed {
									module: self.module,
									vuir_id: *decl_id,
								},
							});
							namespace_decls.insert(decl.name, decl_id);
						}
					}

					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls = namespace_decls;
					});
				}

				let fields = fields
					.iter()
					.map(|field| {
						let ty = match &field.ty {
							vuir::FieldTy::Ref(r) => self.resolve_type(*block, r, &field.name.span)?,
							vuir::FieldTy::Body(instructions) => {
								let ty = self
									.analyze_comptime_block(*block, instructions)?
									.expect("a field type body must return a value");
								ty.as_interned()
							},
						};

						Ok(StructField {
							name: field.name.symbol,
							ty,
						})
					})
					.try_collect::<Vec<_>>()?;
				let layout = if *packed {
					for f in &fields {
						let (key, value) = self.cu.values.index_to_key_value(f.ty);
						let value::Key::Type(ty) = key else {
							return Err(AnalyzeError::AnalysisFailed);
						};
						match ty {
							value::Type::Isize
							| value::Type::Usize
							| value::Type::F16
							| value::Type::F32
							| value::Type::F64
							| value::Type::F128
							| value::Type::Bool
							| value::Type::Void
							| value::Type::Int { .. } => {},
							value::Type::Struct(..) if matches!(value.as_struct().as_ref().layout, value::StructLayout::Packed { .. }) => {
							},
							value::Type::Anyint
							| value::Type::Anyfloat
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
							| value::Type::Never
							| value::Type::EnumLiteral => return Err(AnalyzeError::AnalysisFailed),
						}
					}

					let mut bit_offset = 0u32;
					let packed_fields = fields
						.iter()
						.map(|f| {
							let (value::Key::Type(ty), value) = self.cu.values.index_to_key_value(f.ty) else {
								unreachable!("packed struct field is not a type")
							};
							match ty {
								value::Type::Int { bits, .. } => {
									let bit_width = *bits as u32;
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: bit_width,
									};
									bit_offset += bit_width;
									info
								},
								value::Type::Bool => {
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: 1,
									};
									bit_offset += 1;
									info
								},
								value::Type::F16 => {
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: 16,
									};
									bit_offset += 16;
									info
								},
								value::Type::F32 => {
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: 32,
									};
									bit_offset += 32;
									info
								},
								value::Type::F64 => {
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: 64,
									};
									bit_offset += 64;
									info
								},
								value::Type::F128 => {
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: 128,
									};
									bit_offset += 128;
									info
								},
								value::Type::Isize | value::Type::Usize => {
									let ptr_size = self.cu.resolved_target.ptr_width_in_bits as _;
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: ptr_size,
									};
									bit_offset += ptr_size;
									info
								},
								value::Type::Void => PackedStructFieldInfo {
									offset: bit_offset,
									width: 0,
								},
								value::Type::Struct(_) => {
									let value::Value::Struct(r#struct) = value else {
										unreachable!("struct type without struct value")
									};
									let StructLayout::Packed { fields_bits, .. } = r#struct.as_ref().layout else {
										unreachable!()
									};
									let bit_width = fields_bits;
									let info = PackedStructFieldInfo {
										offset: bit_offset,
										width: bit_width,
									};
									bit_offset += bit_width;
									info
								},
								value::Type::Anyint
								| value::Type::Anyfloat
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
								| value::Type::EnumLiteral => unreachable!("invalid packed struct field type"),
							}
						})
						.collect::<Vec<_>>();

					let storage_bits = bit_offset.div_ceil(8) * 8;

					StructLayout::Packed {
						storage_bits,
						fields_bits: bit_offset,
						packed_fields: self.cu.values.alloc_slice(&packed_fields),
					}
				} else {
					StructLayout::Standard
				};

				let fields_static = self.cu.values.alloc_slice(&fields);
				let struct_ty = self.cu.values.value_allocate(TypeStruct {
					name: self.make_type_name(*block, id, *naming),
					fields: fields_static,
					layout,
					namespace,
					linear: *linear,
				});

				// we have struct type built, insert it
				self.cu.values.intern_non_trivial(&struct_key, value::Value::Struct(struct_ty));

				// inject 'Self' into struct namespace, only worth if we have decls inside the struct
				// TODO(zino): will cause issues if fields references Self beforehand..
				if !decls.is_empty() {
					let self_decl_id = {
						let mut cu_decls = self.cu.decls.lock();
						cu_decls.push(Decl {
							name: COMMON_INTERNS.self_ty_symbol,
							module: self.module,
							namespace,
							analysis_state: DeclAnalysisState::Analysed { value: struct_idx },
						})
					};
					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls.insert(COMMON_INTERNS.self_ty_symbol, self_decl_id);
					});
				}

				self.vuir_map.insert(id, vtir::InstructionRef::Interned(struct_idx));

				self.unstack_block(block);

				Ok((vtir::InstructionRef::Interned(struct_idx), ControlFlow::May))
			},
			vuir::Opcode::DeclEnum {
				naming,
				tag_ty,
				linear,
				variants: fields,
				decls,
				captures,
			} => {
				// insert the struct key with a none value so we have an index to give for our namespaces
				let (enum_key, enum_idx) = {
					let captures = self.resolve_vuir_captures(block, captures)?;
					let enum_key = value::Key::Type(value::Type::Enum(value::NamespaceType {
						inst: GlobalVuirInstructionId {
							module: self.module,
							inst: id,
						},
						captures,
					}));
					let enum_idx = self.cu.values.intern_non_trivial(&enum_key, value::Value::none());
					(enum_key, enum_idx)
				};

				let namespace = self
					.cu
					.namespaces
					.with_mut(|namespaces| namespaces.push(Namespace::with_parent(self.blocks[block].namespace, enum_idx)));

				let block = self.child_block(block);
				self.blocks[*block].namespace = namespace;

				if !decls.is_empty() {
					let mut namespace_decls = FxHashMap::default();
					let mut cu_decls = self.cu.decls.lock();
					for decl_id in decls {
						let vuir::Opcode::Declaration(decl) = self.vuir.instructions[*decl_id].clone() else {
							unreachable!()
						};

						let decl_id = cu_decls.push(Decl {
							name: decl.name,
							module: self.module,
							namespace,
							analysis_state: DeclAnalysisState::Unanalysed {
								module: self.module,
								vuir_id: *decl_id,
							},
						});
						namespace_decls.insert(decl.name, decl_id);
					}

					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls = namespace_decls;
					});
				}

				let (tag_ty, had_type) = if let Some((tag_ty, tag_ty_span)) = tag_ty {
					let tag_ty = self.resolve_type(*block, tag_ty, tag_ty_span)?;
					// ensure tag_ty is numeric
					if !matches!(
						self.cu.values.index_to_key(tag_ty),
						value::Key::Type(value::Type::Int { .. }) | value::Key::Type(value::Type::Anyint)
					) {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"a enum tag can only be an integer, found `{}`",
									self.cu.values.display_index(tag_ty)
								))
								.with_label(Label::primary().with_span(self.diag_span(*tag_ty_span))),
						);
						return Err(AnalyzeError::AnalysisFailed);
					}
					(tag_ty, true)
				} else {
					// Determines how many bits are needed to represent the number of fields
					let bits = fields.len().next_power_of_two().trailing_zeros().max(1).min(u16::MAX as u32) as u16;

					(
						self.cu
							.values
							.intern_trivial(&value::Key::Type(value::Type::Int { signed: false, bits })),
						false,
					)
				};

				// TODO(zino): replace this with a more direct field-value lookup structure.
				let mut existing_value_to_field_span: HashMap<value::Index, Span> = HashMap::default();
				let fields = fields
					.iter()
					.enumerate()
					.map(|(i, field)| {
						if unlikely(field.value.is_some() && !had_type) {
							self.push_error(
								Diagnostic::error()
									.with_message("cannot assign a value to an enum field without a tag type")
									.with_label(
										Label::primary()
											.with_span(self.diag_span(field.ident.span))
											.with_message("assigned value here"),
									),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}

						let value = field
							.value
							.map(|(value, value_span)| {
								let value = self.resolve_inst(&value);
								let value = self.coerce(*block, tag_ty, value, &value_span)?;
								let value = self.try_resolve_comptime_value(&value).unwrap();
								Ok(value)
							})
							.transpose()?
							.unwrap_or_else(|| {
								self.cu.values.intern_trivial(&value::Key::Int {
									ty: tag_ty,
									value: Anyint::from(i).into(),
								})
							});

						if let Some(existing_span) = existing_value_to_field_span.insert(value, field.span) {
							self.push_error(
								Diagnostic::error()
									.with_message("enum value already taken")
									.with_label(
										Label::primary()
											.with_span(self.diag_span(field.span))
											.with_message("redefinition here"),
									)
									.with_label(
										Label::secondary()
											.with_span(self.diag_span(existing_span))
											.with_message("first definition here"),
									),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}

						Ok(EnumField {
							name: field.ident.symbol,
							value,
						})
					})
					.try_collect::<Vec<_>>()?;

				let fields_static = self.cu.values.alloc_slice(&fields);
				let enum_ty = self.cu.values.value_allocate(TypeEnum {
					name: self.make_type_name(*block, id, *naming),
					tag_ty,
					fields: fields_static,
					namespace,
					linear: *linear,
				});

				self.cu.values.intern_non_trivial(&enum_key, value::Value::Enum(enum_ty));

				if !decls.is_empty() {
					let self_decl_id = {
						let mut cu_decls = self.cu.decls.lock();
						cu_decls.push(Decl {
							name: COMMON_INTERNS.self_ty_symbol,
							module: self.module,
							namespace,
							analysis_state: DeclAnalysisState::Analysed { value: enum_idx },
						})
					};
					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls.insert(COMMON_INTERNS.self_ty_symbol, self_decl_id);
					});
				}

				self.vuir_map.insert(id, vtir::InstructionRef::Interned(enum_idx));
				self.unstack_block(block);
				Ok((vtir::InstructionRef::Interned(enum_idx), ControlFlow::May))
			},
			vuir::Opcode::DeclUnion {
				tag,
				naming,
				linear,
				fields,
				decls,
				captures,
			} => {
				let (captures, union_key, union_idx) = {
					let captures = self.resolve_vuir_captures(block, captures)?;
					let union_key = value::Key::Type(value::Type::Union(value::NamespaceType {
						inst: GlobalVuirInstructionId {
							module: self.module,
							inst: id,
						},
						captures,
					}));
					let union_idx = self.cu.values.intern_non_trivial(&union_key, value::Value::none());
					(captures, union_key, union_idx)
				};

				let namespace = self
					.cu
					.namespaces
					.with_mut(|namespaces| namespaces.push(Namespace::with_parent(self.blocks[block].namespace, union_idx)));

				let block = self.child_block(block);
				self.blocks[*block].namespace = namespace;

				if !decls.is_empty() {
					let mut namespace_decls = FxHashMap::default();
					let mut cu_decls = self.cu.decls.lock();
					{
						for decl_id in decls {
							let vuir::Opcode::Declaration(decl) = self.vuir.instructions[*decl_id].clone() else {
								unreachable!()
							};

							let decl_id = cu_decls.push(Decl {
								name: decl.name,
								module: self.module,
								namespace,
								analysis_state: DeclAnalysisState::Unanalysed {
									module: self.module,
									vuir_id: *decl_id,
								},
							});
							namespace_decls.insert(decl.name, decl_id);
						}
					}

					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls = namespace_decls;
					});
				}

				let union_name = self.make_type_name(*block, id, *naming);

				// Resolve field types
				let fields = fields
					.iter()
					.map(|field| {
						let ty = field
							.ty
							.as_ref()
							.map(|field_ty| {
								let ty = match field_ty {
									vuir::FieldTy::Ref(r) => self.resolve_type(*block, r, &field.name.span)?,
									vuir::FieldTy::Body(instructions) => {
										let ty = self
											.analyze_comptime_block(*block, instructions)?
											.expect("a field type body must return a value");
										ty.as_interned()
									},
								};
								Ok(ty)
							})
							.transpose()?;

						Ok(UnionField {
							name: field.name.symbol,
							ty,
						})
					})
					.try_collect::<Vec<_>>()?;

				// Determine tag type for tagged unions
				// tag: None = bare, Some(None) = auto-tagged enum, Some(Some(ref, span)) = explicit
				let tag_ty = match tag {
					None => None,
					Some(None) => {
						let bits = fields.len().next_power_of_two().trailing_zeros().max(1).min(u16::MAX as u32) as u16;
						let tag_ty = self
							.cu
							.values
							.intern_trivial(&value::Key::Type(value::Type::Int { signed: false, bits }));
						let enum_name = Intern::from(format!("{union_name}_AutoEnumTag").as_str());
						let enum_fields = self
							.cu
							.values
							.alloc_slice_fill_iter(fields.iter().enumerate().map(|(idx, field)| EnumField {
								name: field.name,
								value: self.cu.values.intern_trivial(&value::Key::Int {
									ty: tag_ty,
									value: Anyint::from(idx).into(),
								}),
							}));
						let enum_key = value::Key::Type(value::Type::Enum(value::NamespaceType {
							inst: GlobalVuirInstructionId {
								module: self.module,
								inst: id,
							},
							captures,
						}));
						let enum_idx = self.cu.values.intern_non_trivial(&enum_key, value::Value::none());
						let enum_namespace = self
							.cu
							.namespaces
							.with_mut(|namespaces| namespaces.push(Namespace::with_parent(namespace, enum_idx)));
						let enum_ty = self.cu.values.value_allocate(TypeEnum {
							name: enum_name,
							tag_ty,
							fields: enum_fields,
							namespace: enum_namespace,
							linear: *linear,
						});
						Some(self.cu.values.intern_non_trivial(&enum_key, value::Value::Enum(enum_ty)))
					},
					Some(Some((tag_ref, tag_span))) => {
						let tag_ty = self.resolve_type(*block, tag_ref, tag_span)?;
						if !matches!(self.cu.values.index_to_key(tag_ty), value::Key::Type(value::Type::Enum(..))) {
							self.push_error(
								Diagnostic::error()
									.with_message("union tag type must be an enum")
									.with_label(Label::primary().with_span(self.diag_span(*tag_span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}
						Some(tag_ty)
					},
				};

				let fields_static = self.cu.values.alloc_slice(&fields);
				let union_ty = self.cu.values.value_allocate(TypeUnion {
					name: union_name,
					tag_ty,
					fields: fields_static,
					namespace,
					linear: *linear,
				});

				self.cu.values.intern_non_trivial(&union_key, value::Value::Union(union_ty));

				// inject 'Self' into union namespace
				if !decls.is_empty() {
					let self_decl_id = {
						let mut cu_decls = self.cu.decls.lock();
						cu_decls.push(Decl {
							name: COMMON_INTERNS.self_ty_symbol,
							module: self.module,
							namespace,
							analysis_state: DeclAnalysisState::Analysed { value: union_idx },
						})
					};
					self.cu.namespaces.with_mut(|namespaces| {
						namespaces[namespace].decls.insert(COMMON_INTERNS.self_ty_symbol, self_decl_id);
					});
				}

				self.vuir_map.insert(id, vtir::InstructionRef::Interned(union_idx));

				self.unstack_block(block);

				Ok((vtir::InstructionRef::Interned(union_idx), ControlFlow::May))
			},
			vuir::Opcode::DeclFn {
				ret_ty,
				ret_ty_is_generic,
				params,
				var_args,
				body: _,
				external,
				callconv,
				builtin,
				inline,
				first_positional_arg_index,
				span,
			} => {
				// Collect parameters with both names and types for named argument resolution
				let (mut param_defs, mut comptime_params, runtime_param_error) = {
					let params_block = self.child_block(block);
					let _ = self.analyze_comptime_block(*params_block, params);
					let mut comptime_params = BitVec::new();
					let mut runtime_param_error = false;
					let decl_fn_params = self.blocks[*params_block].decl_fn_params.clone();
					let mut param_defs = Vec::with_capacity(decl_fn_params.len());
					for param in decl_fn_params {
						comptime_params.push(param.comptime);
						if !param.comptime
							&& !self.cu.values.type_contains_generic_poison(param.ty)
							&& self.cu.values.type_is_comptime_only(param.ty)
						{
							self.push_error(
								Diagnostic::error()
									.with_message(format!(
										"runtime parameter `{}` cannot have comptime-only type `{}`",
										param.name,
										self.cu.values.display_index(param.ty)
									))
									.with_label(Label::primary().with_span(self.diag_span(param.span)))
									.with_note("use `comptime`"),
							);
							runtime_param_error = true;
						}
						param_defs.push(param.ty);
					}
					(param_defs, comptime_params, runtime_param_error)
				};
				if runtime_param_error {
					return Err(AnalyzeError::AnalysisFailed);
				}

				// Function is generic if it has generic params or if return type is TypeAny
				let fn_ty = {
					// generics are analyzed in callsite
					let ret_ty = if *ret_ty_is_generic {
						self.cu.values.common.generic_poison_t
					} else {
						let vuir::Opcode::BlockComptime { instructions, .. } = &self.vuir.instructions[ret_ty] else {
							unreachable!();
						};
						let ret_ty = self.analyze_comptime_block(block, instructions)?;
						let ret_ty = ret_ty.unwrap();
						ret_ty.as_interned()
					};

					let callconv = if let Some(callconv) = callconv {
						let vuir::Opcode::BlockComptime { instructions, .. } = &self.vuir.instructions[*callconv] else {
							unreachable!();
						};

						let callconv_value = self.analyze_comptime_block(block, instructions)?;

						let Some(callconv_value) = callconv_value else {
							self.push_error(
								Diagnostic::error()
									.with_message("`#callconv` requires a comptime value")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						};

						let callconv_ty = self.resolve_builtin_calling_convention_type()?;
						let callconv_value = callconv_value.as_interned();
						if self.cu.values.type_of_interned(callconv_value) != callconv_ty {
							self.push_error(
								Diagnostic::error()
									.with_message("`#callconv` expression must have type `CallingConvention`")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}

						let value::Value::Enum(callconv_enum_ty) = self.cu.values.index_to_value(callconv_ty) else {
							unreachable!();
						};

						let value::Key::EnumTag { enum_ty, val } = self.cu.values.index_to_key(callconv_value) else {
							self.push_error(
								Diagnostic::error()
									.with_message("`#callconv` expression must evaluate to a `CallingConvention` variant")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						};

						if *enum_ty != callconv_ty {
							self.push_error(
								Diagnostic::error()
									.with_message("`#callconv` expression must evaluate to `builtin.CallingConvention`")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}

						let val = self.cu.values.index_to_key(*val).as_int().1;
						let val = val.to_u64().unwrap() as u8;
						assert!(callconv_enum_ty.fields.len() == CallingConvention::Count as _);
						assert!(val < CallingConvention::Count as u8);
						// SAFETY: `CallingConvention` is `repr(u8)`, and every value below
						// `Count` names one of its calling-convention variants.
						Some(unsafe { core::mem::transmute::<u8, CallingConvention>(val) })
					} else {
						None
					};

					// `extern "..."` is an extern namespace/library hint, not a calling
					// convention. If no explicit `#callconv(...)` is provided, extern fns
					// default to C calling convention.
					let callconv = if let Some(callconv) = callconv {
						callconv
					} else if *external {
						CallingConvention::C
					} else {
						CallingConvention::Vif
					};

					let fn_ty = value::Key::Type(value::Type::Fn(TypeFn {
						params: self.cu.values.alloc_slice(&param_defs),
						comptime_params: self.cu.values.alloc_bitslice(&comptime_params),
						first_positional_param: *first_positional_arg_index,
						var_args: *var_args,
						ret_ty,
						external: *external,
						callconv,
						inline: *inline,
					}));

					self.cu.values.intern_trivial(&fn_ty)
				};

				let fn_decl = self.cu.values.intern_trivial(&value::Key::FnDecl(FnDecl {
					func_decl_inst: GlobalVuirInstructionId {
						module: self.module,
						inst: id,
					},
					owner_decl: self.owner_decl,
					ty: fn_ty,
				}));

				let fn_decl = InstructionRef::Interned(fn_decl);

				self.vuir_map.insert(id, fn_decl);
				Ok((fn_decl, ControlFlow::May))
			},

			vuir::Opcode::Break {
				block: br_block,
				value,
				value_span,
			} => {
				let value = self.resolve_inst(value);
				if self.blocks[block].comptime {
					return Err(AnalyzeError::ComptimeBreak { block: *br_block, value });
				}

				let br_block = self.resolve_inst(&br_block.as_ref()).as_id().unwrap();

				self.ensure_type_exist_in_runtime(self.type_of(&value), value_span)?;

				let inst = self.inst(block, vtir::Opcode::Break { block: br_block, value });
				self.vuir_map.insert(id, inst);

				// register our break to the block
				let mut block_id = block;
				loop {
					let block = &mut self.blocks[block_id];
					if let Some(vuir_block) = &mut block.vuir_block
						&& vuir_block.block_inst == br_block.as_ref()
					{
						vuir_block.breaks.push(value);
						break;
					}
					block_id = block
						.parent
						.expect("reached bottom of stack without finding the block referenced by the break");
				}

				Ok((inst, ControlFlow::Always))
			},
			vuir::Opcode::Repeat { r#loop } => {
				let loop_block = self.resolve_inst(&r#loop.as_ref()).as_id().unwrap();
				let inst = self.inst(block, vtir::Opcode::Repeat { r#loop: loop_block });
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::Always))
			},
			vuir::Opcode::BreakComptime { block, value } => {
				let value = self.resolve_inst(value);
				Err(AnalyzeError::ComptimeBreak { block: *block, value })
			},
			vuir::Opcode::FieldValFromPtr { lhs, field, span } => {
				let lhs = self.resolve_inst(lhs);
				let inst = self.analyze_field_ptr(block, lhs, field, span, span)?;
				let inst = self.analyze_load(block, inst, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::FieldPtrFromPtr { lhs, field, span } => {
				let inst = self.analyze_field_ptr(block, self.resolve_inst(lhs), field, span, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::FieldValFromVal { lhs, field, span } => {
				let inst = self.analyze_field_val(block, self.resolve_inst(lhs), field, span, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::ArrayIndexElemVal { array_ptr, index, span } => {
				let inst = self.analyze_array_index_ptr(block, *array_ptr, *index, self.diag_span(*span))?;
				let inst = self.analyze_load(block, inst, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::ArrayIndexElemPtr { array_ptr, index, span } => {
				let inst = self.analyze_array_index_ptr(block, *array_ptr, *index, self.diag_span(*span))?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::CaptureGet { idx, span } => {
				let owner_type = self
					.cu
					.namespaces
					.with(|namespaces| namespaces[self.blocks[block].namespace].owner_type);
				let capture = self
					.namespace_captures(owner_type)
					.expect("CaptureGet requires a namespace owner with captures")[*idx];
				let capture = match capture {
					value::Capture::Comptime(capture) => capture.into(),
					value::Capture::Runtime(capture) => {
						let Some(runtime_env) = self.blocks[block].capture_context.runtime_env.clone() else {
							self.push_error(
								Diagnostic::error()
									.with_message("runtime capture used without a runtime capture environment")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						};
						let Some(field_idx) = runtime_env.fields.get(&capture).copied() else {
							self.push_error(
								Diagnostic::error()
									.with_message("runtime capture is not available in the current namespace environment")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						};
						let env_ty = runtime_env.ty;
						let field_ty = match self.cu.values.index_to_key_value(env_ty) {
							(value::Key::Type(value::Type::Struct(_)), value::Value::Struct(env_struct)) => {
								env_struct.as_ref().fields[field_idx].ty
							},
							_ => {
								self.push_error(Diagnostic::error().with_message("runtime capture environment must be a struct"));
								return Err(AnalyzeError::AnalysisFailed);
							},
						};
						let env_ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
							pointee_ty: env_ty,
							packed: None,
							is_const: false,
						})));
						let typed_env_ptr = self.inst(block, vtir::Opcode::BitCast {
							src: runtime_env.ptr,
							dst_ty: env_ptr_ty,
						});
						let capture_field_ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
							pointee_ty: field_ty,
							packed: None,
							is_const: false,
						})));
						let capture_field_ptr = self.inst(block, vtir::Opcode::StructFieldPtr {
							struct_ptr: typed_env_ptr,
							field_idx,
							ret_ty: capture_field_ptr_ty,
						});
						let captured_ptr = self.analyze_load(block, capture_field_ptr, span)?;
						self.analyze_load(block, captured_ptr, span)?
					},
				};
				self.vuir_map.insert(id, capture);
				Ok((capture, ControlFlow::May))
			},
			vuir::Opcode::Undefined { ty, span } => {
				let ty = ty.map_or(Ok(self.cu.values.common.any_t), |ty| self.resolve_type(block, &ty, span))?;
				let inst = self.inst(block, vtir::Opcode::Undefined { ty });
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::AggregateInit { ty, kind, span } => {
				let ty = self.resolve_type(block, ty, span)?;
				let value::Key::Type(ty_key) = self.cu.values.index_to_key(ty) else {
					unreachable!("aggregate initializer target is not a type")
				};
				match kind {
					vuir::AggregateInitKind::Empty => match ty_key {
						value::Type::Union(..) => self.analyze_union_init(id, block, ty, &[], span),
						value::Type::Struct(..) => self.analyze_struct_init(id, block, ty, &[], span),
						value::Type::Array(..) | value::Type::Slice(..) => self.analyze_array_init(id, block, ty, &[], span),
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
						| value::Type::NullPtr
						| value::Type::Any
						| value::Type::Anyptr
						| value::Type::GenericPoison
						| value::Type::Type
						| value::Type::Never
						| value::Type::EnumLiteral => unreachable!("invalid empty aggregate initializer type"),
					},
					vuir::AggregateInitKind::Adt(fields) => match ty_key {
						value::Type::Union(..) => self.analyze_union_init(id, block, ty, fields, span),
						value::Type::Struct(..) => self.analyze_struct_init(id, block, ty, fields, span),
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
						| value::Type::Array(_)
						| value::Type::NullPtr
						| value::Type::Any
						| value::Type::Anyptr
						| value::Type::GenericPoison
						| value::Type::Type
						| value::Type::Never
						| value::Type::EnumLiteral => unreachable!("invalid ADT initializer type"),
					},
					vuir::AggregateInitKind::Array(elements) => self.analyze_array_init(id, block, ty, elements, span),
				}
			},
			vuir::Opcode::DeclFnParam {
				name,
				type_body,
				comptime,
				generic,
				span,
			} => {
				// generics are analyzed in callsite
				let param_ty = if *generic {
					self.cu.values.common.generic_poison_t
				} else if let Some(fn_ty) = self.fun.map(|fun| {
					let k = self.cu.values.index_to_key(fun).as_fn();
					self.cu.values.index_to_key(k.ty).as_type_fn()
				}) {
					let param_idx = self.blocks[block].decl_fn_params.len();
					fn_ty.params[param_idx]
				} else {
					let ty = self.analyze_comptime_block(block, type_body)?.unwrap();
					ty.as_interned()
				};
				self.vuir_map.insert(id, vtir::InstructionRef::Interned(param_ty));

				let param_name = name.symbol;

				self.blocks[block].decl_fn_params.push(DeclFnParam {
					vuir_id: id,
					name: param_name,
					ty: param_ty,
					comptime: *comptime,
					span: *span,
				});
				Ok((vtir::InstructionRef::Interned(param_ty), ControlFlow::May))
			},
			opcode @ (vuir::Opcode::StackAlloc { name, ty, span } | vuir::Opcode::StackAllocMut { name, ty, span }) => {
				let ty = self.resolve_type(block, ty, span)?;
				let is_linear = self.cu.values.type_is_linear(ty);
				let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
					pointee_ty: ty,
					packed: None,
					is_const: false,
				})));
				let inst_ref = {
					if self.cu.values.type_is_comptime_only(ty) {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"a const or mut binding cannot have a comptime-only type (`{}`)",
									self.cu.values.display_index(ty)
								))
								.with_label(Label::primary().with_span(self.diag_span(*span)))
								.with_note("use a comptime binding: `comptime const ...` or `comptime var ...`"),
						);
						return Err(AnalyzeError::AnalysisFailed);
					}
					self.inst(block, vtir::Opcode::StackAlloc { ty: ptr_ty })
				};

				if is_linear {
					self.linear_slots.insert(inst_ref, LinearSlot {
						ty,
						span: name.span,
						consumed: false,
						consumed_at: None,
						name: name.symbol,
					});
				}
				self.vuir_map.insert(id, inst_ref);
				Ok((inst_ref, ControlFlow::May))
			},
			opcode @ (vuir::Opcode::StackAllocComptime { name, ty, span } | vuir::Opcode::StackAllocComptimeMut { name, ty, span }) => {
				let ty = self.resolve_type(block, ty, span)?;
				let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
					pointee_ty: ty,
					packed: None,
					is_const: false,
				})));
				let inst_ref = {
					let const_alloc = self.comptime_memory.allocate(ty, *span);
					vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
						ty: ptr_ty,
						kind: PtrKind::ComptimeAlloc(const_alloc),
					})))
				};
				self.vuir_map.insert(id, inst_ref);
				Ok((inst_ref, ControlFlow::May))
			},
			opcode @ (vuir::Opcode::StackAllocInferred { name, .. } | vuir::Opcode::StackAllocInferredMut { name, .. }) => {
				let is_const = matches!(opcode, vuir::Opcode::StackAllocInferred { .. });
				let inst_ref = self.inst(block, vtir::Opcode::StackAllocInferred { is_comptime: false });
				self.pending_inferred_alloc_to_ty.insert(inst_ref, None);
				self.vuir_map.insert(id, inst_ref);
				if is_const {
					self.allocs
						.potential_comptime_allocs
						.insert(inst_ref, PotentialComptimeAlloc::default());
				}
				Ok((inst_ref, ControlFlow::May))
			},
			opcode @ (vuir::Opcode::StackAllocInferredComptime { name, span }
			| vuir::Opcode::StackAllocInferredComptimeMut { name, span }) => {
				let is_const = matches!(opcode, vuir::Opcode::StackAllocInferredComptime { .. });
				let inst_ref = self.inst(block, vtir::Opcode::StackAllocInferred { is_comptime: true });
				self.allocs
					.potential_comptime_allocs
					.insert(inst_ref, PotentialComptimeAlloc::default());
				self.pending_inferred_alloc_to_ty.insert(inst_ref, None);
				self.vuir_map.insert(id, inst_ref);

				Ok((inst_ref, ControlFlow::May))
			},
			vuir::Opcode::ReifyInferredAlloc { alloc, span } => {
				// Get the name and mutability from the vuir opcode before resolving
				let (alloc_name, is_const_binding) = match &self.vuir.instructions[alloc.as_id().unwrap()] {
					vuir::Opcode::StackAllocInferred { name, .. } | vuir::Opcode::StackAllocInferredComptime { name, .. } => (*name, true),
					vuir::Opcode::StackAllocInferredMut { name, .. } | vuir::Opcode::StackAllocInferredComptimeMut { name, .. } => {
						(*name, false)
					},
					_ => unreachable!(),
				};
				let alloc = self.resolve_inst(alloc);
				let vtir::Opcode::StackAllocInferred { is_comptime } = self.instructions[alloc.as_id().unwrap()] else {
					unreachable!()
				};
				let ty = self
					.pending_inferred_alloc_to_ty
					.remove(&alloc)
					.expect("inferred alloc was not in the pending infer list")
					.ok_or_else(|| {
						self.push_error(
							Diagnostic::error()
								.with_message("cannot infer type of untyped allocation")
								.with_label(Label::primary().with_span(self.diag_span(*span))),
						);
						AnalyzeError::AnalysisFailed
					})?;

				let alloc = if let Some(potential_comptime_alloc) = self.allocs.potential_comptime_allocs.remove(&alloc)
					&& {
						assert!(
							potential_comptime_alloc.stores.len() == 1,
							"TODO(zino): multiple store to inferred alloc comptime is not supported yet, seen {:?}",
							potential_comptime_alloc
						);
						let (store_inst, _) = potential_comptime_alloc.stores[0];
						let vtir::Opcode::Store { src: value, .. } = self.instructions[store_inst.as_id().unwrap()] else {
							unreachable!()
						};

						self.try_resolve_comptime_value(&value).is_some()
					} {
					let (store_inst, store_span) = potential_comptime_alloc.stores[0];
					let vtir::Opcode::Store { src: value, .. } = self.instructions[store_inst.as_id().unwrap()] else {
						unreachable!()
					};

					self.instructions[store_inst.as_id().unwrap()] = vtir::Opcode::Noop;
					self.instructions[alloc.as_id().unwrap()] = vtir::Opcode::Noop;

					{
						let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
							pointee_ty: ty,
							packed: None,
							is_const: true,
						})));
						let ptr = self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
							ty: ptr_ty,
							kind: PtrKind::Value(value.as_interned()),
						}));
						vtir::InstructionRef::Interned(ptr)
					}
				} else {
					if is_comptime {
						self.push_error(
							Diagnostic::error()
								.with_message("comptime initializer could not be resolved at compile time")
								.with_label(Label::primary().with_span(self.diag_span(*span))),
						);
						return Err(AnalyzeError::AnalysisFailed);
					}
					self.ensure_type_exist_in_runtime(ty, span)?;
					let ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
						pointee_ty: ty,
						packed: None,
						is_const: is_const_binding,
					})));
					self.instructions[alloc.as_id().unwrap()] = vtir::Opcode::StackAlloc { ty };
					alloc
				};

				if self.cu.values.type_is_linear(ty) {
					self.linear_slots.insert(alloc, LinearSlot {
						ty,
						span: alloc_name.span,
						consumed: false,
						consumed_at: None,
						name: alloc_name.symbol,
					});
				}
				self.vuir_map.insert(id, alloc);
				Ok((alloc, ControlFlow::May))
			},
			vuir::Opcode::FreezeStackAlloc { alloc, span } => {
				let original_alloc = self.resolve_inst(&alloc.as_ref());
				let alloc_ptr_const_ty = {
					let alloc_ptr_ty = self.cu.values.index_to_key(self.type_of(&original_alloc)).as_type_ptr();
					self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
						pointee_ty: alloc_ptr_ty.pointee_ty,
						packed: alloc_ptr_ty.packed,
						is_const: alloc_ptr_ty.is_const,
					})))
				};

				let alloc = if self.try_resolve_comptime_value(&original_alloc).is_some() {
					let ptr = self.cu.values.index_to_key(original_alloc.as_interned());
					self.coerce(block, alloc_ptr_const_ty, original_alloc, span)?
				} else {
					self.inst(block, vtir::Opcode::BitCast {
						src: original_alloc,
						dst_ty: alloc_ptr_const_ty,
					})
				};

				// Transfer linear slot from original alloc to the frozen alloc
				if let Some(slot) = self.linear_slots.remove(&original_alloc) {
					self.linear_slots.insert(alloc, slot);
				}

				self.vuir_map.insert(id, alloc);
				Ok((alloc, ControlFlow::May))
			},
			vuir::Opcode::DeclVal(ident) => {
				let decl = self
					.lookup_decl_in_namespace_recursively(self.blocks[block].namespace, ident.symbol)
					.ok_or_else(|| {
						let module_value = self.cu.modules.with(|modules| match *modules[self.module].sema_state.lock() {
							ModuleAnalyzeState::Done(v) => v,
							_ => self.cu.values.common.unreachable_value,
						});
						self.diag_decl_not_found(&ident.symbol, module_value, &ident.span);
						AnalyzeError::AnalysisFailed
					})?;
				let inst = self.analyze_decl_val(block, decl, &ident.span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::DeclRef(ident) => {
				let decl = self
					.lookup_decl_in_namespace_recursively(self.blocks[block].namespace, ident.symbol)
					.ok_or_else(|| {
						let module_value = self.cu.modules.with(|modules| match *modules[self.module].sema_state.lock() {
							ModuleAnalyzeState::Done(v) => v,
							_ => self.cu.values.common.unreachable_value,
						});
						self.diag_decl_not_found(&ident.symbol, module_value, &ident.span);
						AnalyzeError::AnalysisFailed
					})?;
				let inst = self.analyze_decl_ptr(block, decl)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::Load { src, span } => {
				let src = self.resolve_inst(src);
				let inst = self.analyze_load(block, src, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::Store { dst, src, span } => {
				let src_ref = self.resolve_inst(src);
				let dst_ref = self.resolve_inst(dst);
				// mutability check
				{
					let dst_ptr_ty = self.cu.values.index_to_key(self.type_of(&dst_ref)).as_type_ptr();

					if dst_ptr_ty.is_const {
						self.push_error(
							Diagnostic::error()
								.with_message("cannot assign to a constant")
								.with_label(Label::primary().with_message("assignement here").with_span(self.diag_span(*span))),
						);
					}
				}

				// Type mismatch
				let dst_ptr_ty = self.type_of(&dst_ref);
				let dst_pointee_ty = self.cu.values.index_to_key(dst_ptr_ty).as_type_ptr().pointee_ty;

				// is the destination ptr a comptime alloc ?
				let value = if dst_ref.is_interned() {
					let ptr = self.cu.values.index_to_key(dst_ref.as_interned()).as_ptr();
					let PtrKind::ComptimeAlloc(comptime_alloc_id) = ptr.kind else {
						unreachable!("tried to perform a comptime store to a non-ComptimeAlloc ptr")
					};

					// ensure the src is comptime known or fail
					if self.try_resolve_comptime_value(&src_ref).is_none() {
						self.push_error(
							Diagnostic::error()
								.with_message("cannot store a runtime value inside a comptime allocation")
								.with_label(Label::primary().with_message("runtime value").with_span(self.diag_span(*span)))
								.with_label(
									Label::secondary()
										.with_message("comptime alloc here")
										.with_span(self.diag_span(self.comptime_memory.allocs[comptime_alloc_id].span)),
								),
						);
						return Err(AnalyzeError::AnalysisFailed);
					}

					let coerced_src = self.coerce(block, dst_pointee_ty, src_ref, span)?;
					self.comptime_memory.allocs[comptime_alloc_id].value = ComptimeAllocValue::Interned(coerced_src.as_interned());

					vtir::InstructionRef::Interned(self.cu.values.common.void_t)
				} else {
					let coerced_src = self.coerce(block, dst_pointee_ty, src_ref, span)?;
					self.inst(block, vtir::Opcode::Store {
						dst: dst_ref,
						src: coerced_src,
					})
				};
				self.vuir_map.insert(id, value);
				Ok((value, ControlFlow::May))
			},
			vuir::Opcode::StoreToInferredAlloc { dst, src, span } => {
				let src = self.resolve_inst(src);
				let src_ty = self.type_of(&src);
				let dst = self.resolve_inst(dst);
				assert!(
					matches!(self.instructions[dst.as_id().unwrap()], vtir::Opcode::StackAllocInferred { .. }),
					"StoreToInferredAlloc destination must be an inferred stack allocation"
				);

				if src_ty == self.cu.values.common.void_t {
					self.push_error(
						Diagnostic::error()
							.with_message("`void` inferred allocations are not yet supported (no comptime)")
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
				}

				let inst = self.inst(block, vtir::Opcode::Store { dst, src });
				self.pending_inferred_alloc_to_ty.insert(dst, Some(src_ty));
				self.link_store_to_potential_comptime_alloc(&dst, &inst, span);

				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::FnCall {
				fun,
				generic_args,
				args,
				ret_ty: expected_ret_ty,
				span,
			} => {
				let fun = self.resolve_inst(fun);
				self.analyze_fn_call(
					id,
					block,
					AnalyzedCallee { fun, env: None },
					generic_args,
					args,
					expected_ret_ty,
					None,
					span,
				)
				.map(|inst| (inst, ControlFlow::May))
			},
			vuir::Opcode::FnCallWithFieldPtrReceiver {
				field_ptr,
				field_name,
				generic_args,
				args,
				ret_ty: expected_ret_ty,
				span,
			} => {
				let field_ptr = self.resolve_inst(field_ptr);
				let field_ptr_ty = self.type_of(&field_ptr);

				let field_ptr_inner_ty = match self.cu.values.index_to_key(field_ptr_ty) {
					value::Key::Type(value::Type::Ptr(ptr)) => ptr.pointee_ty,
					_ => unreachable!(),
				};

				let (callee, receiver) = 'fun: {
					// since we a fn call takes a pointer, we may get a pointer to pointer:
					// 		const S = ...;
					// 		S.call()
					// 		^ this part is a **type, fn calls in vuir generation always takes their callee by ref
					//
					// we need to deref the field_ptr at least once if it contains an other pointer to get the inner object
					let (object_ptr, concrete_ty) =
						if let value::Key::Type(value::Type::Ptr(inner_ptr)) = self.cu.values.index_to_key(field_ptr_inner_ty) {
							let loaded = self.analyze_load(block, field_ptr, span)?;
							(loaded, inner_ptr.pointee_ty)
						} else {
							(field_ptr, field_ptr_inner_ty)
						};

					// first resolve by field: function pointer call, type as callee
					match self.cu.values.index_to_key_value(concrete_ty) {
						(value::Key::Type(value::Type::Struct(_)), value::Value::Struct(r#struct)) => {
							// TODO(zino)
							// fallback to decl resolution for now
						},
						(value::Key::Type(value::Type::Type), _) => {
							let namespace_type = self.analyze_load(block, object_ptr, span)?;
							let fun = self.analyze_field_val(block, namespace_type, &field_name.symbol, &field_name.span, span)?;
							break 'fun (AnalyzedCallee { fun, env: None }, None);
						},
						_ => todo!(),
					}

					// field resolution failed, search in decls:  method call
					let decl = match self.cu.values.index_to_key_value(concrete_ty) {
						(value::Key::Type(value::Type::Struct(_)), value::Value::Struct(r#struct)) => {
							// we have a pointer to a struct, lookup in the struct the field as a decl
							self.lookup_decl_in_namespace(r#struct.namespace, field_name.symbol)
						},
						_ => unreachable!(),
					}
					.ok_or_else(|| {
						self.push_error(
							Diagnostic::error()
								.with_message(format!(
									"no field or member function named `{}` in `{}`",
									field_name.symbol,
									self.cu.values.display_index(concrete_ty)
								))
								.with_label(Label::primary().with_span(self.diag_span(field_name.span))),
						);
						AnalyzeError::AnalysisFailed
					})?;

					// we have a potential function
					let decl = self.analyze_decl_val(block, decl, span)?;
					match self.cu.values.index_to_key(self.type_of(&decl)) {
						value::Key::Type(value::Type::Fn(TypeFn {
							params,
							first_positional_param: Some(first_positional_param),
							..
						})) => {
							let expected_ty = params[*first_positional_param as usize];
							if concrete_ty == expected_ty {
								let deref_callee = self.analyze_load(block, object_ptr, span)?;
								(AnalyzedCallee { fun: decl, env: None }, Some(deref_callee))
							} else if matches!(self.cu.values.index_to_key(expected_ty), value::Key::Type(value::Type::Ptr(ptr_ty)) if ptr_ty.pointee_ty == concrete_ty)
							{
								// we authorize a single dereference for receiver calls so we don't need a special syntax -> like C++
								(AnalyzedCallee { fun: decl, env: None }, Some(object_ptr))
							} else {
								self.push_error(
									Diagnostic::error()
										.with_message(format!(
											"receiver types doesn't match: expected `{}` found `{}`",
											self.cu.values.display_index(expected_ty),
											self.cu.values.display_index(concrete_ty),
										))
										.with_label(Label::primary().with_span(self.diag_span(field_name.span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							}
						},
						_ => {
							self.push_error(
								Diagnostic::error()
									.with_message(format!("`{}` is not a member function", field_name.symbol,))
									.with_label(Label::primary().with_span(self.diag_span(field_name.span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						},
					}
				};

				self.analyze_fn_call(id, block, callee, generic_args, args, expected_ret_ty, receiver, span)
					.map(|inst| (inst, ControlFlow::May))
			},
			vuir::Opcode::Coerce { value, into, span } => {
				let dst_ty = self.resolve_type(block, into, span)?;
				let value = self.resolve_inst(value);
				let value = self.coerce(block, dst_ty, value, span)?;
				self.vuir_map.insert(id, value);
				Ok((value, ControlFlow::May))
			},
			vuir::Opcode::TypeOfCurFnRet => {
				let fun_idx = self.fun.expect("TypeOfCurFn analyzed outside of a function");
				let ret_ty = {
					let fun_key = self.cu.values.index_to_key(fun_idx).as_fn();
					let fn_ty = self.cu.values.index_to_key(fun_key.ty).as_type_fn();
					fn_ty.ret_ty
				};
				let ret_ty = InstructionRef::Interned(ret_ty);
				self.vuir_map.insert(id, ret_ty);
				Ok((ret_ty, ControlFlow::May))
			},
			vuir::Opcode::TypeBuiltinCallingConvention => {
				let callconv_ty = self.resolve_builtin_calling_convention_type()?;
				let callconv_ty = InstructionRef::Interned(callconv_ty);
				self.vuir_map.insert(id, callconv_ty);
				Ok((callconv_ty, ControlFlow::May))
			},
			vuir::Opcode::StructInitTypeOfField { r#struct, field } => {
				let inst = self.resolve_inst(r#struct).as_interned();
				let ty = match self.cu.values.index_to_key(inst) {
					value::Key::Type(value::Type::Union(_)) => {
						let u = self.cu.values.index_to_value(inst).as_union();
						let u = u.as_ref();
						if let Some(idx) = u.field_idx_by_name(field) {
							if let Some(field_ty) = u.fields[idx as usize].ty {
								vtir::InstructionRef::Interned(field_ty)
							} else {
								vtir::InstructionRef::Interned(self.cu.values.common.void_t)
							}
						} else {
							vtir::InstructionRef::Interned(self.cu.values.common.generic_poison_t)
						}
					},
					_ => {
						let r#struct = self.cu.values.index_to_value(inst).as_struct();
						let r#struct = r#struct.as_ref();
						if let Some(field) = r#struct.field_idx_by_name(field) {
							let field = &r#struct.fields[field];
							vtir::InstructionRef::Interned(field.ty)
						} else {
							// no diag, expected to be emitted by the StructInit opcode
							vtir::InstructionRef::Interned(self.cu.values.common.generic_poison_t)
						}
					},
				};
				self.vuir_map.insert(id, ty);
				Ok((ty, ControlFlow::May))
			},
			vuir::Opcode::TypeOfPtrPointee { ptr } => {
				let ptr = self.resolve_inst(ptr);
				let ty = if let vtir::InstructionRef::Instruction(ptr_id) = ptr
					&& matches!(self.instructions[ptr_id], vtir::Opcode::StackAllocInferred { .. })
				{
					self.pending_inferred_alloc_to_ty
						.get(&ptr)
						.copied()
						.flatten()
						.unwrap_or(self.cu.values.common.any_t)
				} else {
					self.cu.values.index_to_key(self.type_of(&ptr)).as_type_ptr().pointee_ty
				};
				self.vuir_map.insert(id, InstructionRef::Interned(ty));
				Ok((InstructionRef::Interned(ty), ControlFlow::May))
			},
			vuir::Opcode::TypeOf { value } => {
				let value = self.resolve_inst(value);
				let ty = self.type_of(&value);
				self.vuir_map.insert(id, InstructionRef::Interned(ty));
				Ok((InstructionRef::Interned(ty), ControlFlow::May))
			},
			vuir::Opcode::Return { value, span } => {
				let value = value
					.map(|value| {
						let value = self.resolve_inst(&value);
						let cur_fn = self.cu.values.index_to_key(self.fun.unwrap()).as_fn();
						let cur_fn_ty = self.cu.values.index_to_key(cur_fn.ty).as_type_fn();
						let value = self.coerce(block, cur_fn_ty.ret_ty, value, span)?;
						Ok(value)
					})
					.transpose()?;
				if self.blocks[block].inlined {
					Err(AnalyzeError::InlineReturn { value })
				} else {
					let inst = self.inst(block, vtir::Opcode::Return { value });
					self.vuir_map.insert(id, inst);
					Ok((inst, ControlFlow::Always))
				}
			},
			vuir::Opcode::DbgSrcLoc { line, col } => {
				let inst = self.inst(block, vtir::Opcode::DbgSrcLoc { line: *line, col: *col });
				Ok((inst, ControlFlow::May))
			},

			// blocks
			kind @ (vuir::Opcode::Block { instructions, span } | vuir::Opcode::Loop { instructions, span }) => {
				// need to lower block inst now before analyzing instructions
				// since they may reference the block (e.g breaks)
				let inst = self.inst_id(block, vtir::Opcode::Invalid);
				self.vuir_map.insert(id, inst.as_ref());

				let unstacked_block = {
					let block = self.child_block_from_vuir_block(block, VuirBlockAnalysisData {
						block_inst: inst.as_ref(),
						breaks: BumpVec::new_in(self.instructions_payload_alloc),
					});
					self.analyze_instruction_block(*block, instructions)?;
					self.unstack_block(block)
				};

				let vuir_block = unstacked_block.vuir_block.unwrap();
				let ret_ty = if vuir_block.breaks.is_empty() {
					self.cu.values.common.void_t
				} else {
					let ret_ty = if vuir_block.breaks[0]
						.as_interned_opt()
						.map(|index| index == self.cu.values.common.void_t)
						.unwrap_or(false)
					{
						self.cu.values.common.void_t
					} else {
						self.type_of(&vuir_block.breaks[0])
					};

					for break_val in &vuir_block.breaks[1..] {
						self.coerce(block, ret_ty, *break_val, span)?;
					}
					ret_ty
				};

				self.instructions[inst] = match kind {
					vuir::Opcode::Block { .. } => vtir::Opcode::Block {
						instructions: unstacked_block.instructions.into_bump_slice(),
						ret_ty,
					},
					vuir::Opcode::Loop { .. } => vtir::Opcode::Loop {
						instructions: unstacked_block.instructions.into_bump_slice(),
						ret_ty,
					},
					_ => unreachable!(),
				};
				Ok((inst.as_ref(), ControlFlow::May))
			},

			// unary
			vuir::Opcode::BoolNot { op, span } => {
				let op = self.resolve_inst(op);
				let op = self.coerce(block, self.cu.values.common.bool_t, op, span)?;

				let inst = if let Some(op) = self.try_resolve_comptime_value(&op) {
					let value = if op == self.cu.values.common.true_value {
						self.cu.values.common.false_value
					} else {
						self.cu.values.common.true_value
					};
					value.into()
				} else {
					self.inst(block, vtir::Opcode::BoolNot { op })
				};

				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::Negate { op, span } => {
				todo!();
			},

			// arithmetic
			opcode @ (vuir::Opcode::Add { lhs, rhs, span }
			| vuir::Opcode::AddSat { lhs, rhs, span }
			| vuir::Opcode::Sub { lhs, rhs, span }
			| vuir::Opcode::SubSat { lhs, rhs, span }
			| vuir::Opcode::Mul { lhs, rhs, span }
			| vuir::Opcode::MulSat { lhs, rhs, span }
			| vuir::Opcode::Div { lhs, rhs, span }
			| vuir::Opcode::Rem { lhs, rhs, span }) => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let inst = self.analyze_arithmetic_op(block, opcode, lhs, rhs, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},

			// bitwise
			opcode @ (vuir::Opcode::Shl { lhs, rhs, span }
			| vuir::Opcode::ShlSat { lhs, rhs, span }
			| vuir::Opcode::ShlWrap { lhs, rhs, span }
			| vuir::Opcode::Shr { lhs, rhs, span }
			| vuir::Opcode::ShrSat { lhs, rhs, span }
			| vuir::Opcode::ShrWrap { lhs, rhs, span }
			| vuir::Opcode::BitAnd { lhs, rhs, span }
			| vuir::Opcode::BitOr { lhs, rhs, span }
			| vuir::Opcode::BitXor { lhs, rhs, span }) => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let inst = self.analyze_bitwise_op(block, opcode, lhs, rhs, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::BitNot { op, span } => {
				let op = self.resolve_inst(op);
				let inst = self.analyze_bitwise_not_op(block, op, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},

			// comparaison
			opcode @ (vuir::Opcode::Lt { lhs, rhs, span }
			| vuir::Opcode::Lte { lhs, rhs, span }
			| vuir::Opcode::Gt { lhs, rhs, span }
			| vuir::Opcode::Gte { lhs, rhs, span }
			| vuir::Opcode::BoolAnd { lhs, rhs, span }
			| vuir::Opcode::BoolOr { lhs, rhs, span }
			| vuir::Opcode::Eq { lhs, rhs, span }
			| vuir::Opcode::Neq { lhs, rhs, span }) => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let inst = self.analyze_comparaison_op(block, opcode, lhs, rhs, span)?;
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},

			vuir::Opcode::Invalid => unreachable!(),

			vuir::Opcode::Declaration(decl) => {
				let block = self.child_block(block);
				self.blocks[*block].comptime = true;

				let value = self
					.analyze_comptime_block(*block, decl.value)?
					.expect("a declaration block must always have a value");
				self.vuir_map.insert(id, value);
				self.unstack_block(block);
				Ok((value, ControlFlow::May))
			},

			vuir::Opcode::BlockComptime { instructions } => {
				let block = self.child_block(block);
				self.blocks[*block].comptime = true;
				let value = self
					.analyze_comptime_block(*block, instructions)?
					.expect("a comptime block must always return a value");
				self.vuir_map.insert(id, value);
				self.unstack_block(block);
				Ok((value, ControlFlow::May))
			},
			// TODO better handling of pointers etc.
			vuir::Opcode::TypePtr {
				pointee, is_const, span, ..
			} => {
				let pointee_ty = self.resolve_inst(pointee);
				let Some(pointee_ty) = self.try_resolve_comptime_value(&pointee_ty) else {
					self.push_error(
						Diagnostic::error()
							.with_message("pointer type must be a valid type but is an arbitrary value")
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				};
				let ty = value::Key::Type(value::Type::Ptr(TypePtr {
					pointee_ty,
					packed: None,
					is_const: *is_const,
				}));
				let ty = vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&ty));
				self.vuir_map.insert(id, ty);
				Ok((ty, ControlFlow::May))
			},
			vuir::Opcode::TypeSlice { elem: pointee_ty, .. } => {
				let pointee_ty = &self.resolve_inst(pointee_ty);
				let pointee_ty = self.try_resolve_comptime_value(pointee_ty).unwrap();
				let ty = value::Key::Type(value::Type::Slice(TypeSlice { pointee_ty }));
				let ty = vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&ty));
				self.vuir_map.insert(id, ty);
				Ok((ty, ControlFlow::May))
			},
			vuir::Opcode::TypeArray {
				elem,
				is_const,
				len,
				sentinel,
				elem_span,
				len_span,
				span,
			} => {
				let elem_ty = self.resolve_type(block, elem, elem_span)?;
				let len = {
					let len = self.resolve_inst(len);
					let len = self.try_resolve_comptime_value(&len).ok_or_else(|| {
						self.push_error(
							Diagnostic::error()
								.with_message("array length must be known at comptime")
								.with_label(Label::primary().with_span(self.diag_span(*len_span))),
						);
						AnalyzeError::AnalysisFailed
					})?;
					let len = self.cu.values.index_to_key(len).as_int().1;
					len.to_u64().ok_or_else(|| {
						self.push_error(
							Diagnostic::error()
								.with_message(format!("array length `{len}` too big to len_span in an array"))
								.with_label(Label::primary().with_span(self.diag_span(*len_span)))
								.with_note(format!("maximum length of an array is {} (u64 maximum value)", u64::MAX)),
						);
						AnalyzeError::AnalysisFailed
					})?
				};

				let ty = vtir::InstructionRef::Interned(
					self.cu
						.values
						.intern_trivial(&value::Key::Type(value::Type::Array(value::TypeArray { elem_ty, len }))),
				);
				self.vuir_map.insert(id, ty);
				Ok((ty, ControlFlow::May))
			},
			vuir::Opcode::Branch {
				cond: (cond, cond_span),
				then_body,
				else_body,
				span: _,
			} => {
				let cond = self.resolve_inst(cond);
				let bool_ty = self.cu.values.common.bool_t;
				let cond = self.coerce(block, bool_ty, cond, cond_span)?;

				let inst = if let Some(cond) = self.try_resolve_comptime_value(&cond) {
					let body = if self.cu.values.index_to_key(cond).as_bool() {
						then_body
					} else {
						else_body
					};
					self.analyze_comptime_block(block, body)?
						.unwrap_or(self.cu.values.common.void_value.into())
				} else {
					let then_body = {
						let block = self.child_block(block);
						self.analyze_comptime_block(*block, then_body)?;
						self.unstack_block(block).instructions.into_bump_slice()
					};
					let else_body = {
						let block = self.child_block(block);
						self.analyze_comptime_block(*block, else_body)?;
						self.unstack_block(block).instructions.into_bump_slice()
					};
					let inst = self.inst(block, vtir::Opcode::Branch {
						cond,
						then_body,
						else_body,
					});
					self.vuir_map.insert(id, inst);
					inst
				};
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::Switch {
				operand,
				single_cases,
				multi_cases,
				else_body,
				span,
			} => {
				let operand_ref = self.resolve_inst(operand);

				// For tagged unions, extract the tag and switch on that
				let original_operand_ty = self.type_of(&operand_ref);
				let operand_ref = {
					if let (value::Key::Type(value::Type::Union(_)), value::Value::Union(u)) =
						self.cu.values.index_to_key_value(original_operand_ty)
					{
						let u = u.as_ref();
						if let Some(tag_ty) = u.tag_ty {
							self.inst(block, vtir::Opcode::UnionTag {
								union_val: operand_ref,
								tag_ty,
							})
						} else {
							self.push_error(
								Diagnostic::error()
									.with_message("cannot switch on a bare union")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}
					} else {
						operand_ref
					}
				};

				let mut vtir_cases = BumpVec::new_in(self.instructions_payload_alloc);

				// If the operand is comptime, resolve the switch at compile time
				if let Some(operand_val) = self.try_resolve_comptime_value(&operand_ref) {
					let operand_ty = self.type_of(&operand_ref);

					// Find the matching case
					let mut matched_body = None;

					for case in single_cases.iter() {
						let item = self.resolve_inst(&case.pattern);
						let item = self.coerce(block, operand_ty, item, &case.pattern_span)?;
						if self.try_resolve_comptime_value(&item) == Some(operand_val) {
							matched_body = Some(case.body);
							break;
						}
					}

					if matched_body.is_none() {
						for case in multi_cases.iter() {
							for item_ref in case.items.iter() {
								let item = self.resolve_inst(item_ref);
								let item = self.coerce(block, operand_ty, item, &case.patterns_span)?;
								if self.try_resolve_comptime_value(&item) == Some(operand_val) {
									matched_body = Some(case.body);
									break;
								}
							}
							if matched_body.is_some() {
								break;
							}
						}
					}

					let body = matched_body.or(*else_body).expect("non-exhaustive switch with comptime operand");

					let inst = self
						.analyze_comptime_block(block, body)?
						.unwrap_or(self.cu.values.common.void_value.into());
					self.vuir_map.insert(id, inst);
					return Ok((inst, ControlFlow::May));
				}

				// runtime switch

				let operand_ty = self.type_of(&operand_ref);

				// Process single-pattern cases
				for case in single_cases.iter() {
					let pattern = self.resolve_inst(&case.pattern);
					let pattern = self.coerce(block, operand_ty, pattern, &case.pattern_span)?;
					let body = {
						let child = self.child_block(block);
						let expected_len = self.blocks.len();
						self.analyze_comptime_block(*child, case.body)?;
						while self.blocks.len() > expected_len {
							self.blocks.pop();
						}
						self.unstack_block(child).instructions.into_bump_slice()
					};
					vtir_cases.push(vtir::SwitchCase {
						items: self.instructions_payload_alloc.alloc_slice_copy(&[pattern]),
						body,
					});
				}

				// Process multi-pattern cases
				for case in multi_cases.iter() {
					let mut items = BumpVec::new_in(self.instructions_payload_alloc);
					for item_ref in case.items.iter() {
						let item = self.resolve_inst(item_ref);
						let item = self.coerce(block, operand_ty, item, &case.patterns_span)?;
						items.push(item);
					}
					let body = {
						let child = self.child_block(block);
						let expected_len = self.blocks.len();
						self.analyze_comptime_block(*child, case.body)?;
						while self.blocks.len() > expected_len {
							self.blocks.pop();
						}
						self.unstack_block(child).instructions.into_bump_slice()
					};
					vtir_cases.push(vtir::SwitchCase {
						items: items.into_bump_slice(),
						body,
					});
				}

				// Check exhaustiveness: else is required unless switching on an enum/union
				// and all variants are covered
				if else_body.is_none() {
					let exhaustive_ty = if matches!(
						self.cu.values.index_to_key(original_operand_ty),
						value::Key::Type(value::Type::Union(_))
					) {
						original_operand_ty
					} else {
						operand_ty
					};
					let is_exhaustive = self.check_switch_exhaustive(exhaustive_ty, &vtir_cases, span);
					if !is_exhaustive {
						return Err(AnalyzeError::AnalysisFailed);
					}
				}

				// Process else body
				let else_body = if let Some(else_body) = else_body {
					let child = self.child_block(block);
					let expected_len = self.blocks.len();
					self.analyze_comptime_block(*child, else_body)?;
					while self.blocks.len() > expected_len {
						self.blocks.pop();
					}
					self.unstack_block(child).instructions.into_bump_slice()
				} else {
					&[]
				};

				let inst = self.inst(block, vtir::Opcode::Switch {
					operand: operand_ref,
					cases: vtir_cases.into_bump_slice(),
					else_body,
				});
				self.vuir_map.insert(id, inst);
				Ok((inst, ControlFlow::May))
			},
			vuir::Opcode::Defer { body, .. } => {
				// Vale model: defer consumes linear values at the defer site.
				// After `defer val.drop()`, `val` is considered moved.
				let _ = self.analyze_instruction_block(block, body)?;
				let value = self.cu.values.common.void_value.into();
				self.vuir_map.insert(id, value);
				Ok((value, ControlFlow::May))
			},
			vuir::Opcode::SwitchCapture {
				switch_operand,
				case_pattern,
				span,
			} => {
				let union_val = self.resolve_inst(switch_operand);
				let union_ty = self.type_of(&union_val);

				let (key, value_ref) = self.cu.values.index_to_key_value(union_ty);
				match (key, value_ref) {
					(value::Key::Type(value::Type::Union(_)), value::Value::Union(u)) if let Some(tag_ty) = u.tag_ty => {
						let field_idx = {
							let pattern = self.resolve_inst(case_pattern);
							let pattern_enum_tag = self.coerce(block, tag_ty, pattern, span)?.as_interned();
							let pattern_value = match self.cu.values.index_to_key(pattern_enum_tag) {
								value::Key::EnumTag { val, .. } => *val,
								_ => unreachable!(),
							};
							let Some(field_idx) = u.fields.iter().enumerate().find_map(|(field_idx, _)| {
								let field_tag = self.cu.values.intern_enum_tag_from_field_idx(tag_ty, field_idx as u32);
								let field_value = match self.cu.values.index_to_key(field_tag) {
									value::Key::EnumTag { val, .. } => *val,
									_ => unreachable!(),
								};
								(field_value == pattern_value).then_some(field_idx)
							}) else {
								self.push_error(
									Diagnostic::error()
										.with_message("switch pattern does not name a variant of this union")
										.with_label(Label::primary().with_span(self.diag_span(*span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							};
							field_idx
						};

						let field = &u.fields[field_idx];
						let ret_ty = field.ty.unwrap_or(self.cu.values.common.void_t);

						let inst = self.inst(block, vtir::Opcode::UnionFieldValue {
							union_val,
							field_idx,
							ret_ty,
						});
						self.vuir_map.insert(id, inst);
						Ok((inst, ControlFlow::May))
					},
					_ => {
						self.push_error(
							Diagnostic::error()
								.with_message("switch captures are only supported for tagged unions")
								.with_label(Label::primary().with_span(self.diag_span(*span))),
						);
						Err(AnalyzeError::AnalysisFailed)
					},
				}
			},
			vuir::Opcode::RvalueToLvalue { rvalue } => {
				let rvalue = self.resolve_inst(rvalue);
				if let Some(rvalue) = self.try_resolve_comptime_value(&rvalue) {
					let inst = self.cu.values.intern_value_ptr(rvalue).into();
					self.vuir_map.insert(id, inst);
					Ok((inst, ControlFlow::May))
				} else {
					let ty = self.type_of(&rvalue);
					let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
						pointee_ty: ty,
						packed: None,
						is_const: false,
					})));
					let inst = self.inst(block, vtir::Opcode::StackAlloc { ty: ptr_ty });
					self.inst(block, vtir::Opcode::Store { src: rvalue, dst: inst });
					self.vuir_map.insert(id, inst);
					Ok((inst, ControlFlow::May))
				}
			},
		}
	}

	fn analyze_arithmetic_op(
		&mut self,
		block: BlockId,
		inst: &vuir::Opcode,
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let (ty, lhs, rhs) = {
			let lhs_ty = self.type_of(&lhs);
			let rhs_ty = self.type_of(&rhs);

			// first check if types are arithmetic compatible
			if !self.cu.values.index_to_key(lhs_ty).type_is_numeric() {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"cannot perform an arithmetic operation with a `{}`",
							self.cu.values.display_index(lhs_ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}

			if !self.cu.values.index_to_key(rhs_ty).type_is_numeric() {
				self.push_error(
					Diagnostic::error()
						.with_message(format!("cannot perform an arithmetic operation with a `{rhs_ty:?}` (rhs type)`"))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}

			// then check if we can perform arithmetic ops between both types
			match (self.cu.values.index_to_key(lhs_ty), self.cu.values.index_to_key(rhs_ty)) {
				(l, r) if l == r => (lhs_ty, lhs, rhs),
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Int { .. })) => {
					(rhs_ty, self.coerce(block, rhs_ty, lhs, span)?, rhs)
				},
				(value::Key::Type(value::Type::Int { .. }), value::Key::Type(value::Type::Anyint)) => {
					let coerced = self.coerce(block, lhs_ty, rhs, span)?;
					(lhs_ty, lhs, coerced)
				},
				_ => {
					self.push_error(
						Diagnostic::error()
							.with_message(format!(
								"cannot perform an arithmetic operation between a '{}' and '{}'",
								self.cu.values.display_index(lhs_ty),
								self.cu.values.display_index(rhs_ty),
							))
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				},
			}
		};

		let inst = match (
			self.try_resolve_comptime_value(&lhs).map(|v| self.cu.values.index_to_key(v)),
			self.try_resolve_comptime_value(&rhs).map(|v| self.cu.values.index_to_key(v)),
		) {
			(Some(value::Key::Int { value: lhs, .. }), Some(value::Key::Int { value: rhs, .. })) => {
				let res = match inst {
					vuir::Opcode::Add { .. } => self.wrap_int_value(ty, lhs.as_ref() + rhs.as_ref()),
					vuir::Opcode::AddSat { .. } => lhs.as_ref() + rhs.as_ref(),
					vuir::Opcode::Sub { .. } => self.wrap_int_value(ty, lhs.as_ref() - rhs.as_ref()),
					vuir::Opcode::SubSat { .. } => lhs.as_ref() - rhs.as_ref(),
					vuir::Opcode::Mul { .. } => self.wrap_int_value(ty, lhs.as_ref() * rhs.as_ref()),
					vuir::Opcode::MulSat { .. } => lhs.as_ref() * rhs.as_ref(),
					vuir::Opcode::Div { .. } => lhs.as_ref() / rhs.as_ref(),
					vuir::Opcode::Rem { .. } => lhs.as_ref() % rhs.as_ref(),
					_ => unreachable!("{:?}", inst),
				};
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int {
					ty,
					value: Intern::new(res),
				}))
			},
			(Some(value::Key::Float { value: lhs, .. }), Some(value::Key::Float { value: rhs, .. })) => {
				let res = match inst {
					vuir::Opcode::Add { .. } => **lhs + **rhs,
					vuir::Opcode::Sub { .. } => **lhs - **rhs,
					vuir::Opcode::Mul { .. } => **lhs * **rhs,
					vuir::Opcode::Div { .. } => **lhs / **rhs,
					vuir::Opcode::Rem { .. } => todo!("float mod comptime op"),
					vuir::Opcode::AddSat { .. } | vuir::Opcode::SubSat { .. } | vuir::Opcode::MulSat { .. } => {
						// TODO(ldubos): create a proper error message for unsupported float ops
						return Err(AnalyzeError::AnalysisFailed);
					},
					vuir::Opcode::Lt { .. } => {
						return Ok(vtir::InstructionRef::Interned(
							self.cu.values.intern_trivial(&value::Key::Bool(**lhs < **rhs)),
						));
					},
					vuir::Opcode::Lte { .. } => {
						return Ok(vtir::InstructionRef::Interned(
							self.cu.values.intern_trivial(&value::Key::Bool(**lhs <= **rhs)),
						));
					},
					vuir::Opcode::Gt { .. } => {
						return Ok(vtir::InstructionRef::Interned(
							self.cu.values.intern_trivial(&value::Key::Bool(**lhs > **rhs)),
						));
					},
					vuir::Opcode::Gte { .. } => {
						return Ok(vtir::InstructionRef::Interned(
							self.cu.values.intern_trivial(&value::Key::Bool(**lhs >= **rhs)),
						));
					},
					_ => unreachable!(),
				};
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Float { ty, value: Anyfloat(res) }))
			},
			(Some(value::Key::Bool(lhs)), Some(value::Key::Bool(rhs))) => {
				let res = match inst {
					vuir::Opcode::BoolAnd { .. } => *lhs && *rhs,
					vuir::Opcode::BoolOr { .. } => *lhs || *rhs,
					_ => unreachable!(),
				};
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(res)))
			},
			// no comptime value, emit runtime insts
			_ => match inst {
				vuir::Opcode::Add { .. } => self.inst(block, vtir::Opcode::Add { lhs, rhs }),
				vuir::Opcode::AddSat { .. } => self.inst(block, vtir::Opcode::AddSat { lhs, rhs }),
				vuir::Opcode::Sub { .. } => self.inst(block, vtir::Opcode::Sub { lhs, rhs }),
				vuir::Opcode::SubSat { .. } => self.inst(block, vtir::Opcode::SubSat { lhs, rhs }),
				vuir::Opcode::Mul { .. } => self.inst(block, vtir::Opcode::Mul { lhs, rhs }),
				vuir::Opcode::MulSat { .. } => self.inst(block, vtir::Opcode::MulSat { lhs, rhs }),
				vuir::Opcode::Div { .. } => self.inst(block, vtir::Opcode::Div { lhs, rhs }),
				vuir::Opcode::Rem { .. } => self.inst(block, vtir::Opcode::Rem { lhs, rhs }),
				vuir::Opcode::BoolAnd { .. } => self.inst(block, vtir::Opcode::BoolAnd { lhs, rhs }),
				vuir::Opcode::BoolOr { .. } => self.inst(block, vtir::Opcode::BoolOr { lhs, rhs }),
				_ => unreachable!(),
			},
		};
		Ok(inst)
	}

	fn wrap_int_value(
		&self,
		ty: value::Index,
		value: Anyint,
	) -> Anyint {
		let value::Key::Type(ty) = self.cu.values.index_to_key(ty) else {
			return value;
		};
		let (signed, bits) = match ty {
			value::Type::Int { signed, bits } => (*signed, *bits as u64),
			value::Type::Isize => (true, self.cu.resolved_target.ptr_width_in_bits as u64),
			value::Type::Usize => (false, self.cu.resolved_target.ptr_width_in_bits as u64),
			// `Anyint` aren't subject to wrap semantic
			value::Type::Anyint
			| value::Type::Anyfloat
			| value::Type::F16
			| value::Type::F32
			| value::Type::F64
			| value::Type::F128
			| value::Type::Bool
			| value::Type::Void
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
			| value::Type::Never
			| value::Type::EnumLiteral => return value,
		};

		let one = Anyint::from(1u64);
		// modulus = 2^bits
		let modulus = &one << bits;

		// compute the wrapped value in the range: (-2^bits, 2^bits),
		// depending on remainder semantics
		let mut wrapped = &value % &modulus;

		// normalize negative remainders into: [0, 2^bits)
		if wrapped.is_negative() {
			wrapped = &wrapped + &modulus;
		}

		// for signed integers, reinterpret the upper half of the
		// unsigned range as negative values.
		//
		// e.g.
		//
		//   u-range :    0..255
		//   s-range : -128..127
		if signed {
			let sign_bit = &one << (bits - 1);
			if wrapped >= sign_bit {
				wrapped = &wrapped - &modulus;
			}
		}

		wrapped
	}

	fn analyze_bitwise_op(
		&mut self,
		block: BlockId,
		inst: &vuir::Opcode,
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let (ty, lhs, rhs) = {
			let lhs_ty = self.type_of(&lhs);
			let rhs_ty = self.type_of(&rhs);

			let lhs_key = self.cu.values.index_to_key(lhs_ty);
			let rhs_key = self.cu.values.index_to_key(rhs_ty);

			// bitwise ops work on integers and bools
			let is_valid = |k: &value::Key| {
				matches!(
					k,
					value::Key::Type(value::Type::Int { .. })
						| value::Key::Type(value::Type::Anyint)
						| value::Key::Type(value::Type::Isize)
						| value::Key::Type(value::Type::Usize)
						| value::Key::Type(value::Type::Bool)
				)
			};

			if !is_valid(lhs_key) {
				self.push_error(
					Diagnostic::error()
						.with_message(format!("cannot perform a bitwise operation with a `{lhs_ty:?}` (lhs type)`"))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}

			if !is_valid(rhs_key) {
				self.push_error(
					Diagnostic::error()
						.with_message(format!("cannot perform a bitwise operation with a `{rhs_ty:?}` (rhs type)`"))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			}

			match (lhs_key, rhs_key) {
				(l, r) if l == r => (lhs_ty, lhs, rhs),
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Int { .. }))
				| (value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Usize))
				| (value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Isize)) => {
					(rhs_ty, self.coerce(block, rhs_ty, lhs, span)?, rhs)
				},
				(value::Key::Type(value::Type::Int { .. }), value::Key::Type(value::Type::Anyint))
				| (value::Key::Type(value::Type::Usize), value::Key::Type(value::Type::Anyint))
				| (value::Key::Type(value::Type::Isize), value::Key::Type(value::Type::Anyint)) => {
					let coerced = self.coerce(block, lhs_ty, rhs, span)?;
					(lhs_ty, lhs, coerced)
				},
				_ => {
					self.push_error(
						Diagnostic::error()
							.with_message(format!(
								"cannot perform a bitwise operation between a '{lhs_ty:?}' and '{rhs_ty:?}'"
							))
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				},
			}
		};

		let inst = match (
			self.try_resolve_comptime_value(&lhs).map(|v| self.cu.values.index_to_key(v)),
			self.try_resolve_comptime_value(&rhs).map(|v| self.cu.values.index_to_key(v)),
		) {
			(Some(value::Key::Int { value: lhs, .. }), Some(value::Key::Int { value: rhs, .. })) => {
				let res = match inst {
					vuir::Opcode::BitAnd { .. } => lhs.as_ref() & rhs.as_ref(),
					vuir::Opcode::BitOr { .. } => lhs.as_ref() | rhs.as_ref(),
					vuir::Opcode::BitXor { .. } => lhs.as_ref() ^ rhs.as_ref(),
					vuir::Opcode::Shl { .. } | vuir::Opcode::ShlSat { .. } | vuir::Opcode::ShlWrap { .. } => {
						if rhs.is_negative() {
							self.push_error(
								Diagnostic::error()
									.with_message("negative shift amount in comptime shift")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}
						let shift: u64 = rhs.to_u64().unwrap_or({
							// shift amount too large to fit in u64 => always an overflow
							u64::MAX
						});
						// For concrete types, check shift >= bit_width
						if !matches!(self.cu.values.index_to_key(ty), value::Key::Type(value::Type::Anyint)) {
							let bit_width = self.cu.values.type_bit_size(ty) as u64;
							if shift >= bit_width && !matches!(inst, vuir::Opcode::ShlWrap { .. }) {
								self.push_error(
									Diagnostic::error()
										.with_message(format!("shift amount ({shift}) is >= the bit width ({bit_width}) of the type"))
										.with_label(Label::primary().with_span(self.diag_span(*span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							}
						}
						lhs.as_ref() << shift
					},
					vuir::Opcode::Shr { .. } | vuir::Opcode::ShrSat { .. } | vuir::Opcode::ShrWrap { .. } => {
						if rhs.is_negative() {
							self.push_error(
								Diagnostic::error()
									.with_message("negative shift amount in comptime shift")
									.with_label(Label::primary().with_span(self.diag_span(*span))),
							);
							return Err(AnalyzeError::AnalysisFailed);
						}
						let shift: u64 = rhs.to_u64().unwrap_or(u64::MAX);
						if !matches!(self.cu.values.index_to_key(ty), value::Key::Type(value::Type::Anyint)) {
							let bit_width = self.cu.values.type_bit_size(ty) as u64;
							if shift >= bit_width && !matches!(inst, vuir::Opcode::ShrWrap { .. }) {
								self.push_error(
									Diagnostic::error()
										.with_message(format!("shift amount ({shift}) is >= the bit width ({bit_width}) of the type"))
										.with_label(Label::primary().with_span(self.diag_span(*span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							}
						}
						lhs.as_ref() >> shift
					},
					_ => unreachable!(),
				};
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int {
					ty,
					value: Intern::new(res),
				}))
			},
			(Some(value::Key::Bool(lhs)), Some(value::Key::Bool(rhs))) => {
				let res = match inst {
					vuir::Opcode::BitAnd { .. } => *lhs & *rhs,
					vuir::Opcode::BitOr { .. } => *lhs | *rhs,
					vuir::Opcode::BitXor { .. } => *lhs ^ *rhs,
					_ => {
						self.push_error(
							Diagnostic::error()
								.with_message("shift operations are not supported on bool")
								.with_label(Label::primary().with_span(self.diag_span(*span))),
						);
						return Err(AnalyzeError::AnalysisFailed);
					},
				};
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(res)))
			},
			_ => match inst {
				vuir::Opcode::Shl { .. } => self.inst(block, vtir::Opcode::Shl { lhs, rhs }),
				vuir::Opcode::ShlSat { .. } => self.inst(block, vtir::Opcode::ShlSat { lhs, rhs }),
				vuir::Opcode::ShlWrap { .. } => self.inst(block, vtir::Opcode::ShlWrap { lhs, rhs }),
				vuir::Opcode::Shr { .. } => self.inst(block, vtir::Opcode::Shr { lhs, rhs }),
				vuir::Opcode::ShrSat { .. } => self.inst(block, vtir::Opcode::ShrSat { lhs, rhs }),
				vuir::Opcode::ShrWrap { .. } => self.inst(block, vtir::Opcode::ShrWrap { lhs, rhs }),
				vuir::Opcode::BitAnd { .. } => self.inst(block, vtir::Opcode::BitAnd { lhs, rhs }),
				vuir::Opcode::BitOr { .. } => self.inst(block, vtir::Opcode::BitOr { lhs, rhs }),
				vuir::Opcode::BitXor { .. } => self.inst(block, vtir::Opcode::BitXor { lhs, rhs }),
				_ => unreachable!(),
			},
		};
		Ok(inst)
	}

	fn analyze_bitwise_not_op(
		&mut self,
		block: BlockId,
		op: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let op_ty = self.type_of(&op);
		let op_key = self.cu.values.index_to_key(op_ty);

		let is_valid = matches!(
			op_key,
			value::Key::Type(value::Type::Int { .. })
				| value::Key::Type(value::Type::Anyint)
				| value::Key::Type(value::Type::Isize)
				| value::Key::Type(value::Type::Usize)
				| value::Key::Type(value::Type::Bool)
		);

		if !is_valid {
			self.push_error(
				Diagnostic::error()
					.with_message(format!("cannot perform bitwise NOT on a `{op_ty:?}`"))
					.with_label(Label::primary().with_span(self.diag_span(*span))),
			);
			return Err(AnalyzeError::AnalysisFailed);
		}

		let inst = match self.try_resolve_comptime_value(&op).map(|v| self.cu.values.index_to_key(v)) {
			Some(value::Key::Int { value, .. }) => {
				let res = !value.as_ref();
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int {
					ty: op_ty,
					value: Intern::new(res),
				}))
			},
			Some(value::Key::Bool(b)) => vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(!*b))),
			_ => self.inst(block, vtir::Opcode::BitNot { op }),
		};
		Ok(inst)
	}

	fn analyze_comparaison_op(
		&mut self,
		block: BlockId,
		inst: &vuir::Opcode,
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let (ty, lhs, rhs) = {
			let lhs_ty = self.type_of(&lhs);
			let rhs_ty = self.type_of(&rhs);
			match (self.cu.values.index_to_key(lhs_ty), self.cu.values.index_to_key(rhs_ty)) {
				(l, r) if l == r => (lhs_ty, lhs, rhs),
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Int { .. })) => {
					(rhs_ty, self.coerce(block, rhs_ty, lhs, span)?, rhs)
				},
				(value::Key::Type(value::Type::Int { .. }), value::Key::Type(value::Type::Anyint)) => {
					let coerced = self.coerce(block, lhs_ty, rhs, span)?;
					(lhs_ty, lhs, coerced)
				},
				_ => {
					self.push_error(
						Diagnostic::error()
							.with_message(format!(
								"cannot compare a `{}` with an `{}`",
								self.cu.values.display_index(lhs_ty),
								self.cu.values.display_index(rhs_ty)
							))
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				},
			}
		};

		#[inline]
		fn cmp_comptime<T>(
			opcode: &vuir::Opcode,
			lhs: &T,
			rhs: &T,
		) -> bool
		where
			T: PartialEq + PartialOrd,
		{
			match opcode {
				vuir::Opcode::Eq { .. } => lhs >= rhs,
				vuir::Opcode::Neq { .. } => lhs >= rhs,
				vuir::Opcode::Lt { .. } => lhs < rhs,
				vuir::Opcode::Lte { .. } => lhs <= rhs,
				vuir::Opcode::Gt { .. } => lhs > rhs,
				vuir::Opcode::Gte { .. } => lhs >= rhs,
				_ => unreachable!("{:?} not implemented", opcode),
			}
		}

		let inst = match (
			self.try_resolve_comptime_value(&lhs).map(|v| self.cu.values.index_to_key(v)),
			self.try_resolve_comptime_value(&rhs).map(|v| self.cu.values.index_to_key(v)),
		) {
			(Some(value::Key::Int { value: lhs, .. }), Some(value::Key::Int { value: rhs, .. })) => {
				let res = cmp_comptime(inst, lhs, rhs);
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(res)))
			},
			(Some(value::Key::Float { value: lhs, .. }), Some(value::Key::Float { value: rhs, .. })) => {
				let res = cmp_comptime(inst, lhs, rhs);
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(res)))
			},
			(Some(value::Key::Bool(lhs)), Some(value::Key::Bool(rhs))) => {
				let res = cmp_comptime(inst, lhs, rhs);
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Bool(res)))
			},
			// no comptime value, emit runtime insts
			_ => match inst {
				vuir::Opcode::Eq { .. } => self.inst(block, vtir::Opcode::Eq { lhs, rhs }),
				vuir::Opcode::Neq { .. } => self.inst(block, vtir::Opcode::Neq { lhs, rhs }),
				vuir::Opcode::Lt { .. } => self.inst(block, vtir::Opcode::Lt { lhs, rhs }),
				vuir::Opcode::Lte { .. } => self.inst(block, vtir::Opcode::Lte { lhs, rhs }),
				vuir::Opcode::Gt { .. } => self.inst(block, vtir::Opcode::Gt { lhs, rhs }),
				vuir::Opcode::Gte { .. } => self.inst(block, vtir::Opcode::Gte { lhs, rhs }),
				vuir::Opcode::BoolAnd { .. } => self.inst(block, vtir::Opcode::BoolAnd { lhs, rhs }),
				vuir::Opcode::BoolOr { .. } => self.inst(block, vtir::Opcode::BoolOr { lhs, rhs }),
				_ => unreachable!("{inst:?}"),
			},
		};
		Ok(inst)
	}

	fn analyze_field_val(
		&mut self,
		block: BlockId,
		lhs: vtir::InstructionRef,
		field: &Intern<str>,
		field_span: &Span,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let lhs_ty = self.type_of(&lhs);

		let (lhs_pointee_ty, lhs_is_pointer_to) = match self.cu.values.index_to_key(lhs_ty) {
			value::Key::Type(value::Type::Ptr(ptr)) => (ptr.pointee_ty, true),
			_ => (lhs_ty, false),
		};

		let inst = match self.cu.values.index_to_key_value(lhs_pointee_ty) {
			// lhs is a type
			(value::Key::Type(value::Type::Type), _) => {
				let lhs = if lhs_is_pointer_to {
					self.analyze_load(block, lhs, span)?
				} else {
					lhs
				};
				let lhs = self.try_resolve_comptime_value(&lhs).unwrap();
				match self.cu.values.index_to_key_value(lhs) {
					(value::Key::Type(value::Type::Struct(s)), value::Value::Struct(r#struct)) => {
						let r#struct = r#struct.as_ref();
						let decl = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[r#struct.namespace].decls.get(field).copied())
							.ok_or_else(|| {
								self.diag_decl_not_found(field, lhs, field_span);
								AnalyzeError::AnalysisFailed
							})?;
						let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
						vtir::InstructionRef::Interned(value)
					},
					(value::Key::Type(value::Type::Enum(_)), value::Value::Enum(e)) => {
						if let Some(decl) = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[e.namespace].decls.get(field).copied())
						{
							let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
							return Ok(vtir::InstructionRef::Interned(value));
						}
						let field_idx = e.field_idx_by_name(field).ok_or_else(|| {
							self.diag_field_not_found(field, lhs, field_span);
							AnalyzeError::AnalysisFailed
						})?;
						let tag = value::Key::EnumTag {
							enum_ty: lhs,
							val: e.fields[field_idx].value,
						};
						vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&tag))
					},
					(value::Key::Type(value::Type::Union(_)), value::Value::Union(u)) => {
						let u = u.as_ref();
						// check if it's a union field (tag value)
						if let Some(field_idx) = u.field_idx_by_name(field) {
							if let Some(tag_ty) = u.tag_ty {
								vtir::InstructionRef::Interned(self.cu.values.intern_enum_tag_from_field_idx(tag_ty, field_idx))
							} else {
								self.push_error(
									Diagnostic::error()
										.with_message("cannot access field on a bare union as a value")
										.with_label(Label::primary().with_span(self.diag_span(*span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							}
						} else if let Some(decl) = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[u.namespace].decls.get(field).copied())
						{
							let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
							vtir::InstructionRef::Interned(value)
						} else {
							self.diag_decl_not_found(field, lhs, span);
							return Err(AnalyzeError::AnalysisFailed);
						}
					},
					_ => {
						unreachable!("{:?}", self.cu.values.index_to_key_value(lhs))
					},
				}
			},
			// lhs is a value
			(value::Key::Type(value::Type::Struct(s)), value::Value::Struct(r#struct)) => {
				let r#struct = r#struct.as_ref();
				if let Some(field_idx) = r#struct.field_idx_by_name(field) {
					let field_ty = r#struct.fields[field_idx].ty;
					let lhs = if lhs_is_pointer_to {
						self.analyze_load(block, lhs, span)?
					} else {
						lhs
					};
					self.inst(block, vtir::Opcode::StructFieldValue {
						struct_ty: lhs,
						field_idx,
						ret_ty: field_ty,
					})
				} else if let Some(decl) = self
					.cu
					.namespaces
					.with(|namespaces| namespaces[r#struct.namespace].decls.get(field).copied())
				{
					let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
					vtir::InstructionRef::Interned(value)
				} else {
					self.diag_field_not_found(field, lhs_ty, field_span);
					return Err(AnalyzeError::AnalysisFailed);
				}
			},
			(value::Key::Type(value::Type::Ptr(ptr)), _) => {
				let pointee_ty = ptr.pointee_ty;
				let value::Value::Struct(r#struct) = self.cu.values.index_to_value(pointee_ty) else {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("type `{}` doesn't have fields", self.cu.values.display_index(pointee_ty)))
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				};

				let r#struct = r#struct.as_ref();
				if let Some(field_idx) = r#struct.field_idx_by_name(field) {
					let loaded_struct = self.inst(block, vtir::Opcode::Load { ptr: lhs });
					let field_ty = r#struct.fields[field_idx].ty;
					self.inst(block, vtir::Opcode::StructFieldValue {
						struct_ty: loaded_struct,
						field_idx,
						ret_ty: field_ty,
					})
				} else if let Some(decl) = self
					.cu
					.namespaces
					.with(|namespaces| namespaces[r#struct.namespace].decls.get(field).copied())
				{
					let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
					vtir::InstructionRef::Interned(value)
				} else {
					self.diag_field_not_found(field, pointee_ty, field_span);
					return Err(AnalyzeError::AnalysisFailed);
				}
			},
			_ => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"type `{}` doesn't have any field or declaration",
							self.cu.values.display_index(lhs_ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};
		Ok(inst)
	}

	fn analyze_field_ptr(
		&mut self,
		block: BlockId,
		lhs: vtir::InstructionRef,
		field: &Intern<str>,
		field_span: &Span,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let lhs_ty = self.type_of(&lhs);
		let lhs_ptr_ty = match self.cu.values.index_to_key(lhs_ty) {
			value::Key::Type(value::Type::Ptr(ptr)) => *ptr,
			test => {
				let lhs = self.cu.values.index_to_key_value(lhs.as_interned());
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"expected pointer, found `{}` ({lhs:?}, {test:?})",
							self.cu.values.display_index(lhs_ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};

		// if pointee is a pointer, dereference it, allow one dereference at most
		let (lhs_pointee_ty, lhs_is_ptr) = match self.cu.values.index_to_key(lhs_ptr_ty.pointee_ty) {
			value::Key::Type(value::Type::Ptr(ptr)) => (ptr.pointee_ty, true),
			_ => (lhs_ptr_ty.pointee_ty, false),
		};

		let inst = match self.cu.values.index_to_key_value(lhs_pointee_ty) {
			// lhs is a type
			(value::Key::Type(value::Type::Type), _) => {
				// load the field pointer
				let lhs = self.analyze_load(block, lhs, span)?;

				// if pointee is ptr, deref it too
				let lhs = if lhs_is_ptr { self.analyze_load(block, lhs, span)? } else { lhs };

				let lhs = self.try_resolve_comptime_value(&lhs).unwrap();
				match self.cu.values.index_to_key_value(lhs) {
					(value::Key::Type(value::Type::Struct(s)), value::Value::Struct(r#struct)) => {
						let r#struct = r#struct.as_ref();
						let decl = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[r#struct.namespace].decls.get(field).copied())
							.ok_or_else(|| {
								self.diag_decl_not_found(field, lhs, span);
								AnalyzeError::AnalysisFailed
							})?;

						// TODO(zino): Unify with other places we get decl value... where we get ptr
						let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
						let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
							pointee_ty: self.cu.values.type_of_interned(value),
							packed: None,
							is_const: true,
						})));
						let ptr = self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
							ty: ptr_ty,
							kind: PtrKind::Decl(decl),
						}));
						vtir::InstructionRef::Interned(ptr)
					},
					(value::Key::Type(value::Type::Enum(_)), value::Value::Enum(e)) => {
						if let Some(field_idx) = e.field_idx_by_name(field) {
							let tag = value::Key::EnumTag {
								enum_ty: lhs,
								val: e.fields[field_idx].value,
							};
							let value = self.cu.values.intern_trivial(&tag);
							let value = self.cu.values.intern_value_ptr(value);
							vtir::InstructionRef::Interned(value)
						} else if let Some(decl) = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[e.namespace].decls.get(field).copied())
						{
							let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
							let value = self.cu.values.intern_value_ptr(value);
							vtir::InstructionRef::Interned(value)
						} else {
							self.diag_decl_not_found(field, lhs, span);
							return Err(AnalyzeError::AnalysisFailed);
						}
					},
					(value::Key::Type(value::Type::Union(_)), value::Value::Union(u)) => {
						let u = u.as_ref();
						// Check if it's a union field (tag value)
						if let Some(field_idx) = u.field_idx_by_name(field) {
							if let Some(tag_ty) = u.tag_ty {
								let tag = self.cu.values.intern_enum_tag_from_field_idx(tag_ty, field_idx);
								vtir::InstructionRef::Interned(self.cu.values.intern_value_ptr(tag))
							} else {
								self.push_error(
									Diagnostic::error()
										.with_message("cannot access field on a bare union as a value")
										.with_label(Label::primary().with_span(self.diag_span(*span))),
								);
								return Err(AnalyzeError::AnalysisFailed);
							}
						} else if let Some(decl) = self
							.cu
							.namespaces
							.with(|namespaces| namespaces[u.namespace].decls.get(field).copied())
						{
							let value = self.cu.get_or_analyze_decl_value(decl)?.unwrap();
							let value = self.cu.values.intern_value_ptr(value);
							vtir::InstructionRef::Interned(value)
						} else {
							self.diag_decl_not_found(field, lhs, span);
							return Err(AnalyzeError::AnalysisFailed);
						}
					},
					_ => {
						unreachable!("{:?}", self.cu.values.index_to_key_value(lhs))
					},
				}
			},
			// lhs is a value
			(value::Key::Type(value::Type::Struct(s)), value::Value::Struct(r#struct)) => {
				let r#struct = r#struct.as_ref();
				if let Some(field_idx) = r#struct.field_idx_by_name(field) {
					let field_ty = r#struct.fields[field_idx].ty;
					let ptr_to_field_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
						pointee_ty: field_ty,
						packed: None,
						is_const: lhs_ptr_ty.is_const,
					})));
					let struct_ptr = if lhs_is_ptr { self.analyze_load(block, lhs, span)? } else { lhs };
					self.inst(block, vtir::Opcode::StructFieldPtr {
						struct_ptr,
						field_idx,
						ret_ty: ptr_to_field_ty,
					})
				} else {
					self.diag_field_not_found(field, lhs_pointee_ty, span);
					return Err(AnalyzeError::AnalysisFailed);
				}
			},
			_ => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"type `{}` doesn't have fields",
							self.cu.values.display_index(lhs_pointee_ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};

		Ok(inst)
	}

	fn analyze_decl_val(
		&mut self,
		block: BlockId,
		decl: DeclId,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let ptr = self.analyze_decl_ptr(block, decl)?;
		self.analyze_load(block, ptr, span)
	}

	fn analyze_decl_ptr(
		&mut self,
		block: BlockId,
		decl: DeclId,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let Some(value) = self.cu.get_or_analyze_decl_value(decl)? else {
			return Err(AnalyzeError::AnalysisFailed);
		};
		let ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
			pointee_ty: self.cu.values.type_of_interned(value),
			packed: None,
			is_const: true,
		})));
		let ptr = self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
			ty: ptr_ty,
			kind: PtrKind::Decl(decl),
		}));
		Ok(vtir::InstructionRef::Interned(ptr))
	}

	fn analyze_array_index_ptr(
		&mut self,
		block: BlockId,
		array_ptr: vuir::InstructionRef,
		index: vuir::InstructionRef,
		span: DiagSpan,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let array_ptr = self.resolve_inst(&array_ptr);
		let index = self.resolve_inst(&index);

		let array_ptr_ty = self.type_of(&array_ptr);
		let array_pointee_ty = match self.cu.values.index_to_key(array_ptr_ty) {
			value::Key::Type(value::Type::Ptr(ptr)) => ptr.pointee_ty,
			_ => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"indexing requires a pointer as source, is `{}`",
							self.cu.values.display_index(array_ptr_ty)
						))
						.with_label(Label::primary().with_span(span)),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};

		let const_index: Option<usize> = self.try_resolve_comptime_value(&index).and_then(|index| {
			let value::Key::Int { value, .. } = self.cu.values.index_to_key(index) else {
				return None;
			};
			value.to_u64().and_then(|value| value.try_into().ok())
		});

		if let Some(index_usize) = const_index {
			let const_ptr_target = |ptr: value::Ptr| -> Option<value::Index> {
				match ptr.kind {
					PtrKind::Value(value) => Some(value),
					PtrKind::Decl(decl) => self.cu.decls.with_mut(|decls| {
						let DeclAnalysisState::Analysed { value } = decls[decl].analysis_state else {
							unreachable!("constant array backing decl must be analyzed");
						};
						Some(value)
					}),
					PtrKind::ComptimeAlloc(_) => None,
				}
			};

			let const_elem = match self.cu.values.index_to_key(array_pointee_ty) {
				value::Key::Type(value::Type::Slice(_)) => {
					let slice = self.analyze_load(block, array_ptr, &span.span)?;
					let slice = self.try_resolve_comptime_value(&slice);
					match slice.map(|slice| self.cu.values.index_to_key(slice)) {
						Some(value::Key::Str { value, .. }) => value.get(index_usize).map(|byte| {
							let u8_ty = self
								.cu
								.values
								.intern_trivial(&value::Key::Type(value::Type::Int { signed: false, bits: 8 }));
							self.cu.values.intern_trivial(&value::Key::Int {
								ty: u8_ty,
								value: Anyint::from(*byte).into(),
							})
						}),
						Some(value::Key::Slice { ptr, len, .. }) => {
							let len: usize = match self.cu.values.index_to_key(*len) {
								value::Key::Int { value, .. } => value.to_u64().unwrap().try_into().unwrap(),
								_ => unreachable!("slice len must be an integer"),
							};
							if index_usize >= len {
								None
							} else {
								if let Some(backing) = const_ptr_target(*self.cu.values.index_to_key(*ptr).as_ptr()) {
									match self.cu.values.index_to_key(backing) {
										value::Key::Aggregate { values, .. } => values.get(index_usize).copied(),
										_ => None,
									}
								} else {
									None
								}
							}
						},
						_ => None,
					}
				},
				value::Key::Type(value::Type::Array(_)) => {
					let backing = self
						.try_resolve_comptime_value(&array_ptr)
						.and_then(|ptr| const_ptr_target(*self.cu.values.index_to_key(ptr).as_ptr()));
					match backing.map(|backing| self.cu.values.index_to_key(backing)) {
						Some(value::Key::Aggregate { values, .. }) => values.get(index_usize).copied(),
						_ => None,
					}
				},
				_ => None,
			};

			if let Some(value) = const_elem {
				let elem_ptr_ty = match self.cu.values.index_to_key(array_pointee_ty) {
					value::Key::Type(value::Type::Slice(slice)) => {
						self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
							pointee_ty: slice.pointee_ty,
							packed: None,
							is_const: true,
						})))
					},
					value::Key::Type(value::Type::Array(array)) => {
						self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
							pointee_ty: array.elem_ty,
							packed: None,
							is_const: true,
						})))
					},
					_ => unreachable!(),
				};
				let ptr = self.cu.values.intern_trivial(&value::Key::Ptr(Ptr {
					ty: elem_ptr_ty,
					kind: PtrKind::Value(value),
				}));
				return Ok(ptr.into());
			}
		}

		let inst = match self.cu.values.index_to_key(array_pointee_ty) {
			value::Key::Type(value::Type::Slice(slice)) => {
				let elem_ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
					pointee_ty: slice.pointee_ty,
					packed: None,
					is_const: false, // TODO(zino): const
				})));
				let slice = self.analyze_load(block, array_ptr, &span.span)?; // TODO(zino): analyze_load should use diagspan
				self.inst(block, vtir::Opcode::SliceElemPtr { slice, index, elem_ptr_ty })
			},
			value::Key::Type(value::Type::Array(array)) => {
				let elem_ptr_ty = self.cu.values.intern_trivial(&value::Key::Type(value::Type::Ptr(value::TypePtr {
					pointee_ty: array.elem_ty,
					packed: None,
					is_const: false, // TODO(zino): const
				})));
				self.inst(block, vtir::Opcode::PtrElemPtr {
					array_ptr,
					index,
					elem_ptr_ty,
				})
			},
			_ => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"type `{}` is not indexable",
							self.cu.values.display_index(array_pointee_ty)
						))
						.with_label(Label::primary().with_span(span))
						.with_note("only slices and arrays are indexable"),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};

		Ok(inst)
	}

	fn analyze_union_init(
		&mut self,
		id: vuir::InstructionId,
		block: BlockId,
		ty: value::Index,
		fields: &[vuir::AdtInitField],
		span: &Span,
	) -> Result<(vtir::InstructionRef, ControlFlow), AnalyzeError> {
		let union_ty = self.cu.values.index_to_value(ty).as_union();
		let union_ty = union_ty.as_ref();

		if fields.len() != 1 {
			self.push_error(
				Diagnostic::error()
					.with_message("union initialization requires exactly one field")
					.with_label(Label::primary().with_span(self.diag_span(*span))),
			);
			return Err(AnalyzeError::AnalysisFailed);
		}

		let field = &fields[0];
		let field_idx = union_ty.field_idx_by_name(&field.name.symbol).ok_or_else(|| {
			self.diag_field_not_found(&field.name.symbol, ty, &field.name.span);
			AnalyzeError::AnalysisFailed
		})?;

		let union_field = &union_ty.fields[field_idx as usize];
		let value = if let Some(field_ty) = union_field.ty {
			let value = self.resolve_inst(&field.value);
			let value = self.coerce(block, field_ty, value, &field.span)?;
			Some(value)
		} else {
			None
		};

		let inst = match value {
			Some(value) if let Some(payload) = self.try_resolve_comptime_value(&value) => {
				let tag = union_ty
					.tag_ty
					.map(|tag_ty| self.cu.values.intern_enum_tag_from_field_idx(tag_ty, field_idx));
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Union {
					ty,
					tag,
					payload: Some(payload),
				}))
			},
			None => {
				let tag = union_ty
					.tag_ty
					.map(|tag_ty| self.cu.values.intern_enum_tag_from_field_idx(tag_ty, field_idx));
				vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Union { ty, tag, payload: None }))
			},
			value => self.inst(block, vtir::Opcode::UnionInit {
				union_ty: ty,
				field_idx: field_idx as _,
				value,
			}),
		};

		self.vuir_map.insert(id, inst);
		Ok((inst, ControlFlow::May))
	}

	fn analyze_struct_init(
		&mut self,
		id: vuir::InstructionId,
		block: BlockId,
		ty: value::Index,
		fields: &[vuir::AdtInitField],
		span: &Span,
	) -> Result<(vtir::InstructionRef, ControlFlow), AnalyzeError> {
		let ty = {
			if matches!(*self.cu.values.index_to_key(ty), value::Key::Type(value::Type::Struct(..))) {
				Ok(ty)
			} else {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"`{}` must be a struct or union to be initialized with init syntax",
							self.cu.values.display_index(ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				Err(AnalyzeError::AnalysisFailed)
			}
		}?;
		let r#struct = self.cu.values.index_to_value(ty).as_struct();
		let r#struct = r#struct.as_ref();
		let mut analysis_failed = false;
		let (fields, fields_comptime_known) = {
			let mut fields_comptime_known = true;
			let mut out_fields = vec![None; r#struct.fields.len()];
			for field in fields.iter() {
				let Some((field_idx, value)) = ('value: {
					let Some(field_idx) = r#struct.field_idx_by_name(&field.name.symbol) else {
						self.diag_field_not_found(&field.name.symbol, ty, &field.name.span);
						analysis_failed = true;
						break 'value None; // does not exist: filtered
					};

					let field_ty = r#struct.fields[field_idx].ty;
					let value = self.resolve_inst(&field.value);
					let Ok(value) = self.coerce(block, field_ty, value, &field.span) else {
						analysis_failed = true;
						break 'value Some((field_idx, None)); // exist but no value
					};

					fields_comptime_known &= value.is_interned();

					Some((field_idx, Some(value)))
				}) else {
					continue; // field does not exist, skip
				};

				out_fields[field_idx] = value;
			}
			(out_fields, fields_comptime_known)
		};

		// check for uninit field
		for (_, field) in r#struct
			.fields
			.iter()
			.enumerate()
			.filter(|(i, _)| fields.len() <= *i || fields[*i].is_none())
		{
			self.push_error(
				Diagnostic::error()
					.with_message(format!("field `{}` was not initialized", field.name))
					.with_label(Label::primary().with_span(self.diag_span(*span)))
					.with_note(format!("add `.{} = ...,`", field.name)),
			);
			analysis_failed = true;
		}

		if analysis_failed {
			Err(AnalyzeError::AnalysisFailed)
		} else {
			// OPTIM(zino)
			let value = if fields_comptime_known {
				let fields = fields
					.into_iter()
					.map(|field| self.try_resolve_comptime_value(field.as_ref().unwrap()).unwrap());
				let fields = self.cu.values.alloc_slice_fill_iter(fields);
				self.cu.values.intern_trivial(&value::Key::Aggregate { ty, values: fields }).into()
			} else {
				// TODO runtime only
				let fields = self.cu.values.alloc_slice_fill_iter(fields.into_iter().map(|field| field.unwrap()));
				self.inst(block, vtir::Opcode::StructInit { struct_ty: ty, fields })
			};
			self.vuir_map.insert(id, value);
			Ok((value, ControlFlow::May))
		}
	}

	fn analyze_array_init(
		&mut self,
		id: vuir::InstructionId,
		block: BlockId,
		ty: value::Index,
		elements: &[vuir::InstructionRef],
		span: &Span,
	) -> Result<(vtir::InstructionRef, ControlFlow), AnalyzeError> {
		let value = match *self.cu.values.index_to_key(ty) {
			value::Key::Type(value::Type::Slice(slice)) if slice.pointee_ty == self.cu.values.common.anyptr_t => {
				let mut coerced = BumpVec::with_capacity_in(elements.len(), self.instructions_payload_alloc);
				for element in elements {
					let value = self.resolve_inst(element);
					coerced.push(self.coerce(block, self.cu.values.common.anyptr_t, value, span)?);
				}
				self.inst(block, vtir::Opcode::SliceInit {
					slice_ty: ty,
					elements: coerced.into_bump_slice(),
				})
			},
			value::Key::Type(value::Type::Array(array)) => {
				if elements.len() as u64 != array.len {
					self.push_error(
						Diagnostic::error()
							.with_message(format!("expected {} array elements, found {}", array.len, elements.len()))
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}

				let mut coerced = BumpVec::with_capacity_in(elements.len(), self.instructions_payload_alloc);
				let mut comptime_values = Vec::with_capacity(elements.len());
				let mut all_comptime = true;
				for element in elements {
					let value = self.resolve_inst(element);
					let value = self.coerce(block, array.elem_ty, value, span)?;
					if let Some(value) = self.try_resolve_comptime_value(&value) {
						comptime_values.push(value);
					} else {
						all_comptime = false;
					}
					coerced.push(value);
				}
				if all_comptime {
					let values = self.cu.values.alloc_slice_fill_iter(comptime_values.into_iter());
					self.cu.values.intern_trivial(&value::Key::Aggregate { ty, values }).into()
				} else {
					self.inst(block, vtir::Opcode::ArrayInit {
						array_ty: ty,
						elements: coerced.into_bump_slice(),
					})
				}
			},
			_ => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"`{}` must be an array, or `[]anyptr`, to be initialized with array init syntax",
							self.cu.values.display_index(ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return Err(AnalyzeError::AnalysisFailed);
			},
		};

		self.vuir_map.insert(id, value);
		Ok((value, ControlFlow::May))
	}

	fn analyze_type_info(
		&mut self,
		ty: value::Index,
	) -> Result<TypeInfoId, AnalyzeError> {
		if let Some(index) = self.cu.type_to_type_info_id.find(&ty) {
			let type_info_id = self.cu.type_to_type_info_id.kv(index).1.load(std::sync::atomic::Ordering::Acquire);
			return Ok(type_info_id);
		}

		// work here may be duplicated with other workers, it be deduplicated at the end by the sharded index map
		// but we may still leak some memory to the bump allocator
		let mut reference_type = |this: &mut Self, referenced_ty| {
			let id = this.analyze_type_info(referenced_ty).unwrap();
			id.0 as u32
		};

		let type_info_ty = self.cu.builtin_type_info()?;
		let type_info_ty_struct = self.cu.values.index_to_value(type_info_ty).as_struct();
		let values = &self.cu.values;
		let ty_layout = match values.index_to_key(ty) {
			value::Key::Type(value::Type::Fn(_)) => values.type_ptr_layout(&self.cu.resolved_target, ty),
			_ => values.type_layout(&self.cu.resolved_target, ty),
		};
		let u8_ty = self
			.cu
			.values
			.intern_trivial(&value::Key::Type(value::Type::Int { signed: false, bits: 8 }));
		let str_slice_ty = self
			.cu
			.values
			.intern_trivial(&value::Key::Type(value::Type::Slice(TypeSlice { pointee_ty: u8_ty })));
		let builtin_type_namespace = type_info_ty_struct.namespace;
		let field_ty = {
			let decl = self
				.lookup_decl_in_namespace(builtin_type_namespace, Intern::from("Field"))
				.expect("internal compiler error: builtin.Type.Field declaration missing");
			self.cu.get_or_analyze_decl_value(decl)?.unwrap()
		};
		let variant_ty = {
			let decl = self
				.lookup_decl_in_namespace(builtin_type_namespace, Intern::from("Variant"))
				.expect("internal compiler error: builtin.Type.Variant declaration missing");
			self.cu.get_or_analyze_decl_value(decl)?.unwrap()
		};

		let value::Key::Type(ty_key) = values.index_to_key(ty) else {
			unreachable!("expected a type for builtin.Type RTTI")
		};

		let kind = {
			let kind_ty = type_info_ty_struct.fields[type_info_ty_struct.field_idx_by_name("kind").unwrap()].ty;
			let kind_ty_union = self.cu.values.index_to_value(kind_ty).as_union();
			let kind_tag_ty = {
				kind_ty_union
					.tag_ty
					.expect("internal compiler error: builtin.Type.Kind must be a tagged union")
			};

			match ty_key {
				// 0: int
				value::Type::Int { signed, bits } => {
					let payload_ty = kind_ty_union.fields[0].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 0);
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Bool(*signed)),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u16_t,
							value: Anyint::from(*bits).into(),
						}),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::Usize => {
					let payload_ty = kind_ty_union.fields[0].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 0);
					let bits: u16 = self.cu.resolved_target.ptr_width_in_bits.into();
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Bool(false)),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u16_t,
							value: Anyint::from(bits).into(),
						}),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::Isize => {
					let payload_ty = kind_ty_union.fields[0].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 0);
					let bits: u16 = self.cu.resolved_target.ptr_width_in_bits.into();
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Bool(true)),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u16_t,
							value: Anyint::from(bits).into(),
						}),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				// 1: float
				value::Type::F16 => {
					let payload_ty = kind_ty_union.fields[1].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 1);
					let payload = self.cu.values.alloc_slice(&[values.intern_trivial(&value::Key::Int {
						ty: values.common.u16_t,
						value: Anyint::from(16u16).into(),
					})]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::F32 => {
					let payload_ty = kind_ty_union.fields[1].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 1);
					let payload = self.cu.values.alloc_slice(&[values.intern_trivial(&value::Key::Int {
						ty: values.common.u16_t,
						value: Anyint::from(32u16).into(),
					})]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::F64 => {
					let payload_ty = kind_ty_union.fields[1].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 1);
					let payload = self.cu.values.alloc_slice(&[values.intern_trivial(&value::Key::Int {
						ty: values.common.u16_t,
						value: Anyint::from(64u16).into(),
					})]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::F128 => {
					let payload_ty = kind_ty_union.fields[1].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 1);
					let payload = self.cu.values.alloc_slice(&[values.intern_trivial(&value::Key::Int {
						ty: values.common.u16_t,
						value: Anyint::from(128u16).into(),
					})]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				// 2..9: tag-only variants
				value::Type::Bool => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 2)),
					payload: None,
				}),
				value::Type::Void => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 3)),
					payload: None,
				}),
				value::Type::Never => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 4)),
					payload: None,
				}),
				value::Type::Type => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 5)),
					payload: None,
				}),
				value::Type::Any => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 6)),
					payload: None,
				}),
				value::Type::Anyptr => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 7)),
					payload: None,
				}),
				value::Type::Anyint => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 8)),
					payload: None,
				}),
				value::Type::Anyfloat => values.intern_trivial(&value::Key::Union {
					ty: kind_ty,
					tag: Some(values.intern_enum_tag_from_field_idx(kind_tag_ty, 9)),
					payload: None,
				}),

				// 10: ptr
				value::Type::Ptr(ptr) => {
					let payload_ty = kind_ty_union.fields[10].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 10);
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(reference_type(self, ptr.pointee_ty)).into(),
						}),
						values.intern_trivial(&value::Key::Bool(ptr.is_const)),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				// 11: slice
				value::Type::Slice(slice) => {
					let payload_ty = kind_ty_union.fields[11].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 11);
					let payload = self.cu.values.alloc_slice(&[values.intern_trivial(&value::Key::Int {
						ty: values.common.u32_t,
						value: Anyint::from(reference_type(self, slice.pointee_ty)).into(),
					})]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				// 12: array
				value::Type::Array(array) => {
					let payload_ty = kind_ty_union.fields[12].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 12);
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(reference_type(self, array.elem_ty)).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.usize_t,
							value: Anyint::from(array.len).into(),
						}),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				// 13: fn
				value::Type::Fn(function) => {
					let payload_ty = kind_ty_union.fields[13].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 13);
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(reference_type(self, function.ret_ty)).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u16_t,
							value: Anyint::from(function.params.len() as u16).into(),
						}),
						values.intern_trivial(&value::Key::Bool(function.var_args)),
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::Struct(_) => {
					let payload_ty = kind_ty_union.fields[14].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 14);
					let value::Value::Struct(r#struct) = values.index_to_value(ty) else {
						unreachable!("struct type without struct value")
					};
					let r#struct = r#struct.as_ref();
					let field_values = {
						let mut cur_offset = 0u64;
						let mut vals = Vec::with_capacity(r#struct.fields.len());
						for (field_idx, field) in r#struct.fields.iter().enumerate() {
							let offset = match &r#struct.layout {
								StructLayout::Standard => {
									let field_layout = values.type_layout(&self.cu.resolved_target, field.ty);
									cur_offset = cur_offset.next_multiple_of(field_layout.align);
									let offset = cur_offset;
									cur_offset += field_layout.size;
									offset
								},
								StructLayout::Packed { .. } => u64::from(r#struct.get_packed_field_info(field_idx).unwrap().offset / 8),
							};
							let name = values.intern_trivial(&value::Key::Str {
								slice_ty: str_slice_ty,
								value: Intern::from(field.name.as_bytes()),
							});
							vals.push(values.intern_trivial(&value::Key::Aggregate {
								ty: field_ty,
								values: self.cu.values.alloc_slice(&[
									name,
									values.intern_trivial(&value::Key::Int {
										ty: values.common.u32_t,
										value: Anyint::from(reference_type(self, field.ty)).into(),
									}),
									values.intern_trivial(&value::Key::Int {
										ty: values.common.usize_t,
										value: Anyint::from(offset).into(),
									}),
								]),
							}));
						}
						self.cu.values.alloc_slice_fill_iter(vals.into_iter())
					};
					let fields_slice_ty = values.intern_trivial(&value::Key::Type(value::Type::Slice(TypeSlice { pointee_ty: field_ty })));
					let fields_array_ty = values.intern_trivial(&value::Key::Type(value::Type::Array(value::TypeArray {
						elem_ty: field_ty,
						len: field_values.len() as u64,
					})));
					let fields_array = values.intern_trivial(&value::Key::Aggregate {
						ty: fields_array_ty,
						values: field_values,
					});
					let namespace = self.cu.decls.with_mut(|decls| decls[self.owner_decl].namespace);
					let fields_decl = self.cu.decls.lock().push(Decl {
						name: format!("__vif_const_{}", fields_array.as_u32()).as_str().into(),
						module: self.module,
						namespace,
						analysis_state: DeclAnalysisState::Analysed { value: fields_array },
					});
					let fields_ptr_ty = values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
						pointee_ty: field_ty,
						packed: None,
						is_const: true,
					})));
					let fields_ptr = values.intern_trivial(&value::Key::Ptr(Ptr {
						ty: fields_ptr_ty,
						kind: PtrKind::Decl(fields_decl),
					}));
					let fields_len = values.intern_trivial(&value::Key::Int {
						ty: values.common.usize_t,
						value: Anyint::from(field_values.len()).into(),
					});
					let fields_slice = values.intern_trivial(&value::Key::Slice {
						ty: fields_slice_ty,
						ptr: fields_ptr,
						len: fields_len,
					});
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(r#struct.fields.len() as u32).into(),
						}),
						values.intern_trivial(&value::Key::Bool(r#struct.linear)),
						fields_slice,
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::Enum(_) => {
					let payload_ty = kind_ty_union.fields[15].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 15);
					let value::Value::Enum(r#enum) = values.index_to_value(ty) else {
						unreachable!("enum type without enum value")
					};
					let r#enum = r#enum.as_ref();
					let variant_values = self.cu.values.alloc_slice_fill_iter(r#enum.fields.iter().map(|field| {
						let name = values.intern_trivial(&value::Key::Str {
							slice_ty: str_slice_ty,
							value: Intern::from(field.name.as_bytes()),
						});
						let value = values.index_to_key(field.value).as_int().1;
						values.intern_trivial(&value::Key::Aggregate {
							ty: variant_ty,
							values: self.cu.values.alloc_slice(&[
								name,
								values.intern_trivial(&value::Key::Int {
									ty: values.common.usize_t,
									value: *value,
								}),
							]),
						})
					}));
					let variants_slice_ty =
						values.intern_trivial(&value::Key::Type(value::Type::Slice(TypeSlice { pointee_ty: variant_ty })));
					let variants_array_ty = values.intern_trivial(&value::Key::Type(value::Type::Array(value::TypeArray {
						elem_ty: variant_ty,
						len: variant_values.len() as u64,
					})));
					let variants_array = values.intern_trivial(&value::Key::Aggregate {
						ty: variants_array_ty,
						values: variant_values,
					});
					let namespace = self.cu.decls.with_mut(|decls| decls[self.owner_decl].namespace);
					let variants_decl = self.cu.decls.lock().push(Decl {
						name: format!("__vif_const_{}", variants_array.as_u32()).as_str().into(),
						module: self.module,
						namespace,
						analysis_state: DeclAnalysisState::Analysed { value: variants_array },
					});
					let variants_ptr_ty = values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
						pointee_ty: variant_ty,
						packed: None,
						is_const: true,
					})));
					let variants_ptr = values.intern_trivial(&value::Key::Ptr(Ptr {
						ty: variants_ptr_ty,
						kind: PtrKind::Decl(variants_decl),
					}));
					let variants_len = values.intern_trivial(&value::Key::Int {
						ty: values.common.usize_t,
						value: Anyint::from(variant_values.len()).into(),
					});
					let variants_slice = values.intern_trivial(&value::Key::Slice {
						ty: variants_slice_ty,
						ptr: variants_ptr,
						len: variants_len,
					});
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(reference_type(self, r#enum.tag_ty)).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(r#enum.fields.len() as u32).into(),
						}),
						values.intern_trivial(&value::Key::Bool(r#enum.linear)),
						variants_slice,
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},
				value::Type::Union(_) => {
					let payload_ty = kind_ty_union.fields[16].ty.unwrap();
					let tag = values.intern_enum_tag_from_field_idx(kind_tag_ty, 16);
					let value::Value::Union(r#union) = values.index_to_value(ty) else {
						unreachable!("union type without union value")
					};
					let r#union = r#union.as_ref();
					let union_layout = values.type_union_layout(&self.cu.resolved_target, ty);
					let (tag_offset, payload_offset) = if union_layout.tag.size == 0 || union_layout.payload.size == 0 {
						(0u64, 0u64)
					} else if union_layout.tag.align >= union_layout.payload.align {
						(0u64, union_layout.tag.size.next_multiple_of(union_layout.payload.align))
					} else {
						(union_layout.payload.size.next_multiple_of(union_layout.tag.align), 0u64)
					};
					let field_values = self.cu.values.alloc_slice_fill_iter(r#union.fields.iter().map(|field| {
						let name = values.intern_trivial(&value::Key::Str {
							slice_ty: str_slice_ty,
							value: Intern::from(field.name.as_bytes()),
						});
						let field_ty_id = field.ty.unwrap_or(values.common.void_t);
						let offset = if field.ty.is_some() { payload_offset } else { 0 };
						values.intern_trivial(&value::Key::Aggregate {
							ty: field_ty,
							values: self.cu.values.alloc_slice(&[
								name,
								values.intern_trivial(&value::Key::Int {
									ty: values.common.u32_t,
									value: Anyint::from(reference_type(self, field_ty_id)).into(),
								}),
								values.intern_trivial(&value::Key::Int {
									ty: values.common.usize_t,
									value: Anyint::from(offset).into(),
								}),
							]),
						})
					}));
					let fields_slice_ty = values.intern_trivial(&value::Key::Type(value::Type::Slice(TypeSlice { pointee_ty: field_ty })));
					let fields_array_ty = values.intern_trivial(&value::Key::Type(value::Type::Array(value::TypeArray {
						elem_ty: field_ty,
						len: field_values.len() as u64,
					})));
					let fields_array = values.intern_trivial(&value::Key::Aggregate {
						ty: fields_array_ty,
						values: field_values,
					});
					let namespace = self.cu.decls.with_mut(|decls| decls[self.owner_decl].namespace);
					let fields_decl = self.cu.decls.lock().push(Decl {
						name: format!("__vif_const_{}", fields_array.as_u32()).as_str().into(),
						module: self.module,
						namespace,
						analysis_state: DeclAnalysisState::Analysed { value: fields_array },
					});
					let fields_ptr_ty = values.intern_trivial(&value::Key::Type(value::Type::Ptr(TypePtr {
						pointee_ty: field_ty,
						packed: None,
						is_const: true,
					})));
					let fields_ptr = values.intern_trivial(&value::Key::Ptr(Ptr {
						ty: fields_ptr_ty,
						kind: PtrKind::Decl(fields_decl),
					}));
					let fields_len = values.intern_trivial(&value::Key::Int {
						ty: values.common.usize_t,
						value: Anyint::from(field_values.len()).into(),
					});
					let fields_slice = values.intern_trivial(&value::Key::Slice {
						ty: fields_slice_ty,
						ptr: fields_ptr,
						len: fields_len,
					});
					let payload = self.cu.values.alloc_slice(&[
						values.intern_trivial(&value::Key::Bool(r#union.tag_ty.is_some())),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(reference_type(self, r#union.tag_ty.unwrap_or(values.common.void_t))).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.usize_t,
							value: Anyint::from(tag_offset).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.usize_t,
							value: Anyint::from(payload_offset).into(),
						}),
						values.intern_trivial(&value::Key::Int {
							ty: values.common.u32_t,
							value: Anyint::from(r#union.fields.len() as u32).into(),
						}),
						values.intern_trivial(&value::Key::Bool(r#union.linear)),
						fields_slice,
					]);
					let val = values.intern_trivial(&value::Key::Aggregate {
						ty: payload_ty,
						values: payload,
					});
					values.intern_trivial(&value::Key::Union {
						ty: kind_ty,
						tag: Some(tag),
						payload: Some(val),
					})
				},

				value::Type::NullPtr | value::Type::GenericPoison | value::Type::EnumLiteral => {
					unreachable!()
				},
			}
		};

		// TODO(zino): --rtti-strip-names
		let name = format!("{}", values.display_index(ty));

		// now insert to the type infos
		let index = self.cu.type_to_type_info_id.entry(&ty).or_insert_with(|| {
			let type_info_id = self.cu.type_info_entries.push(value::Index::NONE);
			let type_info = self.cu.values.alloc_slice(&[
				// id
				values.intern_trivial(&value::Key::Int {
					ty: values.common.u32_t,
					value: Anyint::from(type_info_id.0 as u32).into(),
				}),
				// name
				values.intern_trivial(&value::Key::Str {
					slice_ty: str_slice_ty,
					value: Intern::from(name.as_bytes()),
				}),
				// size
				values.intern_trivial(&value::Key::Int {
					ty: values.common.usize_t,
					value: Anyint::from(ty_layout.size).into(),
				}),
				// kind
				kind,
			]);
			let type_info = values.intern_trivial(&value::Key::Aggregate {
				ty: type_info_ty,
				values: type_info,
			});
			unsafe {
				self.cu.type_info_entries.replace(type_info_id, type_info);
			}
			type_info_id
		});

		Ok(self.cu.type_to_type_info_id.kv(index).1.load(std::sync::atomic::Ordering::Relaxed))
	}

	#[must_use = "coerce return the coerced instruction / value"]
	#[track_caller]
	fn coerce(
		&mut self,
		block: BlockId,
		dst_ty: value::Index,
		inst: vtir::InstructionRef,
		span: &Span,
	) -> Result<vtir::InstructionRef, AnalyzeError> {
		let inst_ty = if inst.is_interned() && self.cu.values.index_to_key(inst.as_interned()).is_type() {
			inst.as_interned()
		} else {
			self.type_of(&inst)
		};

		// either inst is already of the right type or dst_ty is generic poison, which we can never coerce into
		if self.cu.values.type_contains_generic_poison(dst_ty) || dst_ty == inst_ty {
			return Ok(inst);
		}

		// Not equals, can we coerce both types ?
		let result: Result<vtir::InstructionRef, String> = 'msg: {
			match (self.cu.values.index_to_key(inst_ty), self.cu.values.index_to_key(dst_ty)) {
				(_, value::Key::Type(value::Type::Type)) if self.cu.values.index_to_key(inst_ty).is_type() => Ok(inst),
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Anyptr)) if !self.blocks[block].comptime => {
					let concrete = self.coerce(block, self.cu.values.common.i32_t, inst, span)?;
					let _ = self.analyze_type_info(self.cu.values.common.i32_t)?;
					Ok(self.inst(block, vtir::Opcode::AnyptrInit {
						value: concrete,
						value_ty: self.cu.values.common.i32_t,
					}))
				},
				(_, value::Key::Type(value::Type::Anyptr)) if !self.cu.values.type_is_comptime_only(inst_ty) => {
					let _ = self.analyze_type_info(inst_ty)?;
					Ok(self.inst(block, vtir::Opcode::AnyptrInit {
						value: inst,
						value_ty: inst_ty,
					}))
				},
				(_, value::Key::Type(value::Type::Anyptr)) => Err(format!(
					"cannot coerce comptime-only value of type `{}` to runtime `anyptr`",
					self.cu.values.display_index(inst_ty)
				)),
				(_, value::Key::Type(value::Type::Any)) if self.blocks[block].comptime => Ok(inst),

				// Integer types conversions
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Usize) | value::Key::Type(value::Type::Isize)) => {
					let value::Key::Int { value, .. } = self.cu.values.index_to_key(inst.as_interned()) else {
						unreachable!("encountered a comptime_int that is not a int");
					};

					// TODO(zino): query usize & isize true sizes...

					let value = value::Key::Int { ty: dst_ty, value: *value };
					Ok(vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value)))
				},
				(value::Key::Type(value::Type::Anyint), value::Key::Type(value::Type::Int { signed, bits })) => {
					if let Some(_value) = self.try_resolve_comptime_value(&inst) {
						let value::Key::Int { value, .. } = self.cu.values.index_to_key(inst.as_interned()) else {
							unreachable!("encountered a comptime_int that is not a int");
						};

						// ensure we can fit the literal in the type
						let fits = if *signed {
							value.fits_signed_bits(*bits)
						} else {
							value.fits_unsigned_bits(*bits)
						};
						if !fits {
							break 'msg Err(format!(
								"coercion failed: constant {value} is too large to fit in a {}",
								self.cu.values.display_index(dst_ty)
							));
						}

						// coerce ok
						let value = value::Key::Int { ty: dst_ty, value: *value };
						Ok(InstructionRef::Interned(self.cu.values.intern_trivial(&value)))
					} else {
						if dst_ty == self.cu.values.common.anyint_t {
							break 'msg Err("cannot coerce a comptime_int that does not have a comptime known value".to_string());
						}

						// runtime value
						Ok(self.inst(block, vtir::Opcode::UnsafeIntCast { src: inst, dst_ty }))
					}
				},
				(
					value::Key::Type(value::Type::Anyfloat),
					value::Key::Type(value::Type::F16)
					| value::Key::Type(value::Type::F32)
					| value::Key::Type(value::Type::F64)
					| value::Key::Type(value::Type::F128),
				) => {
					let value::Key::Float { value, .. } = self.cu.values.index_to_key(inst.as_interned()) else {
						unreachable!("encountered a comptile_float that is not a float");
					};

					let (dst_ty_max, converted_val) = match *self.cu.values.index_to_key(dst_ty) {
						value::Key::Type(value::Type::F16) => (f16::MAX as f128, value.0 as f16 as f128),
						value::Key::Type(value::Type::F32) => (f32::MAX as f128, value.0 as f32 as f128),
						value::Key::Type(value::Type::F64) => (f64::MAX as f128, value.0 as f64 as f128),
						value::Key::Type(value::Type::F128) => (f128::MAX, value.0),
						_ => unreachable!(),
					};

					// check if its fit
					if **value > dst_ty_max {
						break 'msg Err(format!(
							"coercion failed: constant {value} is too large to fit in a {}",
							self.cu.values.display_index(dst_ty)
						));
					}

					// check for precision loss
					if **value != converted_val {
						break 'msg Err(format!(
							"coercion failed: constant {value} loses precision when coerced to {}, use a explicit cast or suffix",
							self.cu.values.display_index(dst_ty)
						));
					}

					// coerce ok
					let value = value::Key::Float { ty: dst_ty, value: *value };
					Ok(vtir::InstructionRef::Interned(self.cu.values.intern_trivial(&value)))
				},

				// Pointer conversions
				(value::Key::Type(value::Type::Ptr(ptr_src)), value::Key::Type(value::Type::Ptr(ptr_dst))) => {
					// only same pointee ty
					if ptr_src.pointee_ty != ptr_dst.pointee_ty {
						break 'msg Err(format!(
							"coercion failed: cannot coerce pointee type from `{}` to `{}`",
							self.cu.values.display_index(ptr_src.pointee_ty),
							self.cu.values.display_index(ptr_dst.pointee_ty)
						));
					}

					// we only allow mut => const ptr coercions
					if ptr_src.is_const && !ptr_dst.is_const {
						break 'msg Err("coercion failed: cannot coerce from const pointer to mutable pointer".to_string());
					}

					let inst = self.inst(block, vtir::Opcode::BitCast { src: inst, dst_ty });
					Ok(inst)
				},

				// enum literal to enum
				(value::Key::Type(value::Type::EnumLiteral), value::Key::Type(value::Type::Enum(..))) => {
					let value::Key::EnumLiteral(enum_lit) = self.cu.values.index_to_key(inst.as_interned()) else {
						unreachable!();
					};
					let dst_enum_ty = self.cu.values.index_to_value(dst_ty).as_enum();
					let field = self.analyze_field_val(block, vtir::InstructionRef::Interned(dst_ty), enum_lit, span, span)?;
					Ok(field)
				},

				_ => Err(format!(
					"expected type `{}`, found `{}`",
					self.cu.values.display_index(dst_ty),
					self.cu.values.display_index(inst_ty)
				)),
			}
		};

		match result {
			Ok(coerced_inst) => Ok(coerced_inst),
			Err(error_msg) => {
				self.push_error(
					Diagnostic::error()
						.with_message(error_msg)
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				Err(AnalyzeError::AnalysisFailed)
			},
		}
	}

	/// Used to temporary execute some semantic analysis with a different vuir map, used for e.g function calls
	fn with_different_vuir<R>(
		&mut self,
		other: &Vuir,
		other_mod: ModuleId,
		f: impl FnOnce(&mut Self) -> R,
	) -> R {
		// SAFETY: lifetime is okay we will replace with the old vuir at the end of the scope
		let old_vuir = std::mem::replace(&mut self.vuir, unsafe { std::mem::transmute::<&'_ Vuir, &'a Vuir>(other) });
		let old_module_id = std::mem::replace(&mut self.module, other_mod);
		let result = f(self);
		std::mem::replace(&mut self.vuir, old_vuir);
		std::mem::replace(&mut self.module, old_module_id);
		result
	}

	pub fn with_different_block_namespace<R>(
		&mut self,
		block: BlockId,
		namespace: NamespaceId,
		f: impl FnOnce(&mut Self) -> R,
	) -> R {
		let old_namespace = std::mem::replace(&mut self.blocks[block].namespace, namespace);
		let result = f(self);
		self.blocks[block].namespace = old_namespace;
		result
	}

	/// Get the type of the value this VTIR instruction returns
	fn type_of(
		&self,
		inst: &vtir::InstructionRef,
	) -> value::Index {
		vtir::type_of(&self.cu.values, &self.instructions, inst)
	}

	/// Returns the exact bit-size for types where it is statically known
	fn known_bit_size(
		&self,
		ty: value::Index,
	) -> Option<u64> {
		let (value::Key::Type(ty), value) = self.cu.values.index_to_key_value(ty) else {
			return None;
		};
		match ty {
			value::Type::Int { bits, .. } => Some(*bits as u64),
			value::Type::Bool => Some(1),
			value::Type::F16 => Some(16),
			value::Type::F32 => Some(32),
			value::Type::F64 => Some(64),
			value::Type::F128 => Some(128),
			value::Type::Void => Some(0),
			value::Type::Struct(_) => {
				let value::Value::Struct(r#struct) = value else {
					unreachable!("struct type without struct value")
				};
				if let StructLayout::Packed { fields_bits, .. } = r#struct.as_ref().layout {
					Some(fields_bits as u64)
				} else {
					None
				}
			},
			value::Type::Enum(_) => {
				let value::Value::Enum(r#enum) = value else {
					unreachable!("enum type without enum value")
				};
				self.known_bit_size(r#enum.tag_ty)
			},
			value::Type::Anyint
			| value::Type::Anyfloat
			| value::Type::Usize
			| value::Type::Isize
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
			| value::Type::EnumLiteral => None,
		}
	}

	/// Check if a switch without an else branch is exhaustive.
	/// Returns true if exhaustive, false if not (and pushes an error diagnostic).
	fn check_switch_exhaustive(
		&mut self,
		operand_ty: value::Index,
		cases: &[vtir::SwitchCase],
		span: &Span,
	) -> bool {
		let (value::Key::Type(ty), value) = self.cu.values.index_to_key_value(operand_ty) else {
			unreachable!("switch operand type is not a type")
		};
		let (tag_fields, union_fields) = match ty {
			value::Type::Enum(_) => {
				let value::Value::Enum(e) = value else {
					unreachable!("enum type without enum value")
				};
				(e.fields, None)
			},
			value::Type::Union(_) => {
				let value::Value::Union(u) = value else {
					unreachable!("union type without union value")
				};
				let union_ty = u.as_ref();
				let Some(tag_ty) = union_ty.tag_ty else {
					self.push_error(
						Diagnostic::error()
							.with_message("cannot exhaustively switch on a bare union")
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					return false;
				};
				let (value::Key::Type(value::Type::Enum(_)), value::Value::Enum(tag_enum)) = self.cu.values.index_to_key_value(tag_ty)
				else {
					unreachable!("tagged union tag type must be an enum");
				};
				assert_eq!(union_ty.fields.len(), tag_enum.fields.len());
				(tag_enum.fields, Some(union_ty.fields))
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
			| value::Type::Struct(_)
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
			| value::Type::EnumLiteral => {
				self.push_error(
					Diagnostic::error()
						.with_message("switch must have an else branch")
						.with_label(Label::primary().with_span(self.diag_span(*span))),
				);
				return false;
			},
		};

		let mut covered = FxHashSet::default();
		for item in cases.iter().flat_map(|case| case.items) {
			let value = self
				.try_resolve_comptime_value(item)
				.expect("switch patterns must be comptime-known after coercion");
			let tag = match self.cu.values.index_to_key(value) {
				value::Key::EnumTag { val, .. } => *val,
				_ => value,
			};
			covered.insert(tag);
		}

		let mut missing_count = 0;
		let mut variants = String::new();
		for (field_idx, tag_field) in tag_fields.iter().enumerate() {
			if covered.contains(&tag_field.value) {
				continue;
			}
			if missing_count != 0 {
				variants.push_str("`, `");
			}
			let name = union_fields.map_or(tag_field.name, |fields| fields[field_idx].name);
			variants.push_str(&name);
			missing_count += 1;
		}

		if missing_count == 0 {
			return true;
		}

		self.push_error(
			Diagnostic::error()
				.with_message(format!(
					"switch is not exhaustive, missing variant{} `{variants}`",
					if missing_count > 1 { "s" } else { "" },
				))
				.with_label(Label::primary().with_span(self.diag_span(*span))),
		);
		false
	}

	fn resolve_type(
		&mut self,
		block: BlockId,
		inst: &vuir::InstructionRef,
		span: &Span,
	) -> Result<value::Index, AnalyzeError> {
		match inst {
			vuir::InstructionRef::Interned(i) => Ok(*i),
			vuir::InstructionRef::Instruction(id) => {
				let opcode = self.resolve_inst(inst);

				// we try to coerce to the type `type` to obtain the underlying type
				let coerced_inst = self.coerce(block, self.cu.values.common.type_t, opcode, span)?;
				let coerced_inst = self.try_resolve_comptime_value(&coerced_inst).ok_or_else(|| {
					self.push_error(
						Diagnostic::error()
							.with_message("type must be a comptime known value")
							.with_label(Label::primary().with_span(self.diag_span(*span))),
					);
					AnalyzeError::AnalysisFailed
				})?;
				Ok(coerced_inst)
			},
		}
	}

	#[inline]
	fn resolve_inst(
		&self,
		r: &vuir::InstructionRef,
	) -> vtir::InstructionRef {
		match r {
			vuir::InstructionRef::Instruction(id) => *self.vuir_map.get(id).unwrap_or_else(|| {
				panic!(
					"Instruction {} ({:?}) not visited/lowered yet! : {:?}",
					id, self.vuir.instructions[id], self.vuir_map
				)
			}),
			vuir::InstructionRef::Interned(i) => vtir::InstructionRef::Interned(*i),
		}
	}

	#[inline]
	fn try_resolve_comptime_value(
		&self,
		inst: &vtir::InstructionRef,
	) -> Option<value::Index> {
		match inst {
			vtir::InstructionRef::Interned(i) => Some(*i),
			vtir::InstructionRef::Instruction(_) => None,
		}
	}

	/// Promotes a variadic argument to the appropriate C ABI type.
	///
	/// C varargs require specific type promotions:
	/// - bool and integers smaller than 32 bits → i32
	/// - f16, f32 → f64
	/// - all other runtime integers and pointers → unchanged
	///
	/// Returns the promoted instruction and its type, or an error if the type
	/// cannot be used as a variadic argument.
	fn promote_variadic_arg(
		&mut self,
		block: BlockId,
		resolved_value: vtir::InstructionRef,
		arg_ty: value::Index,
		span: Span,
	) -> Result<(vtir::InstructionRef, value::Index), AnalyzeError> {
		let value::Key::Type(ty) = self.cu.values.index_to_key(arg_ty) else {
			unreachable!("variadic argument type is not a type")
		};
		match ty {
			value::Type::F64 | value::Type::Ptr(_) | value::Type::Isize | value::Type::Usize | value::Type::Int { bits: 32..=64, .. } => {
				Ok((resolved_value, arg_ty))
			},

			value::Type::Bool => {
				let dst_ty = self.cu.values.common.i32_t;
				let promoted = if let Some(value) = self.try_resolve_comptime_value(&resolved_value) {
					let value::Key::Bool(value) = self.cu.values.index_to_key(value) else {
						unreachable!("bool comptime value must be Bool");
					};
					InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int {
						ty: dst_ty,
						value: Anyint::from(u8::from(*value)).into(),
					}))
				} else {
					self.inst(block, vtir::Opcode::UnsafeIntCast {
						src: resolved_value,
						dst_ty,
					})
				};
				let promoted_ty = self.type_of(&promoted);
				Ok((promoted, promoted_ty))
			},

			value::Type::Int { bits, .. } => {
				if *bits > 64 {
					self.push_error(
						Diagnostic::error()
							.with_message("invalid variadic parameter type, expected i1-64 or u1-64")
							.with_label(Label::primary().with_span(self.diag_span(span))),
					);
					return Err(AnalyzeError::AnalysisFailed);
				}
				let dst_ty = self.cu.values.common.i32_t;
				let promoted = if let Some(value) = self.try_resolve_comptime_value(&resolved_value) {
					let value::Key::Int { value, .. } = self.cu.values.index_to_key(value) else {
						unreachable!("int comptime value must be Int");
					};
					InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Int { ty: dst_ty, value: *value }))
				} else {
					self.inst(block, vtir::Opcode::UnsafeIntCast {
						src: resolved_value,
						dst_ty,
					})
				};
				let promoted_ty = self.type_of(&promoted);
				Ok((promoted, promoted_ty))
			},

			// f16, f32 → f64
			value::Type::F16 | value::Type::F32 => {
				let promoted = if let Some(value) = self.try_resolve_comptime_value(&resolved_value) {
					let value::Key::Float { value, .. } = self.cu.values.index_to_key(value) else {
						unreachable!("float comptime value must be Float");
					};
					InstructionRef::Interned(self.cu.values.intern_trivial(&value::Key::Float {
						ty: self.cu.values.common.f64_t,
						value: *value,
					}))
				} else {
					self.inst(block, vtir::Opcode::UnsafeFloatCast {
						src: resolved_value,
						dst_ty: self.cu.values.common.f64_t,
					})
				};
				let promoted_ty = self.type_of(&promoted);
				Ok((promoted, promoted_ty))
			},

			// Unsupported types
			value::Type::Anyint
			| value::Type::Anyfloat
			| value::Type::F128
			| value::Type::Void
			| value::Type::Struct(_)
			| value::Type::Enum(_)
			| value::Type::Union(_)
			| value::Type::Fn(_)
			| value::Type::Slice(_)
			| value::Type::Array(_)
			| value::Type::NullPtr
			| value::Type::Any
			| value::Type::Anyptr
			| value::Type::GenericPoison
			| value::Type::Type
			| value::Type::Never
			| value::Type::EnumLiteral => {
				self.push_error(
					Diagnostic::error()
						.with_message(format!(
							"invalid variadic parameter type, expected usize, isize, i1-64, u1-64, f16, f32, f64, str or bool, got `{}`",
							self.cu.values.display_index(arg_ty)
						))
						.with_label(Label::primary().with_span(self.diag_span(span))),
				);
				Err(AnalyzeError::AnalysisFailed)
			},
		}
	}

	/// If the provided allocation is a potential comptime alloc (as confirmed by being present in the hashmap)
	/// register a store to it so we can replay them later
	#[inline]
	fn link_store_to_potential_comptime_alloc(
		&mut self,
		alloc_inst: &vtir::InstructionRef,
		store_inst: &vtir::InstructionRef,
		span: &Span,
	) {
		if let Some(potential_alloc) = self.allocs.potential_comptime_allocs.get_mut(alloc_inst) {
			potential_alloc.stores.push((*store_inst, *span));
		}
	}

	#[inline]
	fn make_type_name(
		&mut self,
		block: BlockId,
		id: vuir::InstructionId,
		naming: vuir::NamingKind,
	) -> Intern<str> {
		match naming {
			vuir::NamingKind::FromDecl => self.cu.decls.with_mut(|decls| decls[self.owner_decl].name),
			vuir::NamingKind::Anonymous => {
				let base = self.blocks[block].base_type_name;
				let name = format!("{base}_{id}");
				Intern::from(name.as_str())
			},
			vuir::NamingKind::FromPreviousStackAlloc => match &self.vuir.instructions[id - 1] {
				vuir::Opcode::StackAlloc { name, .. }
				| vuir::Opcode::StackAllocMut { name, .. }
				| vuir::Opcode::StackAllocComptime { name, .. }
				| vuir::Opcode::StackAllocComptimeMut { name, .. }
				| vuir::Opcode::StackAllocInferred { name, .. }
				| vuir::Opcode::StackAllocInferredMut { name, .. }
				| vuir::Opcode::StackAllocInferredComptime { name, .. }
				| vuir::Opcode::StackAllocInferredComptimeMut { name, .. } => name.symbol,
				_ => unreachable!("{:?}", self.vuir.instructions[id - 1]),
			},
			vuir::NamingKind::Named(name) => name,
		}
	}

	#[track_caller]
	fn ensure_type_exist_in_runtime(
		&mut self,
		ty: value::Index,
		span: &Span,
	) -> Result<(), AnalyzeError> {
		if self.cu.values.type_is_comptime_only(ty) {
			self.push_error(
				Diagnostic::error()
					.with_message(format!(
						"type `{}` cannot be used in runtime code, it is or contains a comptime-only type",
						self.cu.values.display_index(ty)
					))
					.with_label(Label::primary().with_span(self.diag_span(*span)))
					.with_note("try using `const`, giving an explicit type for `anyxxx` types or removing comptime fields"),
			);
			Err(AnalyzeError::AnalysisFailed)
		} else {
			Ok(())
		}
	}

	#[inline]
	fn push_error(
		&mut self,
		diagnostic: Diagnostic,
	) {
		self.cu.sema_errors.with_mut(|errors| {
			let errors = errors.entry(self.module).or_default();
			errors.push(diagnostic);
		});
	}

	// Diagnostics
	#[track_caller]
	fn diag_field_not_found(
		&mut self,
		field: &str,
		ty: value::Index,
		span: &Span,
	) {
		self.push_error(
			Diagnostic::error()
				.with_message(format!(
					"field `{}` does not exist in type `{}`",
					field,
					self.cu.values.display_index(ty),
				))
				.with_label(Label::primary().with_span(self.diag_span(*span))),
		);
	}

	#[track_caller]
	#[inline]
	fn diag_decl_not_found(
		&mut self,
		decl: &Intern<str>,
		ty: value::Index,
		span: &Span,
	) {
		self.push_error(
			Diagnostic::error()
				.with_message(format!(
					"declaration `{}` does not exist in type `{}`",
					decl,
					self.cu.values.display_index(ty),
				))
				.with_label(Label::primary().with_span(self.diag_span(*span))),
		);
	}

	#[track_caller]
	#[inline]
	fn diag_expected_type(
		&mut self,
		expected: value::Index,
		found: value::Index,
		span: DiagSpan,
	) {
		self.push_error(
			Diagnostic::error()
				.with_message(format!(
					"expected type `{}`, found `{}`",
					self.cu.values.display_index(expected),
					self.cu.values.display_index(found)
				))
				.with_label(Label::primary().with_span(span)),
		);
	}
}
