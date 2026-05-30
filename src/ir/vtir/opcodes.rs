use std::fmt::Debug;

use internment::Intern;

use crate::{
	common::IndexVec,
	ir::vtir,
	value::{
		self,
		ValueStore,
	},
};

#[derive(Debug)]
pub enum Opcode {
	Invalid,
	Noop,

	Block {
		instructions: &'static [vtir::InstructionRef],
		ret_ty: value::Index,
	},
	Break {
		block: vtir::InstructionId,
		value: vtir::InstructionRef,
	},
	Loop {
		instructions: &'static [vtir::InstructionRef],
		ret_ty: value::Index,
	},
	Repeat {
		r#loop: vtir::InstructionId,
	},

	FnParam {
		name: Intern<str>,
		ty: value::Index,
	},

	StackAlloc {
		ty: value::Index,
	},

	/// Invalid placeholder instruction for inferred stack allocation
	/// It is replaced at uir2tir time by a StackAlloc instruction with the proper type
	StackAllocInferred {
		is_comptime: bool,
	},
	Load {
		ptr: vtir::InstructionRef,
	},
	Store {
		src: vtir::InstructionRef,
		dst: vtir::InstructionRef,
	},
	Return {
		value: Option<vtir::InstructionRef>,
	},
	FnCall {
		callee: vtir::InstructionRef,
		params: Vec<vtir::InstructionRef>,
	},
	// structs
	/// Initialize a structure, returning its value
	StructInit {
		struct_ty: value::Index,
		fields: &'static [vtir::InstructionRef],
	},
	ArrayInit {
		array_ty: value::Index,
		elements: &'static [vtir::InstructionRef],
	},
	SliceInit {
		slice_ty: value::Index,
		elements: &'static [vtir::InstructionRef],
	},
	AnyptrInit {
		value: vtir::InstructionRef,
		value_ty: value::Index,
	},
	/// Obtain the value of a struct field
	StructFieldValue {
		struct_ty: vtir::InstructionRef,
		field_idx: usize,
		ret_ty: value::Index,
	},
	/// Obtain a pointer to a struct field
	StructFieldPtr {
		struct_ptr: vtir::InstructionRef,
		field_idx: usize,
		ret_ty: value::Index,
	},

	// unions
	/// Initialize a tagged union with a specific field
	UnionInit {
		union_ty: value::Index,
		field_idx: usize,
		/// The payload value, None for void fields
		value: Option<vtir::InstructionRef>,
	},
	/// Extract the tag from a tagged union value
	UnionTag {
		union_val: vtir::InstructionRef,
		tag_ty: value::Index,
	},
	/// Extract the payload from a union, bitcast to the given field type
	UnionFieldValue {
		union_val: vtir::InstructionRef,
		field_idx: usize,
		ret_ty: value::Index,
	},

	// Unary
	BoolNot {
		op: vtir::InstructionRef,
	},

	// arithmetic
	Add {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	AddSat {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Sub {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	SubSat {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Mul {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	MulSat {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Div {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Rem {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Lt {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Lte {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Gt {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Gte {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BoolAnd {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BoolOr {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Eq {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Neq {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},

	// bitwise
	Shl {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	ShlSat {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	ShlWrap {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	Shr {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	ShrSat {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	ShrWrap {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BitAnd {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BitOr {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BitXor {
		lhs: vtir::InstructionRef,
		rhs: vtir::InstructionRef,
	},
	BitNot {
		op: vtir::InstructionRef,
	},

	// builtins
	UnsafeIntCast {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	UnsafeFloatCast {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	IntToFloat {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	SizeOf {
		// TODO(ldubos): maybe store [`value::Index`] directly instead of an instruction ref, since
		// it must always be an interned type?
		of: vtir::InstructionRef,
	},
	Zeroed {
		ty: value::Index,
	},
	Undefined {
		ty: value::Index,
	},
	BitCast {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	AnyptrIs {
		value: vtir::InstructionRef,
		target_ty: value::Index,
	},
	AnyptrAs {
		value: vtir::InstructionRef,
		target_ty: value::Index,
	},
	SliceFromRawParts {
		slice_ty: value::Index,
		ptr: vtir::InstructionRef,
		len: vtir::InstructionRef,
	},
	SlicePtr {
		slice: vtir::InstructionRef,
		ptr_ty: value::Index,
	},
	SliceLen {
		slice: vtir::InstructionRef,
	},
	PtrToInt {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	IntToPtr {
		src: vtir::InstructionRef,
		dst_ty: value::Index,
	},
	SliceCopyNonoverlapping {
		slice_ty: value::Index,
		src: vtir::InstructionRef,
		dst: vtir::InstructionRef,
	},
	SliceElemPtr {
		slice: vtir::InstructionRef,
		index: vtir::InstructionRef,
		elem_ptr_ty: value::Index,
	},
	PtrElemPtr {
		array_ptr: vtir::InstructionRef,
		index: vtir::InstructionRef,
		elem_ptr_ty: value::Index,
	},

	DbgSrcLoc {
		line: usize,
		col: usize,
	},

	Branch {
		cond: vtir::InstructionRef,
		then_body: &'static [vtir::InstructionRef],
		else_body: &'static [vtir::InstructionRef],
	},
	Switch {
		operand: vtir::InstructionRef,
		cases: &'static [SwitchCase],
		else_body: &'static [vtir::InstructionRef],
	},

	Abort,
	Unreachable,
}

#[derive(Debug)]
pub struct SwitchCase {
	pub items: &'static [vtir::InstructionRef],
	pub body: &'static [vtir::InstructionRef],
}

#[inline(always)]
pub fn type_of(
	values: &ValueStore,
	instructions: &IndexVec<vtir::InstructionId, Opcode>,
	inst: &vtir::InstructionRef,
) -> value::Index {
	match inst {
		vtir::InstructionRef::Interned(i) => values.type_of_interned(*i),
		vtir::InstructionRef::Instruction(id) => {
			let opcode = &instructions[*id];
			match opcode {
				Opcode::Invalid => unreachable!(),
				Opcode::Block { ret_ty, .. } | Opcode::Loop { ret_ty, .. } => *ret_ty,
				Opcode::FnCall { callee, .. } => values.index_to_key(type_of(values, instructions, callee)).as_type_fn().ret_ty,
				Opcode::FnParam { ty, .. } => *ty,
				Opcode::StackAlloc { ty } => *ty,
				Opcode::Load { ptr } => {
					let ptr_ty = type_of(values, instructions, ptr);
					values.index_to_key(ptr_ty).as_type_ptr().pointee_ty
				},
				Opcode::Branch { .. } | Opcode::Switch { .. } => values.common.void_t,

				// unary
				Opcode::BoolNot { .. } => values.common.bool_t,

				// arithmetics
				Opcode::Add { lhs, .. }
				| Opcode::AddSat { lhs, .. }
				| Opcode::Sub { lhs, .. }
				| Opcode::SubSat { lhs, .. }
				| Opcode::Mul { lhs, .. }
				| Opcode::MulSat { lhs, .. }
				| Opcode::Div { lhs, .. }
				| Opcode::Rem { lhs, .. } => type_of(values, instructions, lhs),

				Opcode::Lt { .. }
				| Opcode::Lte { .. }
				| Opcode::Gt { .. }
				| Opcode::Gte { .. }
				| Opcode::BoolAnd { .. }
				| Opcode::BoolOr { .. }
				| Opcode::Eq { .. }
				| Opcode::Neq { .. } => values.common.bool_t,

				// Bitwise ops keep the lhs type.
				Opcode::Shl { lhs, .. }
				| Opcode::ShlSat { lhs, .. }
				| Opcode::ShlWrap { lhs, .. }
				| Opcode::Shr { lhs, .. }
				| Opcode::ShrSat { lhs, .. }
				| Opcode::ShrWrap { lhs, .. }
				| Opcode::BitAnd { lhs, .. }
				| Opcode::BitOr { lhs, .. }
				| Opcode::BitXor { lhs, .. } => type_of(values, instructions, lhs),
				Opcode::BitNot { op, .. } => type_of(values, instructions, op),

				Opcode::StructInit { struct_ty, .. } => *struct_ty,
				Opcode::ArrayInit { array_ty, .. } => *array_ty,
				Opcode::SliceInit { slice_ty, .. } => *slice_ty,
				Opcode::AnyptrInit { .. } => values.common.anyptr_t,
				Opcode::StructFieldValue { ret_ty, .. } | Opcode::StructFieldPtr { ret_ty, .. } => *ret_ty,
				Opcode::UnionInit { union_ty, .. } => *union_ty,
				Opcode::UnionTag { tag_ty, .. } => *tag_ty,
				Opcode::UnionFieldValue { ret_ty, .. } => *ret_ty,

				// builtins
				Opcode::UnsafeIntCast { dst_ty, .. } => *dst_ty,
				Opcode::UnsafeFloatCast { dst_ty, .. } => *dst_ty,
				Opcode::IntToFloat { dst_ty, .. } => *dst_ty,
				Opcode::SizeOf { of: _ } => values.common.usize_t,
				Opcode::Zeroed { ty } => *ty,
				Opcode::Undefined { ty } => *ty,
				Opcode::BitCast { dst_ty, .. } => *dst_ty,
				Opcode::AnyptrIs { .. } => values.common.bool_t,
				Opcode::AnyptrAs { target_ty, .. } => *target_ty,
				Opcode::SliceFromRawParts { slice_ty, .. } => *slice_ty,
				Opcode::SlicePtr { ptr_ty, .. } => *ptr_ty,
				Opcode::SliceLen { .. } => values.common.usize_t,
				Opcode::PtrToInt { dst_ty, .. } => *dst_ty,
				Opcode::IntToPtr { dst_ty, .. } => *dst_ty,
				Opcode::Abort | Opcode::Unreachable => values.common.never_t,
				Opcode::SliceCopyNonoverlapping { .. } => values.common.void_t,
				Opcode::PtrElemPtr { elem_ptr_ty, .. } | Opcode::SliceElemPtr { elem_ptr_ty, .. } => *elem_ptr_ty,

				// these are statements and thus do not produce any value
				Opcode::Store { .. }
				| Opcode::Return { .. }
				| Opcode::DbgSrcLoc { .. }
				| Opcode::Break { .. }
				| Opcode::Repeat { .. }
				| Opcode::Noop => values.common.void_t,

				// invalid insts, should not be analyzed upon
				Opcode::StackAllocInferred { .. } => {
					unreachable!("instruction {id} is untyped ({:?})", opcode)
				},
			}
		},
	}
}
