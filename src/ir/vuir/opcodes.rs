use std::fmt::Debug;

use bitvec::vec::BitVec;
use internment::Intern;

use super::{
	InstructionId,
	InstructionRef,
};
use crate::{
	common::{
		IndexVec,
		Span,
	},
	frontend::ast,
};

#[derive(Copy, Clone, Debug)]
pub enum BuiltinKind {
	UnsafeIntCast,
	SizeOf,
	BitSizeOf,
	Zeroed,
	IntFromEnum,
	IntToFloat,
	Import,
	Nullptr,
	SliceFromRawParts,
	SlicePtr,
	SliceLen,
	Abort,
	Unreachable,
	PtrToInt,
	IntToPtr,
	Forget,
	Bitcast,
	SliceCopyNonoverlapping,
	AnyptrIs,
	AnyptrAs,
}

#[derive(Copy, Clone, Debug)]
pub enum FieldTy {
	Ref(InstructionRef),
	Body(&'static [InstructionId]),
}

#[derive(Copy, Clone, Debug)]
pub struct Field {
	pub name: ast::Ident,
	pub ty: FieldTy,
}

#[derive(Copy, Clone, Debug)]
pub struct UnionField {
	pub name: ast::Ident,
	pub ty: Option<FieldTy>,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct EnumVariant {
	pub ident: ast::Ident,
	pub value: Option<(InstructionRef, Span)>,
	pub span: Span,
}

/// Determine how to name something when its name is not known at VUIR generation time
#[derive(Copy, Clone, Debug)]
pub enum NamingKind {
	Anonymous,
	FromDecl,
	FromPreviousStackAlloc,
	Named(Intern<str>),
}

#[derive(Copy, Clone, Debug)]
pub struct AdtInitField {
	pub name: ast::Ident,
	pub value: InstructionRef,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct FnCallArg {
	/// The name of the argument if it's a named argument, None for positional
	pub name: Option<Intern<str>>,
	pub body: &'static [InstructionId],
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct FnCallGenericArg {
	/// The name of the generic argument if named, None for positional
	pub name: Intern<str>,
	pub value: InstructionRef,
	pub span: Span,
}

/// A value capture from another namespace
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Capture {
	/// Take parent value
	FromParent(usize),
	Id(InstructionId),
}

#[derive(Clone, Debug)]
pub enum AggregateInitKind {
	/// A special case for `.{}` where there is an ambiguity between array vs adt, this allows semantic analysis
	/// to disambiguate
	Empty,
	Array(&'static [InstructionRef]),
	Adt(&'static [AdtInitField]),
}

#[derive(Clone, Debug)]
pub enum Opcode {
	/// Generally used as a placeholder to construct instructions later, if
	/// encountered = instant ICE
	Invalid,

	/// Obtain the return function of the current function
	TypeOfCurFnRet,

	/// Obtain the type of a struct field in the context of a StructInit expression
	/// Returns a poisoned value if the field does not exist, without any diagnostic emitted. It is
	/// expected that the diagnostic is emitted from the actual StructInit expression so we can accumulate as much diagnostic as possible.
	StructInitTypeOfField {
		r#struct: InstructionRef,
		field: Intern<str>,
	},
	/// Type of `builtin.CallingConvention`, resolved during sema.
	TypeBuiltinCallingConvention,
	TypeOfPtrPointee {
		ptr: InstructionRef,
	},
	TypeOf {
		value: InstructionRef,
	},

	TypePtr {
		pointee: InstructionRef,
		is_const: bool,
		is_volatile: bool,
		span: Span,
	},
	TypeSlice {
		elem: InstructionRef,
		is_const: bool,
		sentinel: Option<InstructionRef>,
	},
	TypeArray {
		elem: InstructionRef,
		is_const: bool,
		len: InstructionRef,
		sentinel: Option<InstructionRef>,
		elem_span: Span,
		len_span: Span,
		span: Span,
	},
	// Declarations
	Declaration(Decl),
	DeclFn {
		ret_ty: InstructionId,
		ret_ty_is_generic: bool,
		params: &'static [InstructionId],
		first_positional_arg_index: Option<u16>,
		var_args: bool,
		body: &'static [InstructionId],
		external: bool,
		callconv: Option<InstructionId>,
		builtin: Option<BuiltinKind>,
		inline: ast::Inline,
		span: Span,
	},
	CaptureGet {
		idx: usize,
		span: Span,
	},
	DeclFnParam {
		name: ast::Ident,
		type_body: &'static [InstructionId],
		comptime: bool,
		generic: bool,
		span: Span,
	},

	DeclStruct {
		naming: NamingKind,
		fields: Vec<Field>,
		packed: bool,
		linear: bool,
		decls: Vec<InstructionId>,
		captures: &'static [Capture],
	},
	DeclEnum {
		tag_ty: Option<(InstructionRef, Span)>,
		naming: NamingKind,
		linear: bool,
		variants: Vec<EnumVariant>,
		decls: Vec<InstructionId>,
		captures: &'static [Capture],
	},
	DeclUnion {
		/// None = bare union, Some(None) = auto-tagged union(enum), Some(Some(...)) = explicit tag type
		tag: Option<Option<(InstructionRef, Span)>>,
		naming: NamingKind,
		linear: bool,
		fields: Vec<UnionField>,
		decls: Vec<InstructionId>,
		captures: &'static [Capture],
	},

	StackAlloc {
		name: ast::Ident,
		ty: InstructionRef,
		span: Span,
	},
	StackAllocMut {
		name: ast::Ident,
		ty: InstructionRef,
		span: Span,
	},
	StackAllocComptime {
		name: ast::Ident,
		ty: InstructionRef,
		span: Span,
	},
	StackAllocComptimeMut {
		name: ast::Ident,
		ty: InstructionRef,
		span: Span,
	},
	StackAllocInferred {
		name: ast::Ident,
		span: Span,
	},
	StackAllocInferredMut {
		name: ast::Ident,
		span: Span,
	},
	StackAllocInferredComptime {
		name: ast::Ident,
		span: Span,
	},
	StackAllocInferredComptimeMut {
		name: ast::Ident,
		span: Span,
	},
	/// Freeze a stack alloc (make it immutable)
	FreezeStackAlloc {
		alloc: InstructionId,
		span: Span,
	},

	/// Reify an inferred stack allocation into a concrete allocation using previous store to inferred alloc insts, after this point the original allocation instruction
	/// must never be used again
	ReifyInferredAlloc {
		alloc: InstructionRef,
		span: Span,
	},

	Block {
		instructions: &'static [InstructionId],
		span: Span,
	},
	BlockComptime {
		instructions: &'static [InstructionId],
	},
	Break {
		block: InstructionId,
		value: InstructionRef,
		value_span: Span,
	},
	/// Break from a compile-time block of instructions, the block may be a BlockComptime or any other instruction that qualify as a block (= that contains instructions)
	/// such as function declarations and etc..
	BreakComptime {
		block: InstructionId,
		value: InstructionRef,
	},
	Return {
		value: Option<InstructionRef>,
		span: Span,
	},
	Loop {
		instructions: &'static [InstructionId],
		span: Span,
	},
	/// Jump to the start of a loop
	Repeat {
		r#loop: InstructionId,
	},

	/// Value of a declaration via a simple identifier
	DeclVal(ast::Ident),
	DeclRef(ast::Ident),

	FieldValFromPtr {
		lhs: InstructionRef,
		field: Intern<str>,
		span: Span,
	},
	FieldPtrFromPtr {
		lhs: InstructionRef,
		field: Intern<str>,
		span: Span,
	},
	FieldValFromVal {
		lhs: InstructionRef,
		field: Intern<str>,
		span: Span,
	},

	ArrayIndexElemVal {
		/// ptr to array
		array_ptr: InstructionRef,
		index: InstructionRef,
		span: Span,
	},

	ArrayIndexElemPtr {
		/// ptr to array
		array_ptr: InstructionRef,
		index: InstructionRef,
		span: Span,
	},

	/// An undefined value
	Undefined {
		ty: Option<InstructionRef>,
		span: Span,
	},

	/// Initialize an aggregate of type `ty` and returns it
	AggregateInit {
		ty: InstructionRef,
		kind: AggregateInitKind,
		span: Span,
	},

	Coerce {
		value: InstructionRef,
		into: InstructionRef,
		span: Span,
	},

	FnCall {
		fun: InstructionRef,
		generic_args: &'static [FnCallGenericArg],
		args: &'static [FnCallArg],
		ret_ty: Option<InstructionRef>,
		span: Span,
	},

	/// A FnCall coming from a field ptr
	FnCallWithFieldPtrReceiver {
		field_ptr: InstructionRef,
		field_name: ast::Ident,
		generic_args: &'static [FnCallGenericArg],
		args: &'static [FnCallArg],
		ret_ty: Option<InstructionRef>,
		span: Span,
	},

	Load {
		src: InstructionRef,
		span: Span,
	},
	Store {
		dst: InstructionRef,
		src: InstructionRef,
		span: Span,
	},
	/// Store to a inferred allocation and gives it its type
	StoreToInferredAlloc {
		dst: InstructionRef,
		src: InstructionRef,
		span: Span,
	},

	// arithmetic
	Add {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	AddSat {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Sub {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	SubSat {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Mul {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	MulSat {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Div {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Rem {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Lt {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Lte {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Gt {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Gte {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BoolAnd {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BoolOr {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Negate {
		op: InstructionRef,
		span: Span,
	},
	BoolNot {
		op: InstructionRef,
		span: Span,
	},

	// bitwise
	Shl {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	ShlSat {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	ShlWrap {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Shr {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	ShrSat {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	ShrWrap {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BitAnd {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BitOr {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BitXor {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	BitNot {
		op: InstructionRef,
		span: Span,
	},

	// comparaison ops
	Eq {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},
	Neq {
		lhs: InstructionRef,
		rhs: InstructionRef,
		span: Span,
	},

	DbgSrcLoc {
		line: usize,
		col: usize,
	},

	/// Conditionally branch between two bodies depending on cond
	/// Must be wrapped with a Block with each body breaking back into the block unless both bodies never returns
	Branch {
		cond: (InstructionRef, Span),
		then_body: &'static [InstructionId],
		else_body: &'static [InstructionId],
		span: Span,
	},
	/// Must be wrapped with a Block with each body breaking back into the blockx
	Switch {
		operand: InstructionRef,
		single_cases: &'static [SwitchSingleCase],
		multi_cases: &'static [SwitchMultiCase],
		else_body: Option<&'static [InstructionId]>,
		span: Span,
	},
	Defer {
		body: &'static [InstructionId],
		span: Span,
	},

	/// Captures the payload of a tagged union in a switch case.
	/// Resolved in sema to a UnionFieldValue.
	SwitchCapture {
		switch_operand: InstructionRef,
		case_pattern: InstructionRef,
		span: Span,
	},

	RvalueToLvalue {
		rvalue: InstructionRef,
	},
}

impl Opcode {
	pub fn returns_never(&self) -> bool {
		matches!(self, Opcode::Break { .. } | Opcode::Return { .. } | Opcode::Repeat { .. })
	}
}

#[derive(Clone, Debug)]
pub struct SwitchSingleCase {
	pub pattern: InstructionRef,
	pub capture: Option<ast::Ident>,
	pub body: &'static [InstructionId],
	pub span: Span,
	pub pattern_span: Span,
}

#[derive(Clone, Debug)]
pub struct SwitchMultiCase {
	pub items: &'static [InstructionRef],
	pub body: &'static [InstructionId],
	pub span: Span,
	pub patterns_span: Span,
}

/// A declaration binds a list of instruction that returns a value to a name,
/// such as a structure, a function, a global variable, ..
#[derive(Clone, Debug)]
pub struct Decl {
	pub name: Intern<str>,
	pub value: &'static [InstructionId],
	pub span: Span,
}
