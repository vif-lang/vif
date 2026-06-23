use std::{
	ops::Deref,
	sync::SyncView,
};

use bumpalo::Bump;
use internment::Intern;
use paste::paste;

use super::{
	IdentKind,
	Radix,
};
use crate::common::{
	NonMaxU32,
	Span,
};

pub type NodeId = NonMaxU32;

// =============================================================================
//                                Module (file)
// =============================================================================

pub(super) struct ModuleData(SyncView<Bump>);

impl ModuleData {
	#[inline(always)]
	pub(super) fn new() -> Self {
		Self(SyncView::new(Bump::with_capacity(1024 * 1024)))
	}

	#[inline(always)]
	pub(super) fn zeroed() -> Self {
		Self(SyncView::new(Bump::with_capacity(0)))
	}

	#[inline(always)]
	pub(super) fn push<T: Sized + Copy>(
		&mut self,
		data: &T,
	) -> &'static T {
		let val = self.0.as_mut().alloc(*data);
		// SAFETY: BPA lives as long as ModuleData and references allocated from it
		// should not be used after ModuleData is dropped.
		unsafe { &*(val as *const T) }
	}

	#[inline(always)]
	pub(super) fn push_slice<T: Sized + Copy>(
		&mut self,
		data: &[T],
	) -> &'static [T] {
		let val = self.0.as_mut().alloc_slice_copy(data);
		// SAFETY: BPA lives as long as ModuleData and references allocated from it
		// should not be used after ModuleData is dropped.
		unsafe { &*(val as *const [T]) }
	}

	#[inline(always)]
	pub(super) fn push_slice_from<T: Sized + Copy, I>(
		&mut self,
		iterator: I,
	) -> &'static [T]
	where
		I: IntoIterator<Item = T>,
		I::IntoIter: ExactSizeIterator,
	{
		let val = self.0.as_mut().alloc_slice_fill_iter::<T, I>(iterator);

		// SAFETY: BPA lives as long as ModuleData and references allocated from it
		// should not be used after ModuleData is dropped.
		unsafe { &*(val as *const [T]) }
	}
}

impl core::fmt::Debug for ModuleData {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		// write!(
		// f,
		// "ModuleData({:.2} kB)",
		// self.0.as_ref().allocated_bytes() as f64 / 1024.0
		// )
		todo!()
	}
}

#[derive(Debug)]
pub struct Module {
	pub kind: ModuleKind,
	pub(super) data: ModuleData,
}

#[derive(Debug)]
pub enum ModuleKind {
	StructDecl(StructTy),
	None,
}

// =============================================================================
//                                 Identifier
// =============================================================================

#[derive(Copy, Clone, Debug)]
pub struct Ident {
	pub symbol: Intern<str>,
	pub kind: IdentKind,
	pub span: Span,
}

impl Ident {
	#[inline(always)]
	pub fn is_user(&self) -> bool {
		self.kind.is_user()
	}

	#[inline(always)]
	pub fn is_generic(&self) -> bool {
		self.kind.is_generic()
	}

	#[inline(always)]
	pub fn is_builtin(&self) -> bool {
		self.kind.is_builtin()
	}
}

impl PartialEq for Ident {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.symbol == other.symbol
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.symbol != other.symbol
	}
}

impl Eq for Ident {}

impl core::hash::Hash for Ident {
	#[inline(always)]
	fn hash<H: core::hash::Hasher>(
		&self,
		state: &mut H,
	) {
		self.symbol.hash(state);
	}
}

impl core::fmt::Display for Ident {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		write!(f, "{}", self.symbol)
	}
}

// =============================================================================
//                                 Expressions
// =============================================================================
#[derive(Copy, Clone, Debug)]
pub struct Expr {
	pub id: NodeId,
	pub kind: ExprKind,
	pub span: Span,
}

impl Expr {
	#[inline(always)]
	pub const fn is_control_flow(&self) -> bool {
		matches!(
			self.kind,
			ExprKind::If(..) | ExprKind::Loop(..) | ExprKind::While(..) | ExprKind::For(..) | ExprKind::Switch(..)
		)
	}

	#[inline(always)]
	pub const fn is_user_ident(&self) -> bool {
		if let ExprKind::Ident(ident) = &self.kind {
			ident.kind.is_user()
		} else {
			false
		}
	}

	#[inline(always)]
	pub const fn is_builtin_ident(&self) -> bool {
		if let ExprKind::Ident(ident) = &self.kind {
			ident.kind.is_builtin()
		} else {
			false
		}
	}

	#[inline(always)]
	pub const fn is_generic_ident(&self) -> bool {
		if let ExprKind::Ident(ident) = &self.kind {
			ident.kind.is_generic()
		} else {
			false
		}
	}

	#[inline(always)]
	pub const fn as_generic_ident(&self) -> Option<&Ident> {
		if let ExprKind::Ident(ident) = self.kind
			&& core::hint::unlikely(ident.kind.is_generic())
		{
			return Some(ident);
		}

		None
	}

	#[inline(always)]
	pub const fn as_user_ident(&self) -> Option<&Ident> {
		if let ExprKind::Ident(ident) = self.kind
			&& core::hint::unlikely(ident.kind.is_user())
		{
			return Some(ident);
		}

		None
	}
}

impl PartialEq for Expr {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

impl Eq for Expr {}

#[derive(Copy, Clone, Debug)]
pub enum ExprKind {
	/// `lhs + rhs`
	Add(&'static BinOp),
	/// `lhs +| rhs`
	AddSat(&'static BinOp),
	/// `lhs - rhs`
	Sub(&'static BinOp),
	/// `lhs -| rhs`
	SubSat(&'static BinOp),
	/// `lhs * rhs`
	Mul(&'static BinOp),
	/// `lhs *| rhs`
	MulSat(&'static BinOp),
	/// `lhs / rhs`
	Div(&'static BinOp),
	/// `lhs % rhs`
	Rem(&'static BinOp),
	/// `lhs << rhs`
	Shl(&'static BinOp),
	/// `lhs <<| rhs`
	ShlSat(&'static BinOp),
	/// `lhs <<% rhs`
	ShlWrap(&'static BinOp),
	/// `lhs >> rhs`
	Shr(&'static BinOp),
	/// `lhs >>| rhs`
	ShrSat(&'static BinOp),
	/// `lhs >>% rhs`
	ShrWrap(&'static BinOp),
	/// `lhs & rhs`
	BitAnd(&'static BinOp),
	/// `lhs | rhs`
	BitOr(&'static BinOp),
	/// `lhs ^ rhs`
	BitXor(&'static BinOp),
	/// `lhs < rhs`
	Lt(&'static BinOp),
	/// `lhs <= rhs`
	Lte(&'static BinOp),
	/// `lhs > rhs`
	Gt(&'static BinOp),
	/// `lhs >= rhs`
	Gte(&'static BinOp),
	/// `lhs && rhs`
	BoolAnd(&'static BinOp),
	/// `lhs || rhs`
	BoolOr(&'static BinOp),
	/// `lhs == rhs`
	Eq(&'static BinOp),
	/// `lhs != rhs`
	Neq(&'static BinOp),
	/// `-<expr>`
	Neg(&'static Expr),
	/// `+<expr>`
	Pos(&'static Expr),
	/// `!<expr>`
	Not(&'static Expr),
	/// `~<expr>`
	BitNot(&'static Expr),
	/// `( ... )` or `∅ ... ∅`
	Group(&'static Expr),
	/// `{ ... }`
	Block(&'static Block),
	/// `if cond { ... } else { ... }`
	If(&'static If),
	/// `loop { ... }`
	Loop(&'static Block),
	/// `while cond { ... }`
	While(&'static While),
	/// `for init in src { ... }`
	For(&'static For),
	/// `switch expr { ... }`
	Switch(&'static Switch),
	/// Literal
	Lit(&'static Lit),
	/// Type
	Type(&'static Type),
	/// `foo`
	Ident(&'static Ident),
	/// `foo.bar`
	FieldAccess(&'static FieldAccess),
	/// `foo.*`
	Deref(&'static Expr),
	/// `foo.?`
	UnwrapNullable(&'static Expr),
	/// `<expr>[<index>]`
	Index(&'static Index),
	/// `&foo`
	AddressOf(&'static Expr),
	/// `try <expr>`
	Try(&'static Expr),
	/// `undefined`
	Undefined,
	/// `foo(a, b, c)`
	FnCall(&'static FnCall),
	/// `.{ .x = 1, .y = 2 }`
	StructInit(&'static StructInit),
	/// `[N:sentinel]{ value1, value2, ..., valueN }`
	ArrayInit(&'static ArrayInit),
	/// `_`
	Discard,
}

impl PartialEq for ExprKind {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) == core::mem::discriminant(other)
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) != core::mem::discriminant(other)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct BinOp {
	pub lhs: &'static Expr,
	pub rhs: &'static Expr,
}

#[derive(Copy, Clone, Debug)]
pub struct Block {
	pub id: NodeId,
	pub is_const: bool,
	pub label: Option<Ident>,
	pub stmts: &'static [Statement],
	pub span: Span,
}

impl PartialEq for Block {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, Debug)]
pub enum ElseBlock {
	If(If),
	Body(IfBody),
}

impl ElseBlock {
	#[inline(always)]
	pub fn as_if(&self) -> Option<&If> {
		if let ElseBlock::If(if_block) = self { Some(if_block) } else { None }
	}

	#[inline(always)]
	pub fn as_block(&self) -> Option<&Block> {
		if let ElseBlock::Body(IfBody::Block(block)) = self {
			Some(block)
		} else {
			None
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub enum IfBody {
	Block(Block),
	Expr(&'static Expr),
}

impl IfBody {
	#[inline(always)]
	pub fn span(&self) -> Span {
		match self {
			IfBody::Block(block) => block.span,
			IfBody::Expr(expr) => expr.span,
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct If {
	pub id: NodeId,
	pub cond: &'static Expr,
	pub then_body: &'static IfBody,
	pub else_block: Option<&'static ElseBlock>,
	pub span: Span,
}

impl PartialEq for If {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, Debug)]
pub struct While {
	pub id: NodeId,
	pub inline: bool,
	pub cond: &'static Expr,
	pub body: &'static Block,
	pub span: Span,
}

impl PartialEq for While {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, Debug)]
pub struct For {
	pub id: NodeId,
	pub inline: bool,
	pub iter_var: &'static Expr,
	pub iterable: &'static Expr,
	pub body: &'static Block,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct Switch {
	pub id: NodeId,
	pub label: Option<Ident>,
	pub expr: &'static Expr,
	pub cases: &'static [SwitchCase],
	pub else_capture: Option<&'static Expr>,
	pub else_body: Option<&'static SwitchBody>,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct SwitchCase {
	pub id: NodeId,
	pub patterns: &'static [Expr],
	pub capture: Option<Ident>,
	pub body: &'static SwitchBody,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub enum SwitchBody {
	Block(&'static Block),
	Expr(&'static Expr),
}

impl SwitchBody {
	#[inline(always)]
	pub fn span(&self) -> Span {
		match self {
			SwitchBody::Block(block) => block.span,
			SwitchBody::Expr(expr) => expr.span,
		}
	}
}

impl PartialEq for For {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, Debug)]
pub enum Type {
	/// `[N]T` or `[N:sentinel]T`
	/// If `size` is None, it is inferred (e.g. `[_]u8`)
	Array {
		ty: &'static Expr,
		is_const: bool,
		size: Option<&'static Expr>,
		sentinel: Option<&'static Expr>,
	},
	/// `*T` (and friends)
	Ptr {
		ty: &'static Expr,
		modifiers: PtrModifiers,
	},
	/// `[*]T`, `[*:sentinel]T`, etc.
	ManyPtr {
		ty: &'static Expr,
		sentinel: Option<&'static Expr>,
		modifiers: PtrModifiers,
	},
	/// `[]T` or `[:sentinel]T`
	Slice {
		ty: &'static Expr,
		sentinel: Option<&'static Expr>,
		modifiers: PtrModifiers,
	},
	/// `fn(arg_0, arg_1, ..., arg_N) ret`
	Fn(&'static FnSig),
	/// `?T`
	Nullable(&'static Expr),

	Struct(&'static StructTy),
	Union(&'static UnionTy),
	Enum(&'static EnumTy),

	// Builtin scalar
	Bool,
	Int(IntSuffix),
	Float(FloatSuffix),

	Void,
	Type,
	Anyptr,
	Anyint,
	Anyfloat,
	// `!` a.k.a. noreturn
	Never,
}

impl PartialEq for Type {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) == core::mem::discriminant(other)
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) != core::mem::discriminant(other)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct PtrModifiers {
	pub is_const: bool,
	pub is_volatile: bool,
	pub addrspace: Option<&'static Expr>,
}

#[derive(Copy, Clone, Debug)]
pub struct FnSig {
	pub params: &'static [Expr],
	pub variadic: bool,
	pub ret_ty: &'static Expr,
}

#[derive(Copy, Clone, Debug)]
pub struct Index {
	pub collection: &'static Expr,
	pub kind: IndexKind,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum IndexKind {
	Index(&'static Expr),
	Range {
		start: Option<&'static Expr>,
		end: Option<&'static Expr>,
	},
	RangeInclusive {
		start: Option<&'static Expr>,
		end: &'static Expr,
	},
}

#[derive(Copy, Clone, Debug)]
pub struct FnCall {
	pub callee: &'static Expr,
	pub args: &'static [Arg],
}

#[derive(Copy, Clone, Debug)]
pub enum Arg {
	Named { name: Ident, value: &'static Expr },
	Positional(&'static Expr),
}

#[derive(Copy, Clone, Debug)]
pub struct StructTy {
	pub fields: &'static [FieldDef],
	pub associated_items: &'static [AssociatedItem],
	pub packed: bool,
	pub linear: bool,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct FieldDef {
	pub id: NodeId,
	pub ident: Ident,
	pub is_pub: bool,
	pub ty: &'static Expr,
	pub default: Option<&'static Expr>,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub enum UnionTagKind {
	/// `union { ... }`: bare/untagged union
	Bare,
	/// `union(enum) { ... }`: auto-tagged union
	AutoEnum,
	/// `union(ExplicitEnumType) { ... }`: tagged union with explicit enum type
	Enum(&'static Expr),
}

#[derive(Copy, Clone, Debug)]
pub struct UnionFieldDef {
	pub id: NodeId,
	pub ident: Ident,
	pub ty: Option<&'static Expr>,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct UnionTy {
	pub tag: UnionTagKind,
	pub fields: &'static [UnionFieldDef],
	pub associated_items: &'static [AssociatedItem],
	pub linear: bool,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct EnumTy {
	pub tag_ty: Option<&'static Expr>,
	pub variants: &'static [EnumVariantDef],
	pub associated_items: &'static [AssociatedItem],
	pub linear: bool,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct EnumVariantDef {
	pub id: NodeId,
	pub ident: Ident,
	pub value: Option<&'static Expr>,
	pub span: Span,
}

#[derive(Copy, Clone, Debug)]
pub struct ErrorTy {}

#[derive(Copy, Clone, Debug)]
pub struct StructInit {
	/// if Some, the type is explicitly specified
	/// `.{ .x = 1, .y = 2 }` vs `Type { .x = 1, .y = 2 }`
	pub ty: Option<&'static Expr>,
	pub fields: &'static [FieldInit],
}

#[derive(Copy, Clone, Debug)]
pub struct ArrayInit {
	pub ty: Option<&'static Expr>,
	pub elements: &'static [Expr],
}

#[derive(Copy, Clone, Debug)]
pub struct FieldInit {
	pub ident: Ident,
	pub value: &'static Expr,
}

#[derive(Copy, Clone, Debug)]
pub struct AssociatedItem {
	pub id: NodeId,
	pub kind: AssociatedItemKind,
	pub is_pub: bool,
	pub span: Span,
}

impl PartialEq for AssociatedItem {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, Debug)]
pub enum AssociatedItemKind {
	Fn(Fn),
	Const(VarBinding),
	Var(VarBinding),
}

impl PartialEq for AssociatedItemKind {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) == core::mem::discriminant(other)
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(self) != core::mem::discriminant(other)
	}
}

// =============================================================================
//                                  Literals
// =============================================================================

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Lit {
	Null,
	Str(Intern<[u8]>),
	Char(u8),
	Bool(bool),
	Integer {
		symbol: Intern<str>,
		radix: Radix,
		suffix: Option<IntSuffix>,
	},
	Float {
		symbol: Intern<str>,
		suffix: Option<FloatSuffix>,
	},
	/// `.variant`
	EnumVariant(Intern<str>),
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum IntSuffix {
	/// u8, u16, etc.
	U(u16),
	/// i8, i16, etc.
	I(u16),
	/// usize
	Usize,
	/// isize
	Isize,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum FloatSuffix {
	F16,
	F32,
	F64,
	F128,
}

// =============================================================================
//                                 Statements
// =============================================================================

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Statement {
	pub id: NodeId,
	pub kind: StatementKind,
	pub span: Span,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum StatementKind {
	/// `;` expression, effectively ignoring the result.
	Expr(&'static Expr),
	Var(&'static VarBinding),
	Const(&'static VarBinding),
	ComptimeVarBinding(&'static VarBinding),
	Return(Option<&'static Expr>),
	Defer(&'static Expr),
	Errdefer(&'static Expr),
	Assign {
		lhs: &'static Expr,
		op: AssignOp,
		rhs: &'static Expr,
	},
	Break {
		label: Option<Ident>,
		value: Option<&'static Expr>,
	},
	Continue {
		label: Option<Ident>,
		value: Option<&'static Expr>,
	},
}

impl core::fmt::Display for StatementKind {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		match self {
			StatementKind::Expr(..) => write!(f, "Expr"),
			StatementKind::Var(..) => write!(f, "Var"),
			StatementKind::Const(..) => write!(f, "Const"),
			StatementKind::ComptimeVarBinding(..) => write!(f, "Comptime Binding"),
			StatementKind::Return(..) => write!(f, "Return"),
			StatementKind::Defer(..) => write!(f, "Defer"),
			StatementKind::Errdefer(..) => write!(f, "Errdefer"),
			StatementKind::Assign { .. } => write!(f, "Assign"),
			StatementKind::Break { .. } => write!(f, "Break"),
			StatementKind::Continue { .. } => write!(f, "Continue"),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct VarBinding {
	pub id: NodeId,
	pub name: Ident,
	pub ty: Option<&'static Expr>,
	pub val: &'static Expr,
}

impl PartialEq for VarBinding {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AssignOp {
	/// `=`
	Assign,
	/// `+=`
	Add,
	/// `+|=`
	AddSat,
	/// `-=`
	Sub,
	/// `-|=`
	SubSat,
	/// `*=`
	Mul,
	/// `*|=`
	MulSat,
	/// `/=`
	Div,
	/// `%=`
	Rem,
	/// `&=`
	BitAnd,
	/// `|=`
	BitOr,
	/// `^=`
	BitXor,
	/// `<<=`
	Shl,
	/// `<<|=`
	ShlSat,
	/// `<<%=`
	ShlWrap,
	/// `>>=`
	Shr,
	/// `>>|=`
	ShrSat,
	/// `>>%=`
	ShrWrap,
	/// `&&=`
	BoolAnd,
	/// `||=`
	BoolOr,
}

impl core::fmt::Display for AssignOp {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		match self {
			AssignOp::Assign => write!(f, "="),
			AssignOp::Add => write!(f, "+="),
			AssignOp::AddSat => write!(f, "+|="),
			AssignOp::Sub => write!(f, "-="),
			AssignOp::SubSat => write!(f, "-|="),
			AssignOp::Mul => write!(f, "*="),
			AssignOp::MulSat => write!(f, "*|="),
			AssignOp::Div => write!(f, "/="),
			AssignOp::Rem => write!(f, "%="),
			AssignOp::BitAnd => write!(f, "&="),
			AssignOp::BitOr => write!(f, "|="),
			AssignOp::BitXor => write!(f, "^="),
			AssignOp::Shl => write!(f, "<<="),
			AssignOp::ShlSat => write!(f, "<<|="),
			AssignOp::ShlWrap => write!(f, "<<%="),
			AssignOp::Shr => write!(f, ">>="),
			AssignOp::ShrSat => write!(f, ">>|="),
			AssignOp::ShrWrap => write!(f, ">>%="),
			AssignOp::BoolAnd => write!(f, "&&="),
			AssignOp::BoolOr => write!(f, "||="),
		}
	}
}

// =============================================================================
//                                    Paths
// =============================================================================

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct FieldAccess {
	pub lhs: &'static Expr,
	pub field: &'static Ident,
	pub span: Span,
}

impl core::fmt::Display for FieldAccess {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		let mut parts: Vec<&Ident> = Vec::new();
		let mut current: &Expr = self.lhs;

		while let ExprKind::FieldAccess(fa) = &current.kind {
			parts.push(fa.field);
			current = fa.lhs;
		}

		match &current.kind {
			ExprKind::Ident(gi) => write!(f, "{}", gi)?,
			_ => write!(f, "<expr>")?,
		}

		for field in parts.iter().rev() {
			write!(f, ".{}", field)?;
		}

		write!(f, ".{}", self.field)
	}
}

// =============================================================================
//                                 Functions
// =============================================================================

#[derive(Copy, Clone, Debug)]
pub struct Fn {
	pub ident: Ident,
	pub ext: Extern,
	pub inline: Inline,
	pub comptime: bool,
	pub callconv: Option<&'static Expr>,
	pub variadic: bool,
	pub params: &'static [FnParam],
	pub ret_ty: &'static Expr,
	pub block: Option<Block>,
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Inline {
	None,
	Always,
	Never,
}

impl core::fmt::Display for Inline {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		match self {
			Inline::None => write!(f, "mayinline"),
			Inline::Always => write!(f, "#inline"),
			Inline::Never => write!(f, "#noinline"),
		}
	}
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Extern {
	None,
	Implicit,
	Explicit(Intern<str>),
}

impl core::fmt::Display for Extern {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		match self {
			Extern::None => write!(f, ""),
			Extern::Implicit => write!(f, "extern"),
			Extern::Explicit(name) => write!(f, "extern \"{}\"", *name),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct FnParam {
	pub id: NodeId,
	pub comptime: bool,
	pub ident: Ident,
	pub ty: &'static Expr,
}

impl PartialEq for FnParam {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.id == other.id
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.id != other.id
	}
}
