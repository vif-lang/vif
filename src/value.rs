use std::{
	fmt::{
		Debug,
		Display,
		Pointer,
	},
	hash::{
		Hash,
		Hasher,
	},
	marker::{
		PhantomData,
		PhantomPinned,
	},
	ops::Deref,
	sync::{
		LazyLock,
		atomic::Ordering,
		nonpoison::{
			Mutex,
			MutexGuard,
		},
	},
};

use bitvec::vec::BitVec;
use internment::Intern;
use rustc_hash::FxBuildHasher;

pub use crate::int::Anyint;
use crate::{
	common::{
		self,
		sharded_index_map::ShardedIndexMap,
	},
	compile_unit::{
		DeclId,
		NamespaceId,
		ResolvedTargetInfo,
		module::ModuleId,
	},
	frontend::ast::{
		self,
		Inline,
	},
	ir::vuir,
};

pub type Index = common::sharded_index_map::Index;

/// Wraps a vuir InstructionId with its originating module
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GlobalVuirInstructionId {
	pub module: ModuleId,
	pub inst: vuir::InstructionId,
}

// =============================================================================
//                             Primitive Wrappers
// =============================================================================

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Anyfloat(pub f128);

impl Eq for Anyfloat {}

impl Hash for Anyfloat {
	fn hash<H: std::hash::Hasher>(
		&self,
		state: &mut H,
	) {
		self.0.to_bits().hash(state);
	}
}

impl Display for Anyfloat {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		self.0.fmt(f)
	}
}

impl Deref for Anyfloat {
	type Target = f128;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

// =============================================================================
//                                Pointer Types
// =============================================================================

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ComptimeAllocId(pub usize);
impl From<ComptimeAllocId> for usize {
	fn from(value: ComptimeAllocId) -> Self {
		value.0
	}
}
impl From<usize> for ComptimeAllocId {
	fn from(value: usize) -> Self {
		Self(value)
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PtrKind {
	Decl(DeclId),
	ComptimeAlloc(ComptimeAllocId),
	Value(Index),
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypePtrPacked {
	pub bit_offset: u32,
	pub bit_width: u32,
	pub underlying_int_bits: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypePtr {
	pub pointee_ty: Index,
	/// If `Some`, this pointer points at a specific
	/// bit of a byte (used for packed struct ptrs)
	pub packed: Option<TypePtrPacked>,
	pub is_const: bool,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Ptr {
	/// The [`TypePtr`] of this pointer.
	pub ty: Index,
	pub kind: PtrKind,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeSlice {
	pub pointee_ty: Index,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeArray {
	pub elem_ty: Index,
	pub len: u64,
}

// =============================================================================
//                                Struct Types
// =============================================================================

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct StructField {
	pub name: Intern<str>,
	pub ty: Index,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PackedStructFieldInfo {
	/// Offset in bits from the start of the struct
	pub offset: u32,
	/// Width in bits of the field
	pub width: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum StructLayout {
	Standard,
	Packed {
		/// Storage size in bits of the entire struct
		storage_bits: u32,
		/// Total width in bits of all fields
		fields_bits: u32,
		/// Bit field layout information
		packed_fields: &'static [PackedStructFieldInfo],
	},
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeStruct {
	pub name: Intern<str>,
	pub fields: &'static [StructField],
	pub layout: StructLayout,
	pub namespace: NamespaceId,
	pub linear: bool,
}

impl TypeStruct {
	pub fn field_idx_by_name(
		&self,
		name: &str,
	) -> Option<usize> {
		self.fields.iter().position(|f| &*f.name == name)
	}

	pub fn get_packed_field_info(
		&self,
		field_index: usize,
	) -> Option<&PackedStructFieldInfo> {
		if let StructLayout::Packed { packed_fields, .. } = &self.layout {
			packed_fields.get(field_index)
		} else {
			None
		}
	}

	#[inline(always)]
	pub fn is_packed(&self) -> bool {
		matches!(self.layout, StructLayout::Packed { .. })
	}
}

// =============================================================================
//                                Enum Types
// =============================================================================

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct EnumField {
	pub name: Intern<str>,
	pub value: Index,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeEnum {
	pub name: Intern<str>,
	pub tag_ty: Index,
	pub fields: &'static [EnumField],
	pub namespace: NamespaceId,
	pub linear: bool,
}
impl TypeEnum {
	pub fn field_idx_by_name(
		&self,
		name: &str,
	) -> Option<usize> {
		self.fields.iter().position(|f| &*f.name == name)
	}
}

// =============================================================================
//                                Union Types
// =============================================================================

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct UnionField {
	pub name: Intern<str>,
	pub ty: Option<Index>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeUnion {
	pub name: Intern<str>,
	pub tag_ty: Option<Index>,
	pub fields: &'static [UnionField],
	pub namespace: NamespaceId,
	pub linear: bool,
}

impl TypeUnion {
	pub fn field_idx_by_name(
		&self,
		name: &str,
	) -> Option<u32> {
		self.fields.iter().position(|f| &*f.name == name).map(|i| i as u32)
	}
}

// =============================================================================
//                               Function Types
// =============================================================================

/// A function declaration, hold the base TypeFn of the function in respect to its signature
/// Semantic analysis may cause new function types to be generated as a consequence of monomorphization
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FnDecl {
	pub owner_decl: DeclId,
	pub func_decl_inst: GlobalVuirInstructionId,
	pub ty: Index,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TypeFn {
	pub params: &'static [Index],
	pub comptime_params: &'static bitvec::slice::BitSlice<u8>,
	// TODO(zino): not sure this is the right place / way of storing the first non-generic parameter, but having
	// a seperate slice for generics is also suboptimal
	pub first_positional_param: Option<u16>,
	pub var_args: bool,
	pub ret_ty: Index,
	pub external: bool,
	pub callconv: CallingConvention,
	pub inline: ast::Inline,
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
/// Calling conventions, should be in sync with builtin.vif
pub enum CallingConvention {
	Vif,
	C,
	Fast,
	Cold,
	X86_64Windows,

	Count,
}

/// A concrete function value with resolved generic arguments
#[derive(Copy, Clone, Eq, Debug)]
pub struct FnKey {
	pub ty: Index,
	pub decl: Index,
	pub comptime_args: &'static [Option<Index>],

	/// !Hash && !PartialEq
	pub owner_decl: DeclId, // REMOVE
}
impl PartialEq for FnKey {
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.ty == other.ty && self.decl == other.decl && self.comptime_args == other.comptime_args
	}
}
impl Hash for FnKey {
	fn hash<H: std::hash::Hasher>(
		&self,
		state: &mut H,
	) {
		self.ty.hash(state);
		self.decl.hash(state);
		self.comptime_args.hash(state);
	}
}

/// Function value data stored via bump allocator.
#[derive(Copy, Clone, Debug)]
pub struct FnValue {
	pub owner_decl: DeclId,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Capture {
	Comptime(Index),
	Runtime(GlobalVuirInstructionId),
}

/// A type with its own namespace and captures.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NamespaceType {
	pub inst: GlobalVuirInstructionId,
	pub captures: &'static [Capture],
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Type {
	// Types
	Int {
		signed: bool,
		bits: u16,
	},
	Anyint,
	Anyfloat,
	Usize,
	Isize,
	F16,
	F32,
	F64,
	F128,
	Bool,
	Void,
	Struct(NamespaceType),
	Enum(NamespaceType),
	Union(NamespaceType),
	Fn(TypeFn),
	Ptr(TypePtr),
	Slice(TypeSlice),
	Array(TypeArray),
	NullPtr,

	/// Generic type that can be any type possible
	Any,
	/// Runtime type-erased pointer plus compiler type identity
	Anyptr,
	/// Indicate a unknown generic which should be resolved
	GenericPoison,

	Type,
	Never,
	EnumLiteral,
}

/// Hashable and comparable part of a `Value` used to retrieve its index inside the `ValueMap`
///
/// # Triviality
///
/// A key is said to be trivial if its associated value part is trivially deductible from only the key itself.
/// An example is a TypeVoid, it's value is simply None.
///
/// A non-trivial key example is TypeStruct, as its value part store mutable runtime state such as the analysis state.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Key {
	// Values
	Undefined {
		ty: Index,
	},
	Str {
		slice_ty: Index,
		value: Intern<[u8]>,
	},
	Int {
		ty: Index,
		value: Intern<Anyint>,
	},
	Float {
		ty: Index,
		value: Anyfloat,
	},
	Bool(bool),
	Ptr(Ptr),
	Fn(FnKey),
	EnumTag {
		enum_ty: Index,
		val: Index,
	},
	Aggregate {
		ty: Index,
		values: &'static [Index],
	},
	NullPtr,
	Void,
	Unreachable,
	Union {
		ty: Index,
		tag: Option<Index>,
		payload: Option<Index>,
	},

	// Types
	Type(Type),

	/// Per-param generic poison, unique per comptime type param,
	/// so structural unification can distinguish different generics.
	GenericPoison {
		param_id: vuir::InstructionId,
		name: internment::Intern<str>,
	},

	// Misc
	DeclRef {
		vuir: vuir::InstructionId,
	},
	FnDecl(FnDecl),
	EnumLiteral(Intern<str>),
}
impl Key {
	pub fn as_type_fn(&self) -> &TypeFn {
		match self {
			Key::Type(Type::Fn(t)) => t,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_type_ptr(&self) -> &TypePtr {
		match self {
			Key::Type(Type::Ptr(t)) => t,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_type_slice(&self) -> &TypeSlice {
		match self {
			Key::Type(Type::Slice(t)) => t,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_type_array(&self) -> &TypeArray {
		match self {
			Key::Type(Type::Array(t)) => t,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_fn(&self) -> &FnKey {
		match self {
			Key::Fn(f) => f,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_ptr(&self) -> &Ptr {
		match self {
			Key::Ptr(ptr) => ptr,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_int(&self) -> (Index, &Intern<Anyint>) {
		match self {
			Key::Int { ty, value } => (*ty, value),
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_struct(&self) -> &NamespaceType {
		match self {
			Key::Type(Type::Struct(s)) => s,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_enum(&self) -> &NamespaceType {
		match self {
			Key::Type(Type::Enum(e)) => e,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_fn_decl(&self) -> &FnDecl {
		match self {
			Key::FnDecl(t) => t,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn as_bool(&self) -> bool {
		match self {
			Key::Bool(b) => *b,
			_ => unreachable!("{self:?}"),
		}
	}

	pub fn is_type(&self) -> bool {
		matches!(self, Key::Type(_) | Key::GenericPoison { .. })
	}

	pub fn type_is_numeric(&self) -> bool {
		matches!(
			self,
			Key::Type(
				Type::Int { .. }
					| Type::Anyint | Type::Anyfloat
					| Type::F16 | Type::F32
					| Type::F64 | Type::F128
					| Type::Isize | Type::Usize
			)
		)
	}
}

pub struct Layout {
	pub size: u64,
	pub align: u64,
}
impl Layout {
	/// A zero-sized layout but well-aligned (align of 1)
	pub const fn zeroed() -> Self {
		Self { size: 0, align: 1 }
	}
}

pub struct UnionLayout {
	pub union_layout: Layout,
	pub tag: Layout,
	pub payload: Layout,
	/// To minimize padding due to payload/tag alignement store the most aligned field
	/// for later codegen to place tag/payload accordingly
	pub most_aligned_field: (usize, Layout),
	/// Padding inserted after tag and payload to match `union_layout` align
	pub trailing_padding: u64,
}

pub struct DisplayIndex<'store> {
	index: Index,
	store: &'store ValueStore,
}
impl<'store> Display for DisplayIndex<'store> {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		let (key, value) = self.store.value_map.kv(self.index);
		match key {
			Key::Type(ty) => match ty {
				Type::Type => write!(f, "type"),
				Type::Any => write!(f, "any"),
				Type::Anyptr => write!(f, "anyptr"),
				Type::Anyint => write!(f, "anyint"),
				Type::Anyfloat => write!(f, "anyfloat"),
				Type::Usize => write!(f, "usize"),
				Type::Isize => write!(f, "isize"),
				Type::Void => write!(f, "void"),
				Type::Int { signed, bits } => {
					write!(f, "{}{}", if *signed { "i" } else { "u" }, bits)
				},
				Type::F16 => write!(f, "f16"),
				Type::F32 => write!(f, "f32"),
				Type::F64 => write!(f, "f64"),
				Type::F128 => write!(f, "f128"),
				Type::Bool => write!(f, "bool"),
				Type::Struct(_) => {
					let s = value.load(Ordering::Relaxed).as_struct();
					write!(f, "{}", s.name)
				},
				Type::Enum(_) => {
					let e = value.load(Ordering::Relaxed).as_enum();
					write!(f, "{}", e.name)
				},
				Type::Union(_) => {
					let u = value.load(Ordering::Relaxed).as_union();
					write!(f, "{}", u.name)
				},
				Type::GenericPoison => write!(f, "any_poison"),
				Type::Ptr(ptr) => write!(f, "*{}", DisplayIndex {
					store: self.store,
					index: ptr.pointee_ty
				}),
				Type::Slice(slice) => write!(f, "[]{}", DisplayIndex {
					store: self.store,
					index: slice.pointee_ty
				}),
				Type::Array(arr) => write!(f, "[{}]{}", arr.len, DisplayIndex {
					store: self.store,
					index: arr.elem_ty
				}),
				Type::Fn(_) => write!(f, "fn"),
				Type::NullPtr => write!(f, "nullptr"),
				Type::Never => write!(f, "never"),
				Type::EnumLiteral => write!(f, "enum_literal"),
			},
			Key::GenericPoison { name, .. } => write!(f, "{}", name),
			Key::Undefined { .. }
			| Key::Str { .. }
			| Key::Int { .. }
			| Key::Float { .. }
			| Key::Bool(_)
			| Key::Ptr(_)
			| Key::Fn(_)
			| Key::EnumTag { .. }
			| Key::Aggregate { .. }
			| Key::NullPtr
			| Key::Void
			| Key::Unreachable
			| Key::Union { .. }
			| Key::DeclRef { .. }
			| Key::FnDecl(_)
			| Key::EnumLiteral(_) => write!(f, "<value {:?}>", key),
		}
	}
}

/// Wraps a nullable pointer to a usize to be used inside Pod types...
#[derive(Copy)]
#[repr(transparent)]
pub struct ValueStoreBumpRef<T: 'static> {
	ptr: usize,
	_t: PhantomData<T>,
}
impl<T: 'static> ValueStoreBumpRef<T> {
	// # Safety
	//
	// - `ptr` must be a valid pointer to a value inside the bump allocator of a ValueStore
	// - the value pointed by `ptr` must not have any mutable reference alive while this ref exists
	#[inline(always)]
	pub unsafe fn new(ptr: *const T) -> Self {
		assert!(!ptr.is_null());
		Self {
			ptr: ptr as usize,
			_t: PhantomData,
		}
	}

	#[inline(always)]
	pub fn null() -> Self {
		// SAFETY: null ptr
		unsafe { Self::new(std::ptr::null()) }
	}

	#[inline(always)]
	pub fn as_ptr(&self) -> *const T {
		self.ptr as *const _
	}

	/// Pointers stored in `UsizePtr` are expected to originate from the ValueStore bump allocator,
	/// therefore always valid with no aliasing mutable references.
	///
	/// # Panics
	///
	/// Panics if ptr is null
	#[inline(always)]
	pub fn as_ref(&self) -> &T {
		debug_assert!(!self.as_ptr().is_null());
		// SAFETY: `ValueStoreBumpRef` is only constructed from non-null bump references.
		unsafe { self.as_ptr().as_ref_unchecked() }
	}
}
impl<T: Copy + 'static> Clone for ValueStoreBumpRef<T> {
	#[inline(always)]
	fn clone(&self) -> Self {
		*self
	}
}

// SAFETY: `ValueStoreBumpRef` is a transparent wrapper around a non-owning pointer
// and is constrained to `Copy + 'static` payloads.
unsafe impl<T: Copy + 'static> bytemuck::Pod for ValueStoreBumpRef<T> {}
// SAFETY: Rust guarentee that null pointers are the same as zero-initialized pointers
unsafe impl<T: Copy + 'static> bytemuck::Zeroable for ValueStoreBumpRef<T> {}

impl<T: Copy + 'static> Deref for ValueStoreBumpRef<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target {
		self.as_ref()
	}
}

impl<T: Copy + 'static + Debug> Debug for ValueStoreBumpRef<T> {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		f.debug_struct("ValueStoreBumpRef").field("inner", self.as_ref()).finish()
	}
}

/// The value. We try to keep its size as small as possible by relying on the payload
/// bump allocator to offset larger data.
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C, usize)] // usize to remove any padding
pub enum Value {
	None(usize),
	Fn(ValueStoreBumpRef<FnValue>),
	Struct(ValueStoreBumpRef<TypeStruct>),
	Enum(ValueStoreBumpRef<TypeEnum>),
	Union(ValueStoreBumpRef<TypeUnion>),
}

impl Value {
	pub fn none() -> Value {
		Value::None(0)
	}

	#[inline(always)]
	pub fn as_struct(&self) -> ValueStoreBumpRef<TypeStruct> {
		match self {
			Self::Struct(s) => *s,
			_ => unreachable!("{self:?}"),
		}
	}

	#[inline(always)]
	pub fn as_enum(&self) -> ValueStoreBumpRef<TypeEnum> {
		match self {
			Self::Enum(s) => *s,
			_ => unreachable!(),
		}
	}

	#[inline(always)]
	pub fn as_union(&self) -> ValueStoreBumpRef<TypeUnion> {
		match self {
			Self::Union(s) => *s,
			_ => unreachable!(),
		}
	}

	#[inline(always)]
	pub fn as_fn_value(&self) -> ValueStoreBumpRef<FnValue> {
		match self {
			Self::Fn(f) => *f,
			_ => unreachable!("{self:?}"),
		}
	}
}

pub type ValueMap = ShardedIndexMap<Key, Value, FxBuildHasher>;

pub struct ValueStore {
	pub value_map: ValueMap,
	pub common: CommonValues,
	payload_bump_alloc: Mutex<bumpalo::Bump>,

	// we may store pointers to the bump allocator inside keys & values
	_pinned: PhantomPinned,
}
impl ValueStore {
	pub fn new(shard_count: usize) -> Self {
		let value_map = ShardedIndexMap::new(shard_count, 8192, FxBuildHasher);
		let common_types = {
			CommonValues {
				nullptr: value_map.entry(&Key::NullPtr).or_insert_with(Value::none),
				anyint_t: value_map.entry(&Key::Type(Type::Anyint)).or_insert_with(Value::none),
				anyfloat_t: value_map.entry(&Key::Type(Type::Anyfloat)).or_insert_with(Value::none),
				void_t: value_map.entry(&Key::Type(Type::Void)).or_insert_with(Value::none),
				void_value: value_map.entry(&Key::Void).or_insert_with(Value::none),
				any_t: value_map.entry(&Key::Type(Type::Any)).or_insert_with(Value::none),
				anyptr_t: value_map.entry(&Key::Type(Type::Anyptr)).or_insert_with(Value::none),
				type_t: value_map.entry(&Key::Type(Type::Type)).or_insert_with(Value::none),
				generic_poison_t: value_map.entry(&Key::Type(Type::GenericPoison)).or_insert_with(Value::none),
				never_t: value_map.entry(&Key::Type(Type::Never)).or_insert_with(Value::none),
				usize_t: value_map.entry(&Key::Type(Type::Usize)).or_insert_with(Value::none),
				isize_t: value_map.entry(&Key::Type(Type::Isize)).or_insert_with(Value::none),
				u16_t: value_map
					.entry(&Key::Type(Type::Int { signed: false, bits: 16 }))
					.or_insert_with(Value::none),
				u64_t: value_map
					.entry(&Key::Type(Type::Int { signed: false, bits: 64 }))
					.or_insert_with(Value::none),
				i64_t: value_map
					.entry(&Key::Type(Type::Int { signed: true, bits: 64 }))
					.or_insert_with(Value::none),
				u32_t: value_map
					.entry(&Key::Type(Type::Int { signed: false, bits: 32 }))
					.or_insert_with(Value::none),
				i32_t: value_map
					.entry(&Key::Type(Type::Int { signed: true, bits: 32 }))
					.or_insert_with(Value::none),
				f16_t: value_map.entry(&Key::Type(Type::F16)).or_insert_with(Value::none),
				f32_t: value_map.entry(&Key::Type(Type::F32)).or_insert_with(Value::none),
				f64_t: value_map.entry(&Key::Type(Type::F64)).or_insert_with(Value::none),
				f128_t: value_map.entry(&Key::Type(Type::F128)).or_insert_with(Value::none),
				bool_t: value_map.entry(&Key::Type(Type::Bool)).or_insert_with(Value::none),
				enum_literal_t: value_map.entry(&Key::Type(Type::EnumLiteral)).or_insert_with(Value::none),
				nullptr_t: value_map.entry(&Key::Type(Type::NullPtr)).or_insert_with(Value::none),
				unreachable_value: value_map.entry(&Key::Unreachable).or_insert_with(Value::none),
				true_value: value_map.entry(&Key::Bool(true)).or_insert_with(Value::none),
				false_value: value_map.entry(&Key::Bool(false)).or_insert_with(Value::none),
			}
		};
		Self {
			value_map,
			payload_bump_alloc: Mutex::new(bumpalo::Bump::new()),
			common: common_types,
			_pinned: PhantomPinned,
		}
	}

	#[inline(always)]
	pub fn index_to_key(
		&self,
		index: Index,
	) -> &Key {
		self.value_map.key(index)
	}

	#[inline(always)]
	pub fn index_to_value(
		&self,
		index: Index,
	) -> Value {
		self.value_map.kv(index).1.load(Ordering::Relaxed)
	}

	#[inline(always)]
	pub fn index_to_key_value(
		&self,
		index: Index,
	) -> (&Key, Value) {
		let (key, value) = self.value_map.kv(index);
		(key, value.load(Ordering::Relaxed))
	}

	/// Intern a trivial value by its key.
	///
	/// # Panics
	///
	/// Panic if `key` is one of the non-trivial case. See [`Key`] Triviality doc.
	pub fn intern_trivial(
		&self,
		key: &Key,
	) -> Index {
		match key {
			// trivial
			key @ (Key::Type(
				Type::Any
				| Type::Anyptr
				| Type::Ptr(..)
				| Type::Int { .. }
				| Type::Anyfloat
				| Type::Anyint
				| Type::GenericPoison
				| Type::Usize
				| Type::Isize
				| Type::F16
				| Type::F32
				| Type::F64
				| Type::F128
				| Type::Type
				| Type::Void
				| Type::Bool
				| Type::Slice(..)
				| Type::Array(..)
				| Type::Fn(..)
				| Type::Never
				| Type::Enum(..)
				| Type::EnumLiteral
				| Type::NullPtr,
			)
			| Key::GenericPoison { .. }
			| Key::Ptr(..)
			| Key::Int { .. }
			| Key::Float { .. }
			| Key::Str { .. }
			| Key::Bool(..)
			| Key::FnDecl(..)
			| Key::EnumTag { .. }
			| Key::Aggregate { .. }
			| Key::Union { .. }
			| Key::NullPtr
			| Key::Void
			| Key::Unreachable
			| Key::DeclRef { .. }
			| Key::EnumLiteral(..)
			| Key::Undefined { .. }) => self.value_map.entry(key).or_insert_with(Value::none),

			// non-trivial
			key @ (Key::Type(Type::Struct(..) | Type::Union(..)) | Key::Fn(..)) => {
				panic!("cannot intern_trivial {key:?} as it is non-trivial, use the dedicated functions on the value store")
			},
		}
	}

	/// Intern a non-trivial value (TypeStruct, Fn) that needs special handling, will reinsert the value if needed
	pub fn intern_non_trivial(
		&self,
		key: &Key,
		value: Value,
	) -> Index {
		match self.value_map.entry(key) {
			common::sharded_index_map::Entry::Occupied(occupied) => occupied.insert(value).0,
			common::sharded_index_map::Entry::Vacant(vacant) => vacant.insert(value),
		}
	}

	/// Intern a pointer to another value
	pub fn intern_value_ptr(
		&self,
		value: Index,
	) -> Index {
		let ty = self.intern_trivial(&Key::Type(Type::Ptr(TypePtr {
			pointee_ty: self.type_of_interned(value),
			packed: None,
			is_const: false,
		})));
		self.intern_trivial(&Key::Ptr(Ptr {
			ty,
			kind: PtrKind::Value(value),
		}))
	}

	pub fn intern_enum_tag_from_field_idx(
		&self,
		enum_ty: Index,
		i: u32,
	) -> Index {
		let r#enum = self.index_to_value(enum_ty).as_enum();
		let val = r#enum.fields[i as usize].value;
		self.intern_trivial(&Key::EnumTag {
			enum_ty: r#enum.tag_ty,
			val,
		})
	}

	pub fn display_index(
		&self,
		index: Index,
	) -> DisplayIndex<'_> {
		DisplayIndex { index, store: self }
	}

	// =========================================================================
	// Generic poison helpers
	// =========================================================================

	/// Returns Some(param_id) if this is a per-param generic poison marker.
	pub fn as_generic_poison(
		&self,
		ty: Index,
	) -> Option<vuir::InstructionId> {
		match self.index_to_key(ty) {
			Key::GenericPoison { param_id, .. } => Some(*param_id),
			Key::Undefined { .. }
			| Key::Str { .. }
			| Key::Int { .. }
			| Key::Float { .. }
			| Key::Bool(_)
			| Key::Ptr(_)
			| Key::Fn(_)
			| Key::EnumTag { .. }
			| Key::Aggregate { .. }
			| Key::NullPtr
			| Key::Void
			| Key::Unreachable
			| Key::Union { .. }
			| Key::Type(_)
			| Key::DeclRef { .. }
			| Key::FnDecl(_)
			| Key::EnumLiteral(_) => None,
		}
	}

	/// True if this type IS any generic poison (per-param or legacy).
	pub fn is_any_generic_poison(
		&self,
		ty: Index,
	) -> bool {
		ty == self.common.generic_poison_t || self.as_generic_poison(ty).is_some()
	}

	/// True if this type contains a generic poison anywhere in its structure.
	pub fn type_contains_generic_poison(
		&self,
		ty: Index,
	) -> bool {
		if self.is_any_generic_poison(ty) {
			return true;
		}
		match self.index_to_key(ty) {
			Key::Type(ty) => match ty {
				Type::Ptr(ptr) => self.type_contains_generic_poison(ptr.pointee_ty),
				Type::Slice(slice) => self.type_contains_generic_poison(slice.pointee_ty),
				Type::Array(array) => self.type_contains_generic_poison(array.elem_ty),
				Type::Fn(function) => {
					function.params.iter().any(|ty| self.type_contains_generic_poison(*ty))
						|| self.type_contains_generic_poison(function.ret_ty)
				},
				Type::Struct(ns) | Type::Enum(ns) | Type::Union(ns) => ns.captures.iter().any(|cap| match cap {
					Capture::Comptime(cap) => self.type_contains_generic_poison(*cap),
					Capture::Runtime(_) => false,
				}),
				Type::GenericPoison => true,
				Type::Int { .. }
				| Type::Anyint
				| Type::Anyfloat
				| Type::Usize
				| Type::Isize
				| Type::F16
				| Type::F32
				| Type::F64
				| Type::F128
				| Type::Bool
				| Type::Void
				| Type::NullPtr
				| Type::Any
				| Type::Anyptr
				| Type::Type
				| Type::Never
				| Type::EnumLiteral => false,
			},
			Key::GenericPoison { .. } => true,
			Key::Undefined { .. }
			| Key::Str { .. }
			| Key::Int { .. }
			| Key::Float { .. }
			| Key::Bool(_)
			| Key::Ptr(_)
			| Key::Fn(_)
			| Key::EnumTag { .. }
			| Key::Aggregate { .. }
			| Key::NullPtr
			| Key::Void
			| Key::Unreachable
			| Key::Union { .. }
			| Key::DeclRef { .. }
			| Key::FnDecl(_)
			| Key::EnumLiteral(_) => false,
		}
	}

	/// Intern a per-param generic poison.
	pub fn make_generic_poison(
		&self,
		param_id: vuir::InstructionId,
		name: internment::Intern<str>,
	) -> Index {
		self.intern_trivial(&Key::GenericPoison { param_id, name })
	}

	/// Walk a type tree and replace every GenericPoison node with the
	/// corresponding concrete type from `bindings`. Pure data transform.
	pub fn substitute_poisons(
		&self,
		ty: Index,
		bindings: &rustc_hash::FxHashMap<vuir::InstructionId, Index>,
	) -> Index {
		if let Some(param_id) = self.as_generic_poison(ty) {
			return bindings.get(&param_id).copied().unwrap_or(ty);
		}
		if !self.type_contains_generic_poison(ty) {
			return ty;
		}
		let Key::Type(ty_key) = self.index_to_key(ty) else {
			unreachable!("generic substitution expected a type")
		};
		match ty_key {
			Type::Ptr(ptr) => {
				let new_pointee = self.substitute_poisons(ptr.pointee_ty, bindings);
				self.intern_trivial(&Key::Type(Type::Ptr(TypePtr {
					pointee_ty: new_pointee,
					..*ptr
				})))
			},
			Type::Slice(sl) => {
				let new_elem = self.substitute_poisons(sl.pointee_ty, bindings);
				self.intern_trivial(&Key::Type(Type::Slice(TypeSlice { pointee_ty: new_elem })))
			},
			Type::Array(array) => {
				let elem_ty = self.substitute_poisons(array.elem_ty, bindings);
				self.intern_trivial(&Key::Type(Type::Array(TypeArray { elem_ty, ..*array })))
			},
			Type::Fn(function) => {
				let params = function
					.params
					.iter()
					.map(|ty| self.substitute_poisons(*ty, bindings))
					.collect::<Vec<_>>();
				let ret_ty = self.substitute_poisons(function.ret_ty, bindings);
				self.intern_trivial(&Key::Type(Type::Fn(TypeFn {
					params: self.alloc_slice(&params),
					ret_ty,
					..*function
				})))
			},
			Type::Struct(ns) => {
				let new_captures: Vec<Capture> = ns
					.captures
					.iter()
					.map(|cap| match cap {
						Capture::Comptime(cap) => Capture::Comptime(self.substitute_poisons(*cap, bindings)),
						Capture::Runtime(cap) => Capture::Runtime(*cap),
					})
					.collect();
				let new_captures = self.alloc_slice(&new_captures);
				self.intern_non_trivial(
					&Key::Type(Type::Struct(NamespaceType {
						inst: ns.inst,
						captures: new_captures,
					})),
					// Re-fetch the value for the original to preserve it
					self.index_to_value(ty),
				)
			},
			Type::Enum(ns) => {
				let new_captures: Vec<Capture> = ns
					.captures
					.iter()
					.map(|cap| match cap {
						Capture::Comptime(cap) => Capture::Comptime(self.substitute_poisons(*cap, bindings)),
						Capture::Runtime(cap) => Capture::Runtime(*cap),
					})
					.collect();
				let new_captures = self.alloc_slice(&new_captures);
				self.intern_non_trivial(
					&Key::Type(Type::Enum(NamespaceType {
						inst: ns.inst,
						captures: new_captures,
					})),
					self.index_to_value(ty),
				)
			},
			Type::Union(ns) => {
				let new_captures: Vec<Capture> = ns
					.captures
					.iter()
					.map(|cap| match cap {
						Capture::Comptime(cap) => Capture::Comptime(self.substitute_poisons(*cap, bindings)),
						Capture::Runtime(cap) => Capture::Runtime(*cap),
					})
					.collect();
				let new_captures = self.alloc_slice(&new_captures);
				self.intern_non_trivial(
					&Key::Type(Type::Union(NamespaceType {
						inst: ns.inst,
						captures: new_captures,
					})),
					self.index_to_value(ty),
				)
			},
			Type::Int { .. }
			| Type::Anyint
			| Type::Anyfloat
			| Type::Usize
			| Type::Isize
			| Type::F16
			| Type::F32
			| Type::F64
			| Type::F128
			| Type::Bool
			| Type::Void
			| Type::NullPtr
			| Type::Any
			| Type::Anyptr
			| Type::GenericPoison
			| Type::Type
			| Type::Never
			| Type::EnumLiteral => ty,
		}
	}

	/// Allocate a slice in the bump allocator, returning a &'static reference.
	///
	/// # Safety
	/// The bump allocator outlives all uses of the returned slice.
	pub fn alloc_slice<T: Copy>(
		&self,
		slice: &[T],
	) -> &'static [T] {
		let bump = self.payload_bump_alloc.lock();
		let allocated = bump.alloc_slice_copy(slice);
		// SAFETY: Payload allocations live for the lifetime of the value store.
		unsafe { &*(allocated as *const [T]) }
	}

	/// Allocate a BitVec in the bump allocator, returning a &'static BitSlice.
	pub fn alloc_bitslice(
		&self,
		bitvec: &BitVec<u8>,
	) -> &'static bitvec::slice::BitSlice<u8> {
		let raw = bitvec.as_raw_slice();
		let bump = self.payload_bump_alloc.lock();
		let allocated = bump.alloc_slice_copy(raw);
		let len = bitvec.len();

		let bitslice = bitvec::slice::BitSlice::<u8>::from_slice(allocated);
		// SAFETY: The backing allocation lives for the lifetime of the value store.
		unsafe { core::mem::transmute(bitslice) }
	}

	pub fn alloc_slice_fill_iter<I>(
		&self,
		iter: I,
	) -> &'static [I::Item]
	where
		I: Iterator + ExactSizeIterator,
	{
		let bump = self.payload_bump_alloc.lock();
		let slice = bump.alloc_slice_fill_iter(iter);
		// SAFETY: Payload allocations live for the lifetime of the value store.
		unsafe { core::mem::transmute(slice) }
	}

	#[inline]
	pub fn value_allocate<T>(
		&self,
		val: T,
	) -> ValueStoreBumpRef<T> {
		let bump = self.payload_bump_alloc.lock();
		// SAFETY: The bump allocation is non-null and lives with the value store.
		unsafe { ValueStoreBumpRef::new(bump.alloc(val)) }
	}

	/// Get the type of an interned value by its index.
	pub fn type_of_interned(
		&self,
		i: Index,
	) -> Index {
		let key = self.index_to_key(i);
		match key {
			Key::Type(_) | Key::GenericPoison { .. } => self.common.type_t,
			Key::Ptr(ptr) => ptr.ty,
			Key::Fn(fun) => fun.ty,
			Key::Int { ty, .. } => *ty,
			Key::Float { ty, .. } => *ty,
			Key::Str { slice_ty: ty, .. } => *ty,
			Key::Bool(_) => self.common.bool_t,
			Key::FnDecl(decl) => decl.ty,
			Key::EnumTag { enum_ty: ty, .. } => *ty,
			Key::Undefined { ty } => *ty,
			Key::Aggregate { ty, .. } => *ty,
			Key::Void => self.common.void_t,
			Key::Unreachable => self.common.never_t,
			Key::Union { ty, .. } => *ty,
			Key::EnumLiteral(..) => self.common.enum_literal_t,
			Key::NullPtr => self.common.nullptr_t,
			Key::DeclRef { .. } => {
				unreachable!("value {} is untyped", self.display_index(i))
			},
		}
	}

	pub fn type_is_comptime_only(
		&self,
		ty: Index,
	) -> bool {
		let (key, value) = self.value_map.kv(ty);
		let Key::Type(ty_key) = key else {
			unreachable!("not a type: {}", self.display_index(ty))
		};
		match ty_key {
			// comptime only
			Type::Any | Type::Anyint | Type::Anyfloat | Type::Type | Type::EnumLiteral | Type::NullPtr => true,

			// comptime if inner types is
			Type::Struct(_) => {
				let r#struct = value.load(Ordering::Relaxed).as_struct();
				let r#struct = r#struct.as_ref();
				for field in r#struct.fields {
					if self.type_is_comptime_only(field.ty) {
						return true;
					}
				}

				false
			},
			Type::Ptr(ptr) => self.type_is_comptime_only(ptr.pointee_ty),
			Type::Slice(slice) => self.type_is_comptime_only(slice.pointee_ty),
			Type::Array(arr) => self.type_is_comptime_only(arr.elem_ty),
			Type::Enum(_) => {
				let r#enum = value.load(Ordering::Relaxed).as_enum();
				let r#enum = r#enum.as_ref();
				self.type_is_comptime_only(r#enum.tag_ty)
			},
			Type::Union(_) => {
				let u = value.load(Ordering::Relaxed).as_union();
				let u = u.as_ref();
				for field in u.fields {
					if let Some(ty) = field.ty
						&& self.type_is_comptime_only(ty)
					{
						return true;
					}
				}
				false
			},

			// comptime and runtime
			Type::Bool
			| Type::Anyptr
			| Type::Int { .. }
			| Type::Fn(..)
			| Type::Void
			| Type::Never
			| Type::Usize
			| Type::Isize
			| Type::F16
			| Type::F32
			| Type::F64
			| Type::F128 => false,

			// cannot be determined at all and the compiler should never ask for a generic poison
			Type::GenericPoison => unreachable!(),
		}
	}

	pub fn type_is_linear(
		&self,
		ty: Index,
	) -> bool {
		let (key, value) = self.value_map.kv(ty);
		let Key::Type(ty_key) = key else {
			unreachable!("not a type: {}", self.display_index(ty))
		};
		match ty_key {
			Type::Struct(_) => value.load(Ordering::Relaxed).as_struct().as_ref().linear,
			Type::Enum(_) => value.load(Ordering::Relaxed).as_enum().as_ref().linear,
			Type::Union(_) => value.load(Ordering::Relaxed).as_union().as_ref().linear,
			Type::Int { .. }
			| Type::Anyint
			| Type::Anyfloat
			| Type::Usize
			| Type::Isize
			| Type::F16
			| Type::F32
			| Type::F64
			| Type::F128
			| Type::Bool
			| Type::Void
			| Type::Fn(_)
			| Type::Ptr(_)
			| Type::Slice(_)
			| Type::Array(_)
			| Type::NullPtr
			| Type::Any
			| Type::Anyptr
			| Type::GenericPoison
			| Type::Type
			| Type::Never
			| Type::EnumLiteral => false,
		}
	}

	pub fn type_bit_size(
		&self,
		ty: Index,
	) -> u32 {
		let (Key::Type(ty_key), value) = self.index_to_key_value(ty) else {
			unreachable!("not a type: {}", self.display_index(ty))
		};
		match ty_key {
			Type::Int { bits, .. } => *bits as _,
			Type::Struct(_) => {
				let Value::Struct(r#struct) = value else {
					unreachable!("struct type without struct value")
				};
				if let StructLayout::Packed { storage_bits, .. } = r#struct.as_ref().layout {
					storage_bits
				} else {
					unreachable!()
				}
			},
			Type::Bool => 1,
			Type::Anyint
			| Type::Anyfloat
			| Type::Usize
			| Type::Isize
			| Type::F16
			| Type::F32
			| Type::F64
			| Type::F128
			| Type::Void
			| Type::Enum(_)
			| Type::Union(_)
			| Type::Fn(_)
			| Type::Ptr(_)
			| Type::Slice(_)
			| Type::Array(_)
			| Type::NullPtr
			| Type::Any
			| Type::Anyptr
			| Type::GenericPoison
			| Type::Type
			| Type::Never
			| Type::EnumLiteral => unreachable!("{ty_key:?} has no direct bit size"),
		}
	}

	pub fn type_layout(
		&self,
		target: &ResolvedTargetInfo,
		ty: Index,
	) -> Layout {
		// TODO(zino): proper ABI management of size ofs with target
		let (Key::Type(ty_key), value) = self.index_to_key_value(ty) else {
			unreachable!("not a type: {}", self.display_index(ty))
		};
		match ty_key {
			Type::Bool => Layout { size: 1, align: 1 },
			Type::Int { bits, .. } => {
				// TODO(zino): revisit
				let size = u64::from(*bits).div_ceil(8);
				Layout {
					size,
					align: size.next_power_of_two(),
				}
			},
			Type::F16 => Layout { size: 2, align: 2 },
			Type::F32 => Layout { size: 4, align: 4 },
			Type::F64 => Layout { size: 8, align: 8 },
			Type::F128 => Layout { size: 16, align: 16 },
			Type::Ptr(_) | Type::Usize | Type::Isize => self.type_ptr_layout(target, ty),
			Type::Array(array) => {
				let elem = self.type_layout(target, array.elem_ty);
				Layout {
					size: elem.size.next_multiple_of(elem.align) * array.len,
					align: elem.align,
				}
			},
			Type::Slice(_) => {
				// TODO(zino): slice as ptr
				let ptr_size = target.ptr_width_in_bits.div_exact(8).unwrap() as u64;
				Layout {
					// slice + len (which is ptr length since usize)
					size: ptr_size * 2,
					align: ptr_size,
				}
			},
			Type::Anyptr => {
				let ptr_size = target.ptr_width_in_bits.div_exact(8).unwrap() as u64;
				// pointer + compiler-internal type id (usize) TODO(zino)
				Layout {
					size: ptr_size * 2,
					align: ptr_size,
				}
			},
			Type::Struct(..) => {
				let Value::Struct(r#struct) = value else {
					unreachable!("struct type without struct value")
				};
				match &r#struct.layout {
					StructLayout::Standard => {
						let mut size = 0u64;
						let mut align = 1u64;
						for field in r#struct.fields {
							let field = self.type_layout(target, field.ty);
							size = size.next_multiple_of(field.align);
							size += field.size;
							align = align.max(field.align);
						}
						Layout {
							size: size.next_multiple_of(align),
							align,
						}
					},
					StructLayout::Packed { storage_bits, .. } => Layout {
						size: u64::from(*storage_bits).div_ceil(8),
						align: 1,
					},
				}
			},
			Type::Enum(..) => {
				let Value::Enum(r#enum) = value else {
					unreachable!("enum type without enum value")
				};
				self.type_layout(target, r#enum.tag_ty)
			},
			Type::Union(..) => {
				let Value::Union(_) = value else {
					unreachable!("union type without union value")
				};
				// TODO(zino): this can become a bottleneck, we could cache it in the union
				self.type_union_layout(target, ty).union_layout
			},
			Type::Void => Layout { size: 0, align: 0 },
			Type::Anyint
			| Type::Anyfloat
			| Type::Fn(_)
			| Type::NullPtr
			| Type::Any
			| Type::GenericPoison
			| Type::Type
			| Type::Never
			| Type::EnumLiteral => {
				unreachable!("{ty_key:?} has no runtime layout")
			},
		}
	}

	pub fn type_ptr_layout(
		&self,
		target: &ResolvedTargetInfo,
		ty: Index,
	) -> Layout {
		let size = target.ptr_width_in_bits.div_exact(8).unwrap() as _;
		Layout { size, align: size }
	}

	pub fn type_union_layout(
		&self,
		target: &ResolvedTargetInfo,
		ty: Index,
	) -> UnionLayout {
		let r#union = self.index_to_value(ty).as_union();
		let mut payload_size = 0;
		let mut payload_align = 1;
		let mut most_aligned_field = 0;
		let mut most_aligned_field_layout = Layout { size: 0, align: 0 };
		for (i, field) in union.fields.iter().enumerate().filter(|(_, field)| field.ty.is_some()) {
			let field_ty = field.ty.unwrap();
			let field_ty_layout = self.type_layout(target, field_ty);
			if field_ty_layout.size == 0 {
				continue;
			}
			payload_size = payload_size.max(field_ty_layout.size);
			if field_ty_layout.align >= payload_align {
				payload_align = field_ty_layout.align;
				most_aligned_field = i;
				most_aligned_field_layout = field_ty_layout;
			}
		}

		let tag = union
			.tag_ty
			.map(|tag_ty| self.type_layout(target, tag_ty))
			.unwrap_or(Layout::zeroed());
		let abi_align = tag.align.max(payload_align);
		let content_end = if tag.size == 0 {
			payload_size
		} else if payload_size == 0 {
			tag.size
		} else if tag.align >= payload_align {
			tag.size.next_multiple_of(payload_align) + payload_size
		} else {
			payload_size.next_multiple_of(tag.align) + tag.size
		};
		let abi_size = content_end.next_multiple_of(abi_align);

		UnionLayout {
			union_layout: Layout {
				size: abi_size,
				align: abi_align,
			},
			payload: Layout {
				size: payload_size,
				align: payload_align,
			},
			most_aligned_field: (most_aligned_field, most_aligned_field_layout),
			tag,
			trailing_padding: abi_size - content_end,
		}
	}

	pub fn type_is_int_signed(
		&self,
		ty: Index,
	) -> bool {
		let Key::Type(ty_key) = self.index_to_key(ty) else {
			unreachable!("not a type: {}", self.display_index(ty))
		};
		match ty_key {
			Type::Int { signed, .. } => *signed,
			Type::Usize => false,
			Type::Isize => true,
			Type::Anyint
			| Type::Anyfloat
			| Type::F16
			| Type::F32
			| Type::F64
			| Type::F128
			| Type::Bool
			| Type::Void
			| Type::Struct(_)
			| Type::Enum(_)
			| Type::Union(_)
			| Type::Fn(_)
			| Type::Ptr(_)
			| Type::Slice(_)
			| Type::Array(_)
			| Type::NullPtr
			| Type::Any
			| Type::Anyptr
			| Type::GenericPoison
			| Type::Type
			| Type::Never
			| Type::EnumLiteral => unreachable!("{ty_key:?} is not an integer type"),
		}
	}
}

pub enum TypeComptimeReason {
	StructFieldIsComptime(),
}

#[derive(Debug)]
pub struct CommonValues {
	pub nullptr: Index,
	pub anyint_t: Index,
	pub anyfloat_t: Index,
	pub void_t: Index,
	pub any_t: Index,
	pub anyptr_t: Index,
	pub type_t: Index,
	pub never_t: Index,
	pub generic_poison_t: Index,
	pub usize_t: Index,
	pub isize_t: Index,
	pub u16_t: Index,
	pub u64_t: Index,
	pub i64_t: Index,
	pub u32_t: Index,
	pub i32_t: Index,
	pub f16_t: Index,
	pub f32_t: Index,
	pub f64_t: Index,
	pub f128_t: Index,
	pub bool_t: Index,
	pub enum_literal_t: Index,
	pub nullptr_t: Index,
	pub unreachable_value: Index,
	pub void_value: Index,
	pub true_value: Index,
	pub false_value: Index,
}
