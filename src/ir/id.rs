use std::{
	fmt::{
		Debug,
		Display,
	},
	hash::Hash,
	ops::Sub,
};

use crate::{
	common::NonMaxU32,
	value,
};

#[doc(hidden)]
pub trait IRMarker: Debug + Send + Sync + Copy + Clone + Eq + PartialEq + Hash + Default + 'static {}

/// A unique identifier for an instruction within an IR.
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct InstructionId<IR: IRMarker>(NonMaxU32, core::marker::PhantomData<IR>);

impl<IR: IRMarker> InstructionId<IR> {
	/// The instruction ID reserved for the file/module root.
	pub const FILE_MODULE: Self = Self(NonMaxU32::from_u32(0), core::marker::PhantomData);

	#[inline(always)]
	pub fn from_usize(value: usize) -> Self {
		Self(NonMaxU32::from_u32(value as u32), core::marker::PhantomData)
	}

	#[inline(always)]
	pub fn from_u32(value: NonMaxU32) -> Self {
		Self(value, core::marker::PhantomData)
	}

	#[inline(always)]
	pub fn into_ref(self) -> InstructionRef<IR> {
		InstructionRef::Instruction(self)
	}

	#[inline(always)]
	pub fn as_ref(self) -> InstructionRef<IR> {
		InstructionRef::Instruction(self)
	}
}

impl<IR: IRMarker> From<InstructionId<IR>> for usize {
	#[inline(always)]
	fn from(value: InstructionId<IR>) -> Self {
		value.0.as_usize()
	}
}

impl<IR: IRMarker> From<usize> for InstructionId<IR> {
	#[inline(always)]
	fn from(value: usize) -> Self {
		Self::from_usize(value)
	}
}

impl<IR: IRMarker> Debug for InstructionId<IR> {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "%{}", self.0)
	}
}

impl<IR: IRMarker> Display for InstructionId<IR> {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "%{}", self.0)
	}
}

impl<IR: IRMarker> Sub<u32> for InstructionId<IR> {
	type Output = Self;

	fn sub(
		self,
		rhs: u32,
	) -> Self::Output {
		let rhs = NonMaxU32::from_u32(rhs);
		Self::from_u32(self.0 - rhs)
	}
}

/// A reference to either a concrete instruction or a compile-time interned value.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum InstructionRef<IR: IRMarker> {
	/// Reference to an instruction by its ID.
	Instruction(InstructionId<IR>),
	/// Reference to a compile-time interned constant.
	Interned(value::Index),
}

impl<IR: IRMarker> InstructionRef<IR> {
	/// Returns the instruction ID if this is an `Instruction` variant.
	#[inline]
	pub fn as_id(self) -> Option<InstructionId<IR>> {
		match self {
			InstructionRef::Instruction(id) => Some(id),
			InstructionRef::Interned(_) => None,
		}
	}

	/// Returns the interned value if this is an `Interned` variant.
	#[inline]
	pub fn as_interned(self) -> value::Index {
		match self {
			InstructionRef::Interned(v) => v,
			InstructionRef::Instruction(id) => panic!("not an interned value, is instruction {id}"),
		}
	}

	#[inline]
	pub fn as_interned_opt(self) -> Option<value::Index> {
		match self {
			InstructionRef::Interned(v) => Some(v),
			InstructionRef::Instruction(id) => None,
		}
	}

	/// Returns `true` if this is an instruction reference.
	#[inline]
	pub fn is_instruction(&self) -> bool {
		matches!(self, InstructionRef::Instruction(_))
	}

	/// Returns `true` if this is an interned value reference.
	#[inline]
	pub fn is_interned(&self) -> bool {
		matches!(self, InstructionRef::Interned(_))
	}
}

impl<IR: IRMarker> From<InstructionId<IR>> for InstructionRef<IR> {
	#[inline(always)]
	fn from(id: InstructionId<IR>) -> Self {
		InstructionRef::Instruction(id)
	}
}

impl<IR: IRMarker> From<value::Index> for InstructionRef<IR> {
	#[inline(always)]
	fn from(v: value::Index) -> Self {
		InstructionRef::Interned(v)
	}
}
