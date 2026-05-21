use std::ops::{
	AddAssign,
	Sub,
};

use crate::assume;

#[rustc_layout_scalar_valid_range_end(0xfffffffe)]
#[derive(Copy, Clone, Ord, PartialOrd, Debug)]
pub struct NonMaxU32(u32);

impl NonMaxU32 {
	#[inline(always)]
	pub const fn from_u32(index: u32) -> Self {
		assume!(index <= 0xfffffffe);
		// SAFETY: we assume the index is within the valid range
		unsafe { NonMaxU32(index) }
	}

	#[inline(always)]
	pub const fn as_usize(self) -> usize {
		self.0 as usize
	}

	#[inline(always)]
	pub const fn as_u32(self) -> u32 {
		self.0
	}
}

impl Default for NonMaxU32 {
	#[inline(always)]
	fn default() -> Self {
		NonMaxU32::from_u32(0)
	}
}

impl core::hash::Hash for NonMaxU32 {
	#[inline(always)]
	fn hash<H: core::hash::Hasher>(
		&self,
		state: &mut H,
	) {
		state.write_u32(self.0);
	}
}

impl PartialEq for NonMaxU32 {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.0 == other.0
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		self.0 != other.0
	}
}

impl Eq for NonMaxU32 {}

impl From<u32> for NonMaxU32 {
	#[inline(always)]
	fn from(value: u32) -> Self {
		NonMaxU32::from_u32(value)
	}
}

impl From<usize> for NonMaxU32 {
	#[inline(always)]
	fn from(value: usize) -> Self {
		NonMaxU32::from_u32(value as u32)
	}
}

impl From<NonMaxU32> for u32 {
	#[inline(always)]
	fn from(value: NonMaxU32) -> Self {
		value.as_u32()
	}
}

impl From<NonMaxU32> for usize {
	#[inline(always)]
	fn from(value: NonMaxU32) -> Self {
		value.as_usize()
	}
}

impl AddAssign<u32> for NonMaxU32 {
	#[inline(always)]
	fn add_assign(
		&mut self,
		rhs: u32,
	) {
		assume!(self.as_u32() + rhs < u32::MAX, "Addition overflowed NonMaxU32");

		// SAFETY: we assert after the addition assert_ne!
		unsafe { self.0 += rhs };
	}
}

impl Sub<NonMaxU32> for NonMaxU32 {
	type Output = Self;

	#[inline(always)]
	fn sub(
		self,
		rhs: NonMaxU32,
	) -> Self::Output {
		Self::from_u32(self.0 - rhs.0)
	}
}

impl core::fmt::Display for NonMaxU32 {
	#[inline(always)]
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter<'_>,
	) -> core::fmt::Result {
		self.0.fmt(f)
	}
}
