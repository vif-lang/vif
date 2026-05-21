use core::hint::unlikely;
use std::ops::Range;

use crate::common::non_max_u32::NonMaxU32;

/// A region in the source code.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Span {
	start: NonMaxU32,
	end: NonMaxU32,
}

impl Span {
	#[inline(always)]
	pub fn new(range: Range<usize>) -> Span {
		Span {
			start: range.start.into(),
			end: range.end.into(),
		}
	}

	#[inline(always)]
	pub fn start(&self) -> usize {
		self.start.as_usize()
	}

	#[inline(always)]
	pub fn end(&self) -> usize {
		self.end.as_usize()
	}

	pub fn len(&self) -> usize {
		self.end.as_usize() - self.start.as_usize()
	}

	pub fn start_line_col(
		&self,
		str: &str,
	) -> (usize, usize) {
		self.line_col(str, self.start())
	}

	pub fn end_line_col(
		&self,
		str: &str,
	) -> (usize, usize) {
		self.line_col(str, self.end())
	}

	fn line_col(
		&self,
		str: &str,
		offset: usize,
	) -> (usize, usize) {
		let bytes = str.as_bytes();
		let mut slice_offset = 0;

		let mut line = 1;

		loop {
			let start = slice_offset;
			match memx::memchr(&bytes[slice_offset..offset], b'\n') {
				Some(pos) => {
					slice_offset += pos + 1;
					line += 1;
				},
				None => {
					break (line, offset - start);
				},
			}
		}
	}
}

impl Default for Span {
	#[inline(always)]
	fn default() -> Self {
		Span {
			start: NonMaxU32::from_u32(0),
			end: NonMaxU32::from_u32(0),
		}
	}
}

impl core::fmt::Debug for Span {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter,
	) -> core::fmt::Result {
		f.debug_struct("Span")
			.field("start", &self.start.as_u32())
			.field("end", &self.end.as_u32())
			.finish()
	}
}

impl core::fmt::Display for Span {
	fn fmt(
		&self,
		f: &mut core::fmt::Formatter,
	) -> core::fmt::Result {
		write!(f, "{}..{}", self.start.as_u32(), self.end.as_u32())
	}
}

impl core::ops::Index<Span> for str {
	type Output = str;

	#[inline(always)]
	fn index(
		&self,
		index: Span,
	) -> &Self::Output {
		&self[index.start.as_usize()..index.end.as_usize()]
	}
}

impl core::ops::IndexMut<Span> for str {
	#[inline(always)]
	fn index_mut(
		&mut self,
		index: Span,
	) -> &mut Self::Output {
		&mut self[index.start.as_usize()..index.end.as_usize()]
	}
}

impl core::ops::Index<Span> for [u8] {
	type Output = [u8];

	#[inline(always)]
	fn index(
		&self,
		index: Span,
	) -> &Self::Output {
		&self[index.start.as_usize()..index.end.as_usize()]
	}
}

impl core::ops::IndexMut<Span> for [u8] {
	#[inline(always)]
	fn index_mut(
		&mut self,
		index: Span,
	) -> &mut Self::Output {
		&mut self[index.start.as_usize()..index.end.as_usize()]
	}
}

impl From<(Span, Span)> for Span {
	#[inline(always)]
	fn from((lhs, rhs): (Span, Span)) -> Self {
		let start = if unlikely(rhs.start < lhs.start) { rhs.start } else { lhs.start };
		let end = if unlikely(lhs.end > rhs.end) { lhs.end } else { rhs.end };

		Span { start, end }
	}
}
