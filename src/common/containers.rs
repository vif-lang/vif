use core::slice::GetDisjointMutIndex;
use std::{
	marker::PhantomData,
	ops::{
		Index,
		IndexMut,
		Range,
	},
	slice::SliceIndex,
};

pub mod index_map;
pub mod sharded_index_map;
pub mod virtual_dyn_array;

/// Vec with a custom index type
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	vec: Vec<T>,
	_p: PhantomData<I>,
}

impl<I, T> IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	#[inline(always)]
	pub fn new() -> Self {
		Self {
			vec: Vec::new(),
			_p: PhantomData,
		}
	}

	#[inline(always)]
	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			vec: Vec::with_capacity(capacity),
			_p: PhantomData,
		}
	}

	#[inline(always)]
	pub unsafe fn get_disjoint_unchecked_mut<const N: usize>(
		&mut self,
		indices: [I; N],
	) -> [&mut <usize as SliceIndex<[T]>>::Output; N] {
		unsafe { self.vec.get_disjoint_unchecked_mut(indices.map(|i| i.into())) }
	}

	#[inline(always)]
	pub fn get(
		&self,
		index: I,
	) -> Option<&T> {
		self.vec.get(index.into())
	}

	pub fn push(
		&mut self,
		elem: T,
	) -> I {
		self.vec.push(elem);
		I::from(self.vec.len() - 1)
	}

	pub fn pop(&mut self) -> Option<T> {
		self.vec.pop()
	}

	pub fn remove(
		&mut self,
		index: I,
	) -> T {
		self.vec.remove(index.into())
	}

	pub fn last(&self) -> Option<&T> {
		self.vec.last()
	}

	pub fn len(&self) -> usize {
		self.vec.len()
	}

	pub fn is_empty(&self) -> bool {
		self.vec.is_empty()
	}

	pub fn iter(&self) -> impl Iterator<Item = &T> {
		self.vec.iter()
	}
}

impl<I, T> Default for IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	fn default() -> Self {
		Self {
			vec: Vec::default(),
			_p: PhantomData,
		}
	}
}

impl<I, T> Index<I> for IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	type Output = T;

	fn index(
		&self,
		index: I,
	) -> &Self::Output {
		&self.vec[index.into()]
	}
}

impl<I, T> IndexMut<I> for IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	fn index_mut(
		&mut self,
		index: I,
	) -> &mut Self::Output {
		&mut self.vec[index.into()]
	}
}

impl<I, T> Index<&I> for IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	type Output = T;

	fn index(
		&self,
		index: &I,
	) -> &Self::Output {
		let index = *index;
		&self.vec[index.into()]
	}
}

impl<I, T> Index<Range<I>> for IndexVec<I, T>
where
	I: Copy + From<usize> + Into<usize>,
{
	type Output = [T];

	fn index(
		&self,
		index: Range<I>,
	) -> &Self::Output {
		let start = index.start.into();
		let end = index.end.into();
		&self.vec[start..end]
	}
}
