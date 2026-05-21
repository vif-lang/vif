use core::{
	alloc::{
		AllocError,
		Allocator,
		Layout,
	},
	pin::Pin,
	ptr::NonNull,
};
use std::rc::Rc;

#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct RcLinearAllocator(Pin<Rc<bumpalo::Bump>>);

impl RcLinearAllocator {
	#[inline(always)]
	pub fn new(bump: bumpalo::Bump) -> Self {
		Self(Rc::pin(bump))
	}
}

// SAFETY: It's okay
unsafe impl Allocator for RcLinearAllocator {
	#[inline(always)]
	fn allocate(
		&self,
		layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		let allocator = &*self.0;
		allocator.allocate(layout)
	}

	#[inline(always)]
	unsafe fn deallocate(
		&self,
		ptr: NonNull<u8>,
		layout: Layout,
	) {
		let allocator = &*self.0;

		// SAFETY: Caller must guarantees `ptr` was allocated by this allocator
		unsafe { allocator.deallocate(ptr, layout) }
	}
}
