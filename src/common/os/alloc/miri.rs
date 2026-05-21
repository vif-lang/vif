use std::{
	alloc::Layout,
	ptr::NonNull,
};

const ALIGN: usize = 4096;

pub fn virtmem_reserve(size_in_bytes: usize) -> NonNull<[u8]> {
	// SAFETY: we uphold the function invariants
	let ptr = unsafe { std::alloc::alloc(Layout::from_size_align_unchecked(size_in_bytes, ALIGN)) };
	assert!(!ptr.is_null());

	// SAFETY: we assert that ptr is non-null
	unsafe { NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr as *mut _), size_in_bytes) }
}

pub fn virtmem_commit(range: NonNull<[u8]>) {}

pub fn virtmem_free(alloc: NonNull<[u8]>) {
	// SAFETY: we uphold the function invariants
	unsafe {
		std::alloc::dealloc(
			alloc.as_ptr() as *mut _,
			Layout::from_size_align_unchecked(alloc.len(), ALIGN),
		)
	}
}
