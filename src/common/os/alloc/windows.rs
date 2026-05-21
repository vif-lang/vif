use std::ptr::NonNull;

use windows::Win32::System::Memory::{
	MEM_COMMIT,
	MEM_RELEASE,
	MEM_RESERVE,
	PAGE_READWRITE,
	VirtualAlloc,
	VirtualFree,
};

pub fn virtmem_reserve(size_in_bytes: usize) -> NonNull<[u8]> {
	// SAFETY: we uphold the function invariants
	let ptr = unsafe { VirtualAlloc(None, size_in_bytes, MEM_RESERVE, PAGE_READWRITE) };
	assert!(!ptr.is_null());

	// SAFETY: we assert that ptr is non-null
	unsafe { NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr as *mut _), size_in_bytes) }
}

pub fn virtmem_commit(range: NonNull<[u8]>) {
	// SAFETY: we uphold the function invariants
	unsafe {
		VirtualAlloc(
			Some(range.as_ptr() as *const _),
			range.len(),
			MEM_COMMIT,
			PAGE_READWRITE,
		)
	};
}

pub fn virtmem_free(alloc: NonNull<[u8]>) {
	// SAFETY: we uphold the function invariants
	unsafe { VirtualFree(alloc.as_ptr() as *mut _, 0, MEM_RELEASE) }.unwrap()
}
