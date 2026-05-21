use std::ptr::NonNull;

use libc::{
    c_void,
    mmap,
    mprotect,
    munmap,
    MAP_ANON,
    MAP_FAILED,
    MAP_PRIVATE,
    PROT_NONE,
    PROT_READ,
    PROT_WRITE,
};

pub fn virtmem_reserve(size_in_bytes: usize) -> NonNull<[u8]> {
    assert!(size_in_bytes > 0);

    let ptr = unsafe {
        mmap(
            std::ptr::null_mut(),
            size_in_bytes,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0,
        )
    };

    assert_ne!(ptr, MAP_FAILED);

    unsafe {
        NonNull::slice_from_raw_parts(
            NonNull::new_unchecked(ptr as *mut u8),
            size_in_bytes,
        )
    }
}

pub fn virtmem_commit(range: NonNull<[u8]>) {
    let ptr = range.as_ptr() as *mut u8;
    let len = range.len();

    assert!(len > 0);

    let result = unsafe {
        mprotect(
            ptr as *mut c_void,
            len,
            PROT_READ | PROT_WRITE,
        )
    };

    assert_eq!(result, 0);
}

pub fn virtmem_free(alloc: NonNull<[u8]>) {
    let ptr = alloc.as_ptr() as *mut u8;
    let len = alloc.len();

    assert!(len > 0);

    let result = unsafe {
        munmap(ptr as *mut c_void, len)
    };

    assert_eq!(result, 0);
}
