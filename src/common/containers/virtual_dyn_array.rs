use std::{
	alloc::Layout,
	marker::PhantomData,
	ops::Index,
	ptr::NonNull,
	sync::{
		atomic::{
			AtomicUsize,
			Ordering,
		},
		nonpoison::Mutex,
	},
};

/// A typed arena allocator with a dynarray-like interface backed by virtual memory that is made to be used in concurrent contexts for long-lived data.
///
/// **Deletions are not supported.**
///
/// This data structure is made for specific usecases such as long-lived big arrays
/// where the overhead of having allocating in the OS page granularity is outweigthed by the benefits. Else
/// prefer to use a [`Vec`] or [`LockFreeDynArray`]
///
/// It provides:
/// - Lock-free wait-free reads
/// - Lock-free, with waits, writes
/// - Stable pointers / references thanks to virtual memory
/// - Compared to [`LockFreeDynArray`] no temporary memory leaks in contentions
#[derive(Debug)]
pub struct VirtMemArenaDynArray<I, T>
where
	I: From<usize> + Into<usize>,
{
	reserved: NonNull<[T]>,

	/// Number of elements
	len: AtomicUsize,

	/// Number of initialized elements
	initialized: AtomicUsize,

	/// Committed elements in virtual memory
	committed_elements: AtomicUsize,

	_ph: PhantomData<(I, T)>,
}

// SAFETY: no race condition can happen
unsafe impl<I, T> Send for VirtMemArenaDynArray<I, T> where I: From<usize> + Into<usize> {}
// SAFETY: no race condition can happen
unsafe impl<I, T> Sync for VirtMemArenaDynArray<I, T> where I: From<usize> + Into<usize> {}

impl<I, T> VirtMemArenaDynArray<I, T>
where
	I: From<usize> + Into<usize>,
{
	/// The commit granularity, note that the actual data structure round to the element size
	const COMMIT_GRANULARITY_IN_BYTES: usize = 65_536; // 64 Kio is reasonable enough
	const COMMIT_GRANULARITY_IN_ELEMENTS: usize = {
		let elem_size = std::mem::size_of::<T>();

		Self::COMMIT_GRANULARITY_IN_BYTES.div_ceil(elem_size)
	};

	pub fn with_capacity(capacity: usize) -> Self {
		const {
			assert!(size_of::<T>() > 0, "VirtMemArenaDynArray doesn't support zero-sized types");
		}

		let total_size_in_bytes = capacity
			.checked_mul(size_of::<T>())
			.expect("cannot store {capacity} elements, it overflows memory");

		let data_ptr = crate::common::os::alloc::virtmem_reserve(total_size_in_bytes);
		assert!(
			data_ptr.is_aligned_to(Layout::new::<T>().align()),
			"virtmem_reserve returned a pointer that does not satisfy T alignment"
		);

		Self {
			reserved: NonNull::slice_from_raw_parts(data_ptr.as_non_null_ptr().cast::<T>(), capacity),
			len: AtomicUsize::new(0),
			initialized: AtomicUsize::new(0),
			committed_elements: AtomicUsize::new(0),
			_ph: Default::default(),
		}
	}

	/// Push a element into the arena.
	///
	/// Returns the index to the element, which is guaranteed to be initialized when returned.
	pub fn push(
		&self,
		value: T,
	) -> I {
		let index = self.len.fetch_add(1, Ordering::Relaxed);
		assert!(
			index < self.reserved.len(),
			"overflowing capacity of {} elements",
			self.reserved.len()
		);

		self.ensure_committed_for_index(index);

		// SAFETY: we ensure beforehand that the index is not OOB
		let ptr = unsafe { self.elem_ptr(index) };

		// SAFETY:
		// - `ensure_committed` commit the pages covered by the ptr, making it valid for writes
		// - we ensure that the virtual alloc is properly aligned
		unsafe {
			ptr.write(value);
		}

		// CAS spin-lock to ensure initialized is properly incremented for safety
		loop {
			if self
				.initialized
				.compare_exchange_weak(index, index + 1, Ordering::Release, Ordering::Relaxed)
				.is_ok()
			{
				break;
			}
		}

		index.into()
	}

	/// Get the element at `index`, without performing any out-of-bounds initialized checks.
	///
	/// # Safety
	///
	/// - `index` must points to a index returned by `push` or a computed one that is known to reference a valid, initialized `T`
	pub unsafe fn get_unchecked(
		&self,
		index: I,
	) -> &T {
		let index = index.into();
		debug_assert!(
			index < self.initialized.load(Ordering::Relaxed),
			"index {index} >= initialized {}",
			self.reserved.len()
		);

		// SAFETY: the underlying memory is guarenteed to be commited and initialized by the caller
		unsafe { self.reserved.cast::<T>().add(index).as_ref() }
	}

	/// Get the element at `index`.
	///
	/// # Panics
	///
	/// Panics if `index` is out of bounds.
	pub fn get(
		&self,
		index: I,
	) -> &T {
		let index = index.into();
		assert!(
			index < self.initialized.load(Ordering::Relaxed),
			"index {index} >= initialized {}",
			self.reserved.len()
		);

		// SAFETY: the underlying memory is guarenteed to be commited and initialized by the assertion
		unsafe { self.reserved.cast::<T>().add(index).as_ref() }
	}

	fn ensure_committed_for_index(
		&self,
		index: usize,
	) {
		if index < self.committed_elements.load(Ordering::Acquire) {
			return;
		}

		// need to commit mem, we could here either lock or do a CAS spinlock to ensure one
		// commit is done but as virtmem_commit is idempotent on platforms we support we just don't care
		let new_committed = ((index / Self::COMMIT_GRANULARITY_IN_ELEMENTS) + 1)
			.checked_mul(Self::COMMIT_GRANULARITY_IN_ELEMENTS)
			.unwrap()
			.min(self.reserved.len());

		let current_committed = self.committed_elements.load(Ordering::Relaxed);
		let elements_to_commit = new_committed - current_committed;
		// SAFETY: pointer stays within the reserved region by construction.
		let ptr = unsafe { self.reserved.as_non_null_ptr().add(current_committed).cast::<u8>() };
		crate::common::os::alloc::virtmem_commit(NonNull::slice_from_raw_parts(ptr, elements_to_commit * std::mem::size_of::<T>()));
		self.committed_elements.store(new_committed, Ordering::Release);
	}

	/// # Safety
	///
	/// - index must less than `self.capacity`
	#[inline]
	unsafe fn elem_ptr(
		&self,
		index: usize,
	) -> *mut T {
		// SAFETY: caller ensure safety invariants
		unsafe { self.reserved.as_ptr().cast::<T>().add(index) }
	}
}

impl<I, T> Drop for VirtMemArenaDynArray<I, T>
where
	I: From<usize> + Into<usize>,
{
	fn drop(&mut self) {
		const {
			// NOTE(zino): `needs_drop` is conservative. It may return true even when
			// dropping would be a no-op, but we still reject those types here.
			assert!(!std::mem::needs_drop::<T>());
		}

		crate::common::os::alloc::virtmem_free(NonNull::slice_from_raw_parts(
			self.reserved.as_non_null_ptr().cast::<u8>(),
			self.reserved.len() * std::mem::size_of::<T>(),
		));
	}
}

impl<I, T> Index<I> for VirtMemArenaDynArray<I, T>
where
	I: From<usize> + Into<usize>,
{
	type Output = T;

	#[inline]
	fn index(
		&self,
		index: I,
	) -> &Self::Output {
		self.get(index)
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;

	// -----------------------------------------------------------------------
	// Helper index type: a newtype over usize to exercise the I generic.
	// -----------------------------------------------------------------------

	#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
	struct Idx(usize);

	impl From<usize> for Idx {
		fn from(v: usize) -> Self {
			Idx(v)
		}
	}

	impl From<Idx> for usize {
		fn from(v: Idx) -> usize {
			v.0
		}
	}

	// -----------------------------------------------------------------------
	// Basic correctness
	// -----------------------------------------------------------------------

	#[test]
	fn push_returns_sequential_indices_single_thread() {
		let arena: VirtMemArenaDynArray<Idx, u32> = VirtMemArenaDynArray::with_capacity(64);
		for i in 0..10u32 {
			let idx = arena.push(i);
			assert_eq!(idx, Idx(i as usize));
		}
	}

	#[test]
	fn pushed_values_are_readable_through_elem_ptr() {
		let arena: VirtMemArenaDynArray<Idx, u64> = VirtMemArenaDynArray::with_capacity(64);
		let values = [10u64, 20, 30, 40, 50];
		for &v in &values {
			arena.push(v);
		}
		for (i, &expected) in values.iter().enumerate() {
			// SAFETY: index < capacity and value was pushed
			let got = unsafe { arena.elem_ptr(i).read() };
			assert_eq!(got, expected);
		}
	}

	#[test]
	fn push_across_commit_boundary() {
		// Force multiple commit cycles by pushing more than one granularity's worth of elements.
		// With u8 and COMMIT_GRANULARITY_BYTES=65536, one granularity = 65536 elements.
		// Use a larger T to make the boundary cheaper to cross.
		#[derive(Copy, Clone, PartialEq, Debug)]
		struct Big([u64; 8]); // 64 bytes

		// 65536 / 64 = 1024 elements per granularity. Push 2.5x that.
		const N: usize = 2560;
		let arena: VirtMemArenaDynArray<Idx, Big> = VirtMemArenaDynArray::with_capacity(N + 1);

		for i in 0..N {
			let v = Big([i as u64; 8]);
			let idx = arena.push(v);
			assert_eq!(idx.0, i);
		}

		for i in 0..N {
			// SAFETY: The index was initialized by the preceding pushes.
			let got = unsafe { arena.elem_ptr(i).read() };
			assert_eq!(got, Big([i as u64; 8]));
		}
	}

	// -----------------------------------------------------------------------
	// Index type round-trip
	// -----------------------------------------------------------------------

	#[test]
	fn usize_index_type_works() {
		let arena: VirtMemArenaDynArray<usize, f32> = VirtMemArenaDynArray::with_capacity(16);
		let idx: usize = arena.push(1.0f32);
		assert_eq!(idx, 0usize);
	}

	#[test]
	fn custom_index_type_round_trips() {
		let arena: VirtMemArenaDynArray<Idx, u32> = VirtMemArenaDynArray::with_capacity(16);
		let idx = arena.push(42u32);
		let raw: usize = idx.into();
		let back = Idx::from(raw);
		assert_eq!(back, Idx(0));
	}

	// -----------------------------------------------------------------------
	// Alignment
	// -----------------------------------------------------------------------

	#[test]
	fn base_pointer_is_aligned_for_u8() {
		let arena: VirtMemArenaDynArray<Idx, u8> = VirtMemArenaDynArray::with_capacity(128);
		let ptr = arena.reserved.as_ptr().cast::<u8>();
		assert_eq!(ptr as usize % std::mem::align_of::<u8>(), 0);
	}

	#[test]
	fn base_pointer_is_aligned_for_u64() {
		let arena: VirtMemArenaDynArray<Idx, u64> = VirtMemArenaDynArray::with_capacity(128);
		let ptr = arena.reserved.as_ptr().cast::<u8>();
		assert_eq!(ptr as usize % std::mem::align_of::<u64>(), 0);
	}

	#[test]
	fn base_pointer_is_aligned_for_simd_like_type() {
		// 32-byte aligned type (typical for AVX2 vectors)
		#[repr(C, align(32))]
		#[derive(Copy, Clone)]
		struct Avx2([f32; 8]);

		let arena: VirtMemArenaDynArray<Idx, Avx2> = VirtMemArenaDynArray::with_capacity(64);
		let ptr = arena.reserved.as_ptr().cast::<u8>();
		assert_eq!(ptr as usize % std::mem::align_of::<Avx2>(), 0);
	}

	// -----------------------------------------------------------------------
	// Capacity enforcement
	// -----------------------------------------------------------------------

	#[test]
	#[should_panic]
	fn push_beyond_capacity_panics() {
		let arena: VirtMemArenaDynArray<Idx, u32> = VirtMemArenaDynArray::with_capacity(2);
		arena.push(1u32);
		arena.push(2u32);
		arena.push(3u32); // must panic
	}

	#[test]
	fn push_exactly_at_capacity_does_not_panic() {
		let arena: VirtMemArenaDynArray<Idx, u32> = VirtMemArenaDynArray::with_capacity(4);
		for i in 0..4u32 {
			arena.push(i); // must not panic
		}
	}

	// -----------------------------------------------------------------------
	// Concurrency: unique index assignment
	// -----------------------------------------------------------------------

	#[test]
	fn concurrent_pushes_assign_unique_indices() {
		const THREADS: usize = 8;
		const PER_THREAD: usize = 10_000;
		const TOTAL: usize = THREADS * PER_THREAD;

		let arena = Arc::new(VirtMemArenaDynArray::<Idx, u64>::with_capacity(TOTAL));

		// Each thread records the indices it received.
		let handles: Vec<_> = (0..THREADS)
			.map(|t| {
				let arena = Arc::clone(&arena);
				std::thread::spawn(move || {
					(0..PER_THREAD)
						.map(|i| arena.push((t * PER_THREAD + i) as u64).0)
						.collect::<Vec<usize>>()
				})
			})
			.collect();

		let mut all_indices: Vec<usize> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();

		all_indices.sort_unstable();
		all_indices.dedup();

		assert_eq!(all_indices.len(), TOTAL, "indices must all be unique");
		assert_eq!(all_indices[0], 0);
		assert_eq!(all_indices[TOTAL - 1], TOTAL - 1);
	}

	#[test]
	fn concurrent_pushes_values_are_correct_after_join() {
		const THREADS: usize = 4;
		const PER_THREAD: usize = 4_096;
		const TOTAL: usize = THREADS * PER_THREAD;

		let arena = Arc::new(VirtMemArenaDynArray::<Idx, u64>::with_capacity(TOTAL));

		std::thread::scope(|s| {
			for t in 0..THREADS {
				let arena = &arena;
				s.spawn(move || {
					for i in 0..PER_THREAD {
						let value = (t * PER_THREAD + i) as u64;
						let idx = arena.push(value);
						// Each thread verifies its own write immediately.
						// SAFETY: `idx` was returned by the push immediately above.
						let readback = unsafe { arena.elem_ptr(idx.0).read() };
						assert_eq!(readback, value);
					}
				});
			}
		});

		// After join: all TOTAL slots were written; verify none are zero-initialised
		// collisions by checking the full set of values is present.
		let mut values: Vec<u64> = (0..TOTAL)
			.map(|i| {
				// SAFETY: All indices in this range were initialized by the scoped threads.
				unsafe { arena.elem_ptr(i).read() }
			})
			.collect();
		values.sort_unstable();
		values.dedup();
		assert_eq!(values.len(), TOTAL);
	}

	// -----------------------------------------------------------------------
	// committed_elements grows monotonically
	// -----------------------------------------------------------------------

	#[test]
	fn committed_elements_never_decreases() {
		use std::sync::atomic::Ordering;

		const N: usize = 4_096;
		let arena = Arc::new(VirtMemArenaDynArray::<Idx, u32>::with_capacity(N));
		let prev = Arc::new(AtomicUsize::new(0));

		std::thread::scope(|s| {
			for _ in 0..4 {
				let arena = &arena;
				let prev = Arc::clone(&prev);
				s.spawn(move || {
					for _ in 0..N / 4 {
						arena.push(0u32);
						let c = arena.committed_elements.load(Ordering::Acquire);
						let p = prev.load(Ordering::Acquire);
						assert!(c >= p, "committed_elements went backwards: {c} < {p}");
						prev.fetch_max(c, Ordering::Release);
					}
				});
			}
		});
	}

	// -----------------------------------------------------------------------
	// Different Copy types
	// -----------------------------------------------------------------------

	#[test]
	fn works_with_bool() {
		let arena: VirtMemArenaDynArray<Idx, bool> = VirtMemArenaDynArray::with_capacity(8);
		arena.push(true);
		arena.push(false);
		// SAFETY: Indices 0 and 1 were initialized by the pushes above.
		assert!(unsafe { arena.elem_ptr(0).read() });
		// SAFETY: Indices 0 and 1 were initialized by the pushes above.
		assert!(!unsafe { arena.elem_ptr(1).read() });
	}

	#[test]
	fn works_with_f64() {
		let arena: VirtMemArenaDynArray<Idx, f64> = VirtMemArenaDynArray::with_capacity(8);
		arena.push(std::f64::consts::PI);
		// SAFETY: Index 0 was initialized by the push above.
		let v = unsafe { arena.elem_ptr(0).read() };
		assert!((v - std::f64::consts::PI).abs() < f64::EPSILON);
	}

	#[test]
	fn works_with_tuple() {
		let arena: VirtMemArenaDynArray<Idx, (u32, u32)> = VirtMemArenaDynArray::with_capacity(4);
		arena.push((1, 2));
		arena.push((3, 4));
		// SAFETY: Indices 0 and 1 were initialized by the pushes above.
		assert_eq!(unsafe { arena.elem_ptr(0).read() }, (1, 2));
		// SAFETY: Indices 0 and 1 were initialized by the pushes above.
		assert_eq!(unsafe { arena.elem_ptr(1).read() }, (3, 4));
	}
}
