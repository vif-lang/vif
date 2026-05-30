use std::{
	alloc::{
		Allocator,
		Global,
	},
	collections::HashMap,
	fmt::Debug,
	hash::{
		BuildHasher,
		Hash,
		Hasher,
	},
	ops::Deref,
	sync::nonpoison::Mutex,
};

use atomic::Atomic;
use bytemuck::NoUninit;
use hashbrown::HashTable;
use rustc_hash::FxHashMap;

use crate::common::{
	NonMaxU32,
	virtual_dyn_array::VirtMemArenaDynArray,
};

mod entry;

pub use entry::*;

/// Index into a InternMap.
///
/// # Representation
///
/// Encoded as:
/// - chunk id (6 bits)
/// - chunk local index (26 bits)
///
/// # Validity
///
/// A [`Index`] is considered valid when it points to a initialized/push key-value pair inside a valid shard
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(transparent)]
pub struct Index(u32);
impl Index {
	pub const NONE: Index = Index(u32::MAX);

	const SHARD_BITS: u32 = 6;
	const SHARD_SHIFT: u32 = 32 - Self::SHARD_BITS;
	const LOCAL_MASK: u32 = (1 << Self::SHARD_SHIFT) - 1;

	/// # Safety
	///
	/// - `local_index` must point to a valid, initialized element of the shard slots
	#[inline(always)]
	unsafe fn new(
		shard: usize,
		local_index: ShardLocalIndex,
	) -> Self {
		debug_assert!(shard < (1 << Self::SHARD_BITS));
		debug_assert!(local_index.0 <= Self::LOCAL_MASK);
		Self(((shard as u32) << Self::SHARD_SHIFT) | local_index.0)
	}

	#[inline(always)]
	fn shard(self) -> usize {
		(self.0 >> Self::SHARD_SHIFT) as usize
	}

	#[inline(always)]
	fn local_index(self) -> ShardLocalIndex {
		ShardLocalIndex(self.0 & Self::LOCAL_MASK)
	}

	#[inline(always)]
	fn is_none(self) -> bool {
		self.0 == u32::MAX
	}

	#[inline(always)]
	pub const fn as_u32(self) -> u32 {
		self.0
	}
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
struct ShardLocalIndex(u32);
impl From<usize> for ShardLocalIndex {
	fn from(value: usize) -> Self {
		assert!(value <= u32::MAX as usize);
		Self(value as u32)
	}
}

impl From<ShardLocalIndex> for usize {
	fn from(val: ShardLocalIndex) -> Self {
		val.0 as _
	}
}

struct Shard<K, V>
where
	K: Copy,
	V: NoUninit,
{
	key_to_index: Mutex<HashTable<ShardLocalIndex>>,
	slots: VirtMemArenaDynArray<ShardLocalIndex, (K, Atomic<V>)>,
}

/// A concurrent index map made for small (8 bytes) values.
///
/// # Implementation
///
/// The index map is cut in multiple shards, configurable at construction time.
/// Each shard store a [`HashTable`] wrapped in a Mutex and a [`VirtMemArenaDynArray`] to store the key-value pair.
pub struct ShardedIndexMap<K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	shards: Vec<Shard<K, V>>,
	build_hasher: H,
}

impl<K, V, H> ShardedIndexMap<K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher + Default,
{
	pub fn new(
		shard_count: usize,
		key_value_pair_capacity: usize,
		hasher: H,
	) -> Self {
		assert!(shard_count.is_power_of_two());
		Self {
			shards: (0..shard_count)
				.map(|_| Shard {
					key_to_index: Mutex::default(),
					slots: VirtMemArenaDynArray::with_capacity(key_value_pair_capacity),
				})
				.collect::<Vec<_>>(),
			build_hasher: hasher,
		}
	}
}

impl<K, V, H> ShardedIndexMap<K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub fn insert(
		&self,
		key: K,
		value: V,
	) -> (Index, Option<V>) {
		let (entry, prev) = self.entry(&key).insert_entry(value);
		(entry.index, prev)
	}

	/// Obtain a `entry` from `key`. If a vacant entry is returned, the entry will lock the sharded index map hash table until it is dropped / consumed.
	pub fn entry<'a>(
		&'a self,
		key: &'a K,
	) -> Entry<'a, K, V, H> {
		let (hash, shard_idx) = self.hash_and_shard(key);
		let shard = &self.shards[shard_idx];
		let mut table = shard.key_to_index.lock();

		let index = table
			.find(hash, |idx| {
				// SAFETY: indices inside the hash table were returned by push() calls, therefore they reference
				// valid, initialized elements
				unsafe { &shard.slots.get_unchecked(*idx).0 == key }
			})
			.map(|local_idx| {
				// SAFETY: indices inside the hash table were returned by push() calls, therefore they reference
				// valid, initialized elements
				unsafe { Index::new(shard_idx, *local_idx) }
			});

		match index {
			Some(index) => Entry::Occupied(OccupiedEntry { map: self, table, index }),
			None => Entry::Vacant(VacantEntry {
				map: self,
				table,
				key,
				shard_idx,
				hash,
			}),
		}
	}

	pub fn find(
		&self,
		key: &K,
	) -> Option<Index> {
		let (hash, shard_idx) = self.hash_and_shard(key);
		let shard = &self.shards[shard_idx];
		let mut table = shard.key_to_index.lock();
		table
			.find(hash, |idx| {
				// SAFETY: indices inside the hash table were returned by push() calls, therefore they reference
				// valid, initialized elements
				unsafe { &shard.slots.get_unchecked(*idx).0 == key }
			})
			.map(|local_idx| {
				// SAFETY: indices inside the hash table were returned by push() calls, therefore they reference
				// valid, initialized elements
				unsafe { Index::new(shard_idx, *local_idx) }
			})
	}

	/// Returns the key at `index`
	///
	/// # Safety
	///
	/// - `index` must be a valid Index returned by this IndexMap
	pub unsafe fn key_unchecked(
		&self,
		index: Index,
	) -> &K {
		let shard = {
			let shard_idx = index.shard();

			// SAFETY: a ShardedIndexMap Index guarantees that the shard index inside it is valid
			unsafe { self.shards.get_unchecked(shard_idx) }
		};

		// SAFETY: a ShardedIndexMap Index guarantees that the local index points to a valid, initialized element
		unsafe { &shard.slots.get_unchecked(index.local_index()).0 }
	}

	/// Returns the key at `index`
	///
	/// # Panics
	///
	/// Panics if `index` is not a valid Index returned by this IndexMap
	pub fn key(
		&self,
		index: Index,
	) -> &K {
		let shard = {
			let shard_idx = index.shard();
			&self.shards[shard_idx]
		};

		// SAFETY: a ShardedIndexMap Index guarantees that the local index points to a valid, initialized element
		unsafe { &shard.slots.get(index.local_index()).0 }
	}

	/// Returns the (key, value) pair at `index`
	///
	/// # Panics
	///
	/// Panics if `index` is not a valid Index returned by this IndexMap
	pub fn kv(
		&self,
		index: Index,
	) -> &(K, Atomic<V>) {
		let shard = {
			let shard_idx = index.shard();
			&self.shards[shard_idx]
		};

		// SAFETY: a ShardedIndexMap Index guarantees that the local index points to a valid, initialized element
		unsafe { shard.slots.get(index.local_index()) }
	}

	/// Returns the (key, value) pair at `index`
	///
	/// # Safety
	///
	/// - `index` must be a valid Index returned by this IndexMap
	pub unsafe fn kv_unchecked(
		&self,
		index: Index,
	) -> &(K, Atomic<V>) {
		let shard = {
			let shard_idx = index.shard();
			&self.shards[shard_idx]
		};

		// SAFETY: a ShardedIndexMap Index guarantees that the local index points to a valid, initialized element
		unsafe { shard.slots.get_unchecked(index.local_index()) }
	}

	#[inline]
	fn hash_and_shard(
		&self,
		key: &K,
	) -> (u64, usize) {
		let hash = self.build_hasher.hash_one(key);
		let shard = (hash as usize) & (self.shards.len() - 1);
		(hash, shard)
	}
}

#[cfg(test)]
mod tests {
	use rustc_hash::FxBuildHasher;

	use super::*;

	type TestMap<K, V> = ShardedIndexMap<K, V, FxBuildHasher>;

	fn new_map<K: Hash + PartialEq + Copy, V: NoUninit>(shard_count: usize) -> TestMap<K, V> {
		ShardedIndexMap::new(shard_count, 1000, FxBuildHasher)
	}

	// ── Index encoding ────────────────────────────────────────────────────────

	#[test]
	fn index_roundtrip() {
		for shard in 0..(1 << Index::SHARD_BITS) {
			let local = ShardLocalIndex(42);

			// SAFETY: we don't do any r/w operations from a index map
			let idx = unsafe { Index::new(shard, local) };
			assert_eq!(idx.shard(), shard);
			assert_eq!(idx.local_index(), local);
		}
	}

	#[test]
	fn index_none_is_sentinel() {
		assert!(Index::NONE.is_none());

		// A freshly constructed index should never be NONE.
		// SAFETY: we don't do any r/w operations from a index map
		let idx = unsafe { Index::new(0, ShardLocalIndex(0)) };
		assert!(!idx.is_none());
	}

	#[test]
	fn index_max_local() {
		// The largest valid local index should still round-trip correctly.
		let max_local = ShardLocalIndex(Index::LOCAL_MASK);

		// SAFETY: we don't do any r/w operations from a index map
		let idx = unsafe { Index::new(0, max_local) };
		assert_eq!(idx.local_index(), max_local);
	}

	// ── Basic insertion & retrieval ───────────────────────────────────────────

	#[test]
	fn insert_and_get_key() {
		let map = new_map::<&'static str, u32>(4);
		let idx = map.insert("hello", 1).0;
		// SAFETY: `idx` was returned by this map and still points to the inserted key.
		let k = unsafe { map.key_unchecked(idx) };
		assert_eq!(k, &"hello");
	}

	#[test]
	fn insert_returns_same_index_for_same_key() {
		let map = new_map::<&'static str, u32>(4);
		let i1 = map.insert("dup", 10).0;
		let i2 = map.insert("dup", 99).0; // value ignored on collision
		assert_eq!(i1, i2);
	}

	#[test]
	fn distinct_keys_get_distinct_indices() {
		let map = new_map::<&'static str, u32>(4);
		let i1 = map.insert("a", 1).0;
		let i2 = map.insert("b", 2).0;
		assert_ne!(i1, i2);
	}

	#[test]
	fn index_encodes_correct_shard() {
		let shard_count = 16;
		let map = new_map::<&'static str, u32>(shard_count);
		let key = &"shard_check";
		let idx = map.insert(key, 0).0;

		let (hash, expected_shard) = map.hash_and_shard(key);
		assert_eq!(idx.shard(), expected_shard);
	}

	// ── Many insertions ───────────────────────────────────────────────────────

	#[test]
	fn many_keys_all_unique_indices() {
		let map = new_map::<u64, u8>(8);
		let mut indices = std::collections::HashSet::new();
		for i in 0..1_000u64 {
			let idx = map.insert(i, 0).0;
			assert!(indices.insert(idx.as_u32()), "duplicate index for key {i}");
		}
	}

	#[test]
	fn many_keys_are_idempotent() {
		let map = new_map::<u64, u64>(8);
		for i in 0..1_000u64 {
			map.insert(i, i);
		}
		// Second pass: same keys must return same indices and correct keys.
		for i in 0..1_000u64 {
			let idx = map.insert(i, 0xdead).0;
			// SAFETY: `idx` was returned by this map during the same iteration.
			let k = unsafe { map.key_unchecked(idx) };
			assert_eq!(*k, i);
		}
	}

	// ── Single-shard edge cases ───────────────────────────────────────────────

	#[test]
	fn single_shard() {
		// shard_count = 1 means everything lands in shard 0.
		let map = new_map::<u32, u32>(1);
		for i in 0..100u32 {
			map.insert(i, i * 2);
		}
		for i in 0..100u32 {
			let idx = map.insert(i, 0).0;
			// SAFETY: `idx` was returned by this map during the same iteration.
			let k = unsafe { map.key_unchecked(idx) };
			assert_eq!(*k, i);
		}
	}

	// ── Concurrent reads and writes ───────────────────────────────────────────

	#[test]
	fn concurrent_insert_and_read() {
		use std::{
			sync::Arc,
			thread,
		};

		let map = Arc::new(new_map::<u64, u64>(16));

		// Pre-populate so readers have something to look at immediately.
		for i in 0..256u64 {
			map.insert(i, i);
		}

		let writers: Vec<_> = (0..4)
			.map(|t| {
				let map = map.clone();
				thread::spawn(move || {
					for i in 0..500u64 {
						map.insert((t * 10_000 + i), i);
					}
				})
			})
			.collect();

		let readers: Vec<_> = (0..4)
			.map(|_| {
				let map = map.clone();
				thread::spawn(move || {
					for i in 0..256u64 {
						let idx = map.insert(i, i).0;
						// SAFETY: `idx` was returned by this map during the same iteration.
						let k = unsafe { map.key_unchecked(idx) };
						assert_eq!(*k, i);
					}
				})
			})
			.collect();

		for w in writers {
			w.join().unwrap();
		}
		for r in readers {
			r.join().unwrap();
		}
	}

	#[test]
	fn concurrent_insert_deduplication() {
		use std::{
			sync::Arc,
			thread,
		};

		// Multiple threads insert the same keys. They must all get the same index back.
		let map = Arc::new(new_map::<u64, u64>(8));
		let handles: Vec<_> = (0..8)
			.map(|_| {
				let map = map.clone();
				thread::spawn(move || (0..200u64).map(|i| map.insert(i, i).0).collect::<Vec<_>>())
			})
			.collect();

		let results: Vec<Vec<Index>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

		// Every thread must have observed the exact same index for each key.
		for i in 0..200u64 {
			let expected_index = map.insert(i, i).0;
			for thread_result in &results[0..] {
				assert_eq!(thread_result[i as usize], expected_index, "index mismatch for key {i}");
			}
		}
	}
}
