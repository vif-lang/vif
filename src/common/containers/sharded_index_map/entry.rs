use std::{
	hash::{
		BuildHasher,
		Hash,
	},
	intrinsics::AtomicOrdering,
	sync::nonpoison::MutexGuard,
};

use bytemuck::NoUninit;
use hashbrown::HashTable;

use crate::common::sharded_index_map::{
	Index,
	ShardLocalIndex,
	ShardedIndexMap,
};

pub enum Entry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	Occupied(OccupiedEntry<'map, K, V, H>),
	Vacant(VacantEntry<'map, K, V, H>),
}
impl<'map, K, V, H> Entry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub fn insert_entry(
		self,
		value: V,
	) -> (OccupiedEntry<'map, K, V, H>, Option<V>) {
		match self {
			Self::Occupied(entry) => {
				// SAFETY:
				// - we only use the reference to swap the previous value and we are the sole writer of the shard thanks
				// to the mutex guard `entry.table` protecting the hashtable
				// - the index is valid as the hash table is guaranteed to contains only valid indices
				let kv = unsafe { entry.map.kv_unchecked(entry.index) };
				let prev = kv.1.swap(value, std::sync::atomic::Ordering::Release);
				(entry, Some(prev))
			},
			Self::Vacant(mut entry) => {
				let map = entry.map;
				let index = entry.insert_by_ref(value);
				let table = entry.table;
				(OccupiedEntry { map, table, index }, None)
			},
		}
	}

	pub fn or_insert_with<F>(
		self,
		default: F,
	) -> Index
	where
		F: FnOnce() -> V,
	{
		match self {
			Self::Occupied(entry) => entry.index,
			Self::Vacant(entry) => entry.insert(default()),
		}
	}
}

pub struct OccupiedEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub(super) map: &'map ShardedIndexMap<K, V, H>,
	pub(super) table: MutexGuard<'map, HashTable<ShardLocalIndex>>,
	pub(super) index: Index,
}
impl<'map, K, V, H> OccupiedEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub fn insert(
		mut self,
		value: V,
	) -> (Index, V) {
		let old = self.map.shards[self.index.shard()].slots[self.index.local_index()]
			.1
			.swap(value, std::sync::atomic::Ordering::Release);
		(self.index, old)
	}
}

pub struct VacantEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub(super) map: &'map ShardedIndexMap<K, V, H>,
	pub(super) table: MutexGuard<'map, HashTable<ShardLocalIndex>>,
	pub(super) shard_idx: usize,
	pub(super) hash: u64,
	pub(super) key: &'map K,
}
impl<'map, K, V, H> VacantEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	V: NoUninit,
	H: BuildHasher,
{
	pub fn insert(
		mut self,
		value: V,
	) -> Index {
		self.insert_by_ref(value)
	}

	fn insert_by_ref(
		&mut self,
		value: V,
	) -> Index {
		let shard = &self.map.shards[self.shard_idx];
		let local_index = shard.slots.push((*self.key, atomic::Atomic::new(value)));
		self.table.insert_unique(self.hash, local_index, |index| {
			// SAFETY: index is returned by the previous shard.slots.push call
			let key = unsafe { &shard.slots.get_unchecked(*index).0 };
			self.map.build_hasher.hash_one(key)
		});

		// SAFETY: local_index points to a valid, initialized element inside the shard slots
		// as it is returned from a push() call
		unsafe { Index::new(self.shard_idx, local_index) }
	}
}
