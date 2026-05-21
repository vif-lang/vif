use std::hash::{
	BuildHasher,
	Hash,
};

use hashbrown::HashTable;

mod entry;

pub use entry::*;

use crate::common::IndexVec;

pub struct IndexMap<K, V, H>
where
	K: Hash + PartialEq,
	H: BuildHasher,
{
	hash_table: HashTable<usize>,
	slots: IndexVec<usize, (K, V)>,
	build_hasher: H,
}
impl<K, V, H> IndexMap<K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	pub fn new(build_hasher: H) -> Self {
		Self {
			hash_table: Default::default(),
			slots: Default::default(),
			build_hasher,
		}
	}

	pub fn insert(
		&mut self,
		key: K,
		value: V,
	) -> (usize, Option<V>) {
		let (entry, prev) = self.entry(&key).insert_entry(value);
		(entry.index, prev)
	}

	/// Obtain a `entry` from `key`. If a vacant entry is returned, the entry will lock the sharded index map hash table until it is dropped / consumed.
	pub fn entry<'a>(
		&'a mut self,
		key: &'a K,
	) -> Entry<'a, K, V, H> {
		let hash = self.build_hasher.hash_one(key);
		let index = self.hash_table.find(hash, |idx| &self.slots[idx].0 == key).copied();
		match index {
			Some(index) => Entry::Occupied(OccupiedEntry { map: self, index }),
			None => Entry::Vacant(VacantEntry { map: self, key, hash }),
		}
	}

	pub fn find(
		&self,
		key: &K,
	) -> Option<usize> {
		let hash = self.build_hasher.hash_one(key);
		self.hash_table.find(hash, |idx| &self.slots[idx].0 == key).copied()
	}

	/// Returns the key at `index`
	///
	/// # Panics
	///
	/// Panics if `index` is not a valid Index returned by this IndexMap
	pub fn key(
		&self,
		index: usize,
	) -> &K {
		&self.slots[index].0
	}

	pub fn kv(
		&self,
		index: usize,
	) -> &(K, V) {
		&self.slots[index]
	}

	pub fn kv_mut(
		&mut self,
		index: usize,
	) -> &mut (K, V) {
		&mut self.slots[index]
	}

	pub fn kvs(&self) -> &[(K, V)] {
		&self.slots.vec
	}
}
