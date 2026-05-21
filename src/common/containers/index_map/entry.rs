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

use crate::common::index_map::IndexMap;

pub enum Entry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	Occupied(OccupiedEntry<'map, K, V, H>),
	Vacant(VacantEntry<'map, K, V, H>),
}
impl<'map, K, V, H> Entry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	pub fn insert_entry(
		self,
		value: V,
	) -> (OccupiedEntry<'map, K, V, H>, Option<V>) {
		match self {
			Self::Occupied(entry) => {
				let kv = entry.map.kv_mut(entry.index);
				let prev = std::mem::replace(&mut kv.1, value);
				(entry, Some(prev))
			},
			Self::Vacant(mut entry) => {
				let index = entry.insert_by_ref(value);
				(OccupiedEntry { map: entry.map, index }, None)
			},
		}
	}

	pub fn or_insert_with<F>(
		self,
		default: F,
	) -> usize
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
	H: BuildHasher,
{
	pub(super) map: &'map mut IndexMap<K, V, H>,
	pub(super) index: usize,
}
impl<'map, K, V, H> OccupiedEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	pub fn insert(
		mut self,
		value: V,
	) -> (usize, Option<V>) {
		todo!();
	}
}

pub struct VacantEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	pub(super) map: &'map mut IndexMap<K, V, H>,
	pub(super) hash: u64,
	pub(super) key: &'map K,
}
impl<'map, K, V, H> VacantEntry<'map, K, V, H>
where
	K: Hash + PartialEq + Copy,
	H: BuildHasher,
{
	pub fn insert(
		mut self,
		value: V,
	) -> usize {
		self.insert_by_ref(value)
	}

	fn insert_by_ref(
		&mut self,
		value: V,
	) -> usize {
		let index = self.map.slots.push((*self.key, value));
		self.map.hash_table.insert_unique(self.hash, index, |index| {
			let key = &self.map.slots[index].0;
			self.map.build_hasher.hash_one(key)
		});
		index
	}
}
