use std::{
	cell::UnsafeCell,
	path::PathBuf,
	sync::{
		Arc,
		OnceLock,
		nonpoison::Mutex,
	},
};

use internment::Intern;
use relative_path::RelativePathBuf;
use rustc_hash::FxHashMap;

use crate::{
	common::diagnostic::Diagnostic,
	compile_unit::{
		DeclId,
		NamespaceId,
	},
	frontend::ast,
	ir::vuir::Vuir,
	value,
};

#[derive(Clone, Debug, Default)]
pub enum ModuleAnalyzeState {
	#[default]
	Pending,
	InProgress,
	Done(value::Index),
	Failed,
}

/// A single .vif file
#[derive(Debug)]
pub struct Module {
	pub path: RelativePathBuf,
	pub content: OnceLock<std::io::Result<String>>,
	pub ast: OnceLock<Result<ast::Module, Vec<Diagnostic>>>,
	pub vuir: OnceLock<Result<Vuir, Vec<Diagnostic>>>,
	pub namespace: OnceLock<NamespaceId>,
	pub analyze: Mutex<ModuleAnalyzeState>,
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct ModuleId(usize);
impl From<ModuleId> for usize {
	#[inline(always)]
	fn from(value: ModuleId) -> Self {
		value.0
	}
}
impl From<usize> for ModuleId {
	#[inline(always)]
	fn from(value: usize) -> Self {
		Self(value)
	}
}

pub type ArcModule = Arc<Module>;
