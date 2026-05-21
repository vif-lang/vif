use std::sync::{
	Arc,
	OnceLock,
	nonpoison::Mutex,
};

use relative_path::RelativePathBuf;

use crate::{
	common::diagnostic::{
		DiagSpan,
		Diagnostic,
	},
	compile_unit::NamespaceId,
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
	pub first_imported_by: Option<DiagSpan>,
	pub source: OnceLock<String>,
	pub vuir: OnceLock<Vuir>,
	pub namespace: OnceLock<NamespaceId>,
	pub sema_state: Mutex<ModuleAnalyzeState>,
	pub diagnostics: Mutex<Vec<Diagnostic>>,
}

impl Module {
	pub fn new(
		path: RelativePathBuf,
		first_imported_by: Option<DiagSpan>,
	) -> Self {
		Self {
			path,
			first_imported_by,
			source: Default::default(),
			vuir: Default::default(),
			namespace: Default::default(),
			sema_state: Default::default(),
			diagnostics: Default::default(),
		}
	}
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
