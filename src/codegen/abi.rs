pub mod aarch64;
pub mod x86_64;

use target_lexicon::{
	Architecture,
	OperatingSystem,
};

use crate::{
	compile_unit::CompilationUnit,
	value,
};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Repr {
	/// Passed as-is
	ByValue,
	/// Passed by pointer
	ByRef,
	/// Represented as an integer that can store the type
	AsInteger,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Context {
	Param,
	Return,
}

pub fn compute_type_abi_win64(
	cu: &CompilationUnit,
	ty: value::Index,
	context: Context,
) -> Repr {
	x86_64::compute_type_abi_win64(cu, ty, context)
}

pub fn compute_type_abi_c(
	cu: &CompilationUnit,
	ty: value::Index,
	context: Context,
) -> Repr {
	match (cu.resolved_target.triple.architecture, cu.resolved_target.triple.operating_system) {
		(Architecture::X86_64, OperatingSystem::Windows) => x86_64::compute_type_abi_win64(cu, ty, context),
		(Architecture::X86_64, OperatingSystem::Linux) => x86_64::compute_type_abi_sysv(cu, ty),
		(Architecture::Aarch64(_), OperatingSystem::Darwin(_)) => aarch64::compute_type_abi_darwin(cu, ty, context),
		(architecture, operating_system) => {
			unimplemented!("C ABI lowering is not implemented for {architecture}-{operating_system}")
		},
	}
}
