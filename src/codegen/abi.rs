mod x86_64;

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
	ByValue,
	ByRef,
	/// Represented as an integer that can store the type
	AsInteger,
}

pub fn compute_type_abi_win64(
	cu: &CompilationUnit,
	ty: value::Index,
) -> Repr {
	match (cu.resolved_target.triple.architecture, cu.resolved_target.triple.operating_system) {
		(Architecture::X86_64, OperatingSystem::Windows) => x86_64::compute_type_abi_win64(cu, ty),
		(architecture, operating_system) => {
			unimplemented!("C ABI lowering is not implemented for {architecture}-{operating_system}")
		},
	}
}
