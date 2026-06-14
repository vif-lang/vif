use crate::{
	codegen::abi::{
		Context,
		Repr,
	},
	compile_unit::CompilationUnit,
	value,
};

pub fn compute_type_abi_darwin(
	cu: &CompilationUnit,
	ty: value::Index,
	context: Context,
) -> Repr {
	let value::Key::Type(ty_key) = cu.values.index_to_key(ty) else {
		unreachable!("cannot lower ABI of non-type {}", cu.values.display_index(ty))
	};
	// TODO
	Repr::ByValue
}
