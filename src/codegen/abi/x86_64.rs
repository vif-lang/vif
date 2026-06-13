use crate::{
	codegen::abi::Repr,
	compile_unit::CompilationUnit,
	value,
};

pub fn compute_type_abi_win64(
	cu: &CompilationUnit,
	ty: value::Index,
) -> Repr {
	match cu.values.index_to_key(ty) {
		value::Key::TypePtr(..)
		| value::Key::TypeInt { .. }
		| value::Key::TypeBool
		| value::Key::TypeEnum(..)
		| value::Key::TypeUsize
		| value::Key::TypeIsize
		| value::Key::TypeF16
		| value::Key::TypeF32
		| value::Key::TypeF64
		| value::Key::TypeVoid => Repr::ByValue,

		value::Key::TypeArray(..) | value::Key::TypeSlice(..) | value::Key::TypeStruct(..) | value::Key::TypeUnion(..) => {
			let layout = cu.values.type_layout(&cu.resolved_target, ty);
			match layout.size {
				1 | 2 | 4 | 8 => Repr::AsInteger,
				_ => Repr::ByRef,
			}
		},
		value::Key::TypeF128 => Repr::ByRef,

		_ => unreachable!("cannot lower abi of {}", cu.values.display_index(ty)),
	}
}
