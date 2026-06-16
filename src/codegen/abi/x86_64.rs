use crate::{
	codegen::abi::{
		Context,
		Repr,
	},
	compile_unit::CompilationUnit,
	value,
};

pub fn compute_type_abi_win64(
	cu: &CompilationUnit,
	ty: value::Index,
	context: Context,
) -> Repr {
	let value::Key::Type(ty_key) = cu.values.index_to_key(ty) else {
		unreachable!("cannot lower ABI of non-type {}", cu.values.display_index(ty))
	};
	match ty_key {
		value::Type::Ptr(..)
		| value::Type::Int { .. }
		| value::Type::Bool
		| value::Type::Enum(..)
		| value::Type::Usize
		| value::Type::Isize
		| value::Type::F16
		| value::Type::F32
		| value::Type::F64
		| value::Type::Void => Repr::ByValue,

		value::Type::Array(..) | value::Type::Slice(..) | value::Type::Struct(..) | value::Type::Union(..) => {
			let layout = cu.values.type_layout(&cu.resolved_target, ty);
			match layout.size {
				1 | 2 | 4 | 8 => Repr::AsInteger,
				_ => Repr::ByRef,
			}
		},
		value::Type::F128 => match context {
			Context::Param => Repr::ByRef,
			Context::Return => Repr::ByValue,
		},

		value::Type::Anyint
		| value::Type::Anyfloat
		| value::Type::Fn(_)
		| value::Type::NullPtr
		| value::Type::Anyptr
		| value::Type::Type
		| value::Type::Never
		| value::Type::GenericPoison
		| value::Type::EnumLiteral => unreachable!("cannot lower ABI of {}", cu.values.display_index(ty)),
	}
}

pub fn compute_type_abi_sysv(
	cu: &CompilationUnit,
	ty: value::Index,
) -> Repr {
	let value::Key::Type(ty_key) = cu.values.index_to_key(ty) else {
		unreachable!("cannot lower ABI of non-type {}", cu.values.display_index(ty))
	};
	// TODO
	Repr::ByValue
}
