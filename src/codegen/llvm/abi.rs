use inkwell::{
	AddressSpace,
	types::{
		AnyType,
		AnyTypeEnum,
		BasicMetadataTypeEnum,
		FunctionType,
	},
};

use crate::{
	codegen::llvm::Lowerer,
	common::sharded_index_map::Index,
	value::{
		CallingConvention,
		TypeFn,
	},
};

pub struct ArgAbi {}

#[derive(Eq, PartialEq)]
pub enum RetMode<'ctx> {
	Direct,

	/// Indirect through the first parameter
	SretFirstParam(AnyTypeEnum<'ctx>),
}

pub struct FnAbi<'ctx> {
	pub ret_mode: RetMode<'ctx>,
	pub params: Vec<BasicMetadataTypeEnum<'ctx>>,
	pub ret_ty: AnyTypeEnum<'ctx>,
}

/// Returns whether the function should return by reference or not
/// If it does, the first parameter of the function will contain a pointer to the ret value.
pub fn fn_returns_by_ref_in_first_param<'ctx>(
	lowerer: &mut Lowerer<'ctx>,
	fn_ty: &TypeFn,
) -> bool {
	let ret_ty = lowerer.lower_type(fn_ty.ret_ty);
	match ret_ty {
		// returning aggregates is a very bad idea in LLVM, it explodes its passes time
		// force the usage of sret + hidden fn arg instead
		inkwell::types::AnyTypeEnum::StructType(_) => true,
		_ => false,
	}
}

pub fn compute_fn_abi<'ctx>(
	lowerer: &mut Lowerer<'ctx>,
	fn_ty: &TypeFn,
) -> FnAbi<'ctx> {
	let ret_ty = lowerer.lower_type(fn_ty.ret_ty);
	let ret_mode = if fn_returns_by_ref_in_first_param(lowerer, fn_ty) {
		RetMode::SretFirstParam(ret_ty)
	} else {
		RetMode::Direct
	};

	let params = {
		let mut params = Vec::with_capacity(fn_ty.params.len());

		// if in sret mode, add the return type as first param
		if matches!(ret_mode, RetMode::SretFirstParam(_)) {
			let ret_ptr = lowerer.ctx.ptr_type(AddressSpace::default());
			params.push(ret_ptr.into());
		}

		// runtime function parameters
		for (i, declared_param) in fn_ty.params.iter().enumerate().filter(|(i, _)| !fn_ty.comptime_params[*i]) {
			let concrete_ty = if i < fn_ty.params.len() { fn_ty.params[i] } else { *declared_param };

			let llvm_ty = lowerer.lower_type(concrete_ty);

			// Win64 ABI: for `callconv(.c)` functions, struct params are lowered by size:
			//   <= 64 bits → integer register (i8/i16/i32/i64)
			//   >  64 bits → caller allocates, passes pointer (byval semantics)
			let ty: BasicMetadataTypeEnum = if fn_ty.callconv == Some(CallingConvention::C) {
				if let AnyTypeEnum::StructType(st) = llvm_ty {
					let bits = lowerer.target_machine.get_target_data().get_bit_size(&st);
					match bits {
						8 => lowerer.ctx.i8_type().into(),
						16 => lowerer.ctx.i16_type().into(),
						32 => lowerer.ctx.i32_type().into(),
						64 => lowerer.ctx.i64_type().into(),
						_ => lowerer.ctx.ptr_type(inkwell::AddressSpace::default()).into(),
					}
				} else {
					llvm_ty.try_into().unwrap()
				}
			} else {
				llvm_ty.try_into().unwrap()
			};

			params.push(ty);
		}

		params
	};

	FnAbi {
		params,
		ret_ty: match ret_mode {
			RetMode::Direct => ret_ty,
			RetMode::SretFirstParam(_) => lowerer.ctx.void_type().as_any_type_enum(),
		},
		ret_mode,
	}
}
