mod abi;

use std::sync::OnceLock;

use inkwell::{
	AddressSpace,
	attributes::{
		Attribute,
		AttributeLoc,
	},
	basic_block::BasicBlock,
	builder::BuilderError,
	debug_info::{
		AsDIScope,
		DIFlags,
		DIFlagsConstants,
	},
	intrinsics::{
		self,
		Intrinsic,
	},
	llvm_sys::{
		self,
		support::LLVMParseCommandLineOptions,
		target_machine::*,
	},
	memory_buffer::MemoryBuffer,
	module::Linkage,
	passes::PassBuilderOptions,
	targets::{
		FileType,
		InitializationConfig,
		Target,
		TargetMachine,
		TargetTriple,
	},
	types::{
		AnyType,
		AnyTypeEnum,
		BasicMetadataTypeEnum,
		BasicType,
		BasicTypeEnum,
		StructType,
	},
	values::{
		AnyValue,
		AnyValueEnum,
		BasicMetadataValueEnum,
		BasicValue,
		BasicValueEnum,
		FunctionValue,
		IntValue,
	},
};
use rustc_hash::FxHashMap;

use crate::{
	Build,
	codegen::llvm::{
		self,
		abi::FnAbi,
	},
	compile_unit::{
		CompilationUnit,
		DeclAnalysisState,
		DeclId,
		module::ModuleId,
	},
	frontend::ast,
	ir::{
		id::*,
		vtir::{
			self,
			Vtir,
		},
		vuir::{
			self,
			Vuir,
		},
	},
	value::{
		self,
		CallingConvention,
		ValueStore,
	},
};

struct DebugInfoGen<'ctx> {
	di_ctx: DebugInfoCtx<'ctx>,
	di_file: inkwell::debug_info::DIFile<'ctx>,
	di_lexical_block_stack: Vec<inkwell::debug_info::DILexicalBlock<'ctx>>,
}

#[derive(Debug)]
struct LlvmAttributes {
	alwaysinline: Attribute,
	noinline: Attribute,
	nounwind: Attribute,
	noreturn: Attribute,
	willreturn: Attribute,
	mustprogress: Attribute,
	uwtable_sync: Attribute,
	sret: Attribute,
}

impl LlvmAttributes {
	fn new(ctx: &inkwell::context::Context) -> Self {
		macro_rules! enum_attr {
			($attr:literal) => {
				ctx.create_enum_attribute(Attribute::get_named_enum_kind_id($attr), 0)
			};

			($attr:literal, $val:literal) => {
				ctx.create_enum_attribute(Attribute::get_named_enum_kind_id($attr), $val)
			};
		}
		Self {
			alwaysinline: enum_attr!("alwaysinline"),
			noinline: enum_attr!("noinline"),
			nounwind: enum_attr!("nounwind"),
			noreturn: enum_attr!("noreturn"),
			willreturn: enum_attr!("willreturn"),
			mustprogress: enum_attr!("mustprogress"),
			uwtable_sync: enum_attr!("uwtable", 1),
			sret: enum_attr!("sret"),
		}
	}
}

struct LlvmIntrins {
	trap: Intrinsic,
}
impl LlvmIntrins {
	fn new(ctx: &inkwell::context::Context) -> Self {
		let trap = intrinsics::Intrinsic::find("llvm.trap").unwrap();
		Self { trap }
	}
}

struct FnLowerCtx<'a, 'ctx> {
	compilation_unit: &'a CompilationUnit,
	module: ModuleId,
	lowerer: &'a mut Lowerer<'ctx>,
	vtir_inst_to_llvm_value: FxHashMap<vtir::InstructionId, AnyValueEnum<'ctx>>,
	vtir_block_to_break_list: FxHashMap<vtir::InstructionId, BreakList<'ctx>>,
	// TODO(zino): remove these options once function-lowering state has a cleaner initialization path.
	cur_fn: Option<inkwell::values::FunctionValue<'ctx>>,
	cur_fn_abi: Option<FnAbi<'ctx>>,
	cur_fn_param_idx: u32,
	di_gen: Option<DebugInfoGen<'ctx>>,
}

#[derive(Debug)]
struct BreakList<'ctx> {
	body_bb: Option<BasicBlock<'ctx>>,
	after_bb: BasicBlock<'ctx>,
	breaks: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)>,
}

impl<'a, 'ctx> FnLowerCtx<'a, 'ctx> {
	fn resolve_inst(
		&mut self,
		inst: &vtir::InstructionRef,
	) -> AnyValueEnum<'ctx> {
		match inst {
			vtir::InstructionRef::Instruction(id) => self
				.vtir_inst_to_llvm_value
				.get(id)
				.copied()
				.unwrap_or_else(|| panic!("{id} should be lowered")),
			vtir::InstructionRef::Interned(val) => self.lowerer.lower_interned_value(*val),
		}
	}

	fn builder(&self) -> &inkwell::builder::Builder<'ctx> {
		&self.lowerer.builder
	}

	fn lower_body_inst(
		&mut self,
		vtir: &Vtir,
		parent_bb: BasicBlock<'ctx>,
		id: &vtir::InstructionId,
		inst: &vtir::Opcode,
	) -> Result<(), BuilderError> {
		let ctx = self.lowerer.ctx;

		match inst {
			vtir::Opcode::Invalid => unreachable!(),
			vtir::Opcode::Noop => {},
			vtir::Opcode::Block { instructions, ret_ty } => {
				let after_block_bb = ctx.insert_basic_block_after(parent_bb, "after_block");
				self.vtir_block_to_break_list.insert(*id, BreakList {
					body_bb: None,
					after_bb: after_block_bb,
					breaks: Vec::new(),
				});
				self.lower_body(vtir, parent_bb, instructions);

				// If the body didn't terminate, add fallthrough to after_block
				let current_bb = self.builder().get_insert_block().unwrap();
				if current_bb.get_terminator().is_none() {
					self.builder().build_unconditional_branch(after_block_bb)?;
				}

				self.builder().position_at_end(after_block_bb);

				if *ret_ty != self.compilation_unit.values.common.void_t {
					let ret_ty: BasicTypeEnum = self.lowerer.lower_type(*ret_ty).try_into().unwrap();
					let break_list = self.vtir_block_to_break_list.get(id).unwrap();
					if !break_list.breaks.is_empty() {
						let phi = self.builder().build_phi(ret_ty, "")?;
						for (break_value, break_bb) in &break_list.breaks {
							phi.add_incoming(&[(break_value, *break_bb)]);
						}
						self.vtir_inst_to_llvm_value.insert(*id, phi.as_any_value_enum());
					}
				}
			},
			vtir::Opcode::Loop { instructions, ret_ty } => {
				let loop_bb = ctx.insert_basic_block_after(parent_bb, "loop");
				self.builder().build_unconditional_branch(loop_bb)?;

				let after_loop_bb = ctx.insert_basic_block_after(parent_bb, "after_loop");
				self.vtir_block_to_break_list.insert(*id, BreakList {
					body_bb: Some(loop_bb),
					after_bb: after_loop_bb,
					breaks: Vec::new(),
				});
				self.builder().position_at_end(loop_bb);

				self.lower_body(vtir, loop_bb, instructions);

				self.builder().position_at_end(after_loop_bb);

				if *ret_ty != self.compilation_unit.values.common.void_t {
					let ret_ty: BasicTypeEnum = self.lowerer.lower_type(*ret_ty).try_into().unwrap();
					let break_list = self.vtir_block_to_break_list.get(id).unwrap();
					if !break_list.breaks.is_empty() {
						let phi = self.builder().build_phi(ret_ty, "")?;
						for (break_value, break_bb) in &break_list.breaks {
							phi.add_incoming(&[(break_value, *break_bb)]);
						}
						self.vtir_inst_to_llvm_value.insert(*id, phi.as_any_value_enum());
					}
				}
			},
			vtir::Opcode::Repeat { r#loop } => {
				let break_list = self.vtir_block_to_break_list.get(r#loop).unwrap();
				self.builder().build_unconditional_branch(break_list.body_bb.unwrap())?;
			},
			vtir::Opcode::Break { block, value } => {
				let value_ty = vtir::opcodes::type_of(&self.compilation_unit.values, &vtir.instructions, value);
				let is_void = value_ty == self.compilation_unit.values.common.void_t;
				if is_void {
					let break_list = self.vtir_block_to_break_list.get(block).unwrap();
					self.builder().build_unconditional_branch(break_list.after_bb)?;
				} else {
					let value = self.resolve_inst(value).try_into().unwrap();
					let insert_block = self.builder().get_insert_block().unwrap();
					let break_list = self.vtir_block_to_break_list.get_mut(block).unwrap();
					break_list.breaks.push((value, insert_block));
					// Keep the mutable borrow short before branching.
					self.lowerer.builder.build_unconditional_branch(break_list.after_bb)?;
				}
			},
			vtir::Opcode::StackAlloc { ty: ptr_ty } => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*ptr_ty).as_type_ptr().pointee_ty;
				let ty: BasicTypeEnum = self.lowerer.lower_type(pointee_ty).try_into().unwrap();

				// LLVM recommends to put the alloca instruction at the beginning of the function (the entry bb)
				// This is because the SROA and Mem2Reg passes only optimize alloca inside the entry basic block of the fn
				// So any branching, blocks, inline calls will cause allocas outside the entry block to not be traversed by those passes
				// which can have a great impact for the optimizer (in compile time and execution time)
				let prev_block = self.builder().get_insert_block().unwrap();
				let fn_entry_block = self.cur_fn.unwrap().get_first_basic_block().unwrap();

				// position at the beginning of the entry block not at the end since we may have a terminator (branch, ...) at the end
				match fn_entry_block.get_first_instruction().as_ref() {
					Some(first_instr) => self.builder().position_before(first_instr),
					None => self.builder().position_at_end(fn_entry_block),
				}

				let alloca = self.builder().build_alloca(ty, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, alloca.into());

				self.builder().position_at_end(prev_block);
			},
			vtir::Opcode::Load { ptr } => {
				let pointee_ty = {
					let ptr_ty = vtir.type_of(&self.compilation_unit.values, ptr);
					let pointee_ty = self.compilation_unit.values.index_to_key(ptr_ty).as_type_ptr().pointee_ty;
					self.lowerer.lower_type(pointee_ty).try_into().unwrap()
				};
				let ptr = self.resolve_inst(ptr);
				let val = self
					.builder()
					.build_load::<BasicTypeEnum>(pointee_ty, ptr.into_pointer_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.into());
			},
			vtir::Opcode::Store { src, dst } => {
				let src: BasicValueEnum = self.resolve_inst(src).try_into().unwrap();

				match self
					.compilation_unit
					.values
					.index_to_key(vtir.type_of(&self.compilation_unit.values, dst))
				{
					value::Key::TypePtr(dst_ty_ptr) => {
						let dst = self.resolve_inst(dst).into_pointer_value();
						if let Some(packed) = &dst_ty_ptr.packed {
							let underlying_int_ty = ctx.custom_width_int_type(packed.underlying_int_bits);
							let _pointee_ty = self.lowerer.lower_type(dst_ty_ptr.pointee_ty);
							let pointee_int_ty = ctx.custom_width_int_type(packed.bit_width);

							// first load the dst value
							let dst_val = self
								.builder()
								.build_load(underlying_int_ty, dst, "store.packed.load_underlying_int")?
								.into_int_value();

							// compute mask of values to keep
							let preserve_mask = {
								// init mask of bits we'll touch, perform a z_extend to zero exceess bits
								let mask =
									self.builder()
										.build_int_z_extend(pointee_int_ty.const_int(u64::MAX, false), underlying_int_ty, "")?;

								// shift left mask to put it at the right offset
								let mask = self.builder().build_left_shift(
									mask,
									underlying_int_ty.const_int(packed.bit_offset as _, false),
									"",
								)?;

								// mask ^ ~0 to get mask of all bits to preserve
								self.builder().build_xor(mask, underlying_int_ty.const_int(u64::MAX, false), "")?
							};

							// compute final value
							let src = {
								// clear unpreserved bits of destination
								let dst_val = self.builder().build_and(dst_val, preserve_mask, "")?;

								// src => equivalent-sized int
								let src = self
									.builder()
									.build_bit_cast(src, pointee_int_ty, "store.packed.bitcast")?
									.into_int_value();

								// z_extend src value to dst type
								let src = self.builder().build_int_z_extend(src, underlying_int_ty, "")?;

								// shift src value to right offset
								let src =
									self.builder()
										.build_left_shift(src, underlying_int_ty.const_int(packed.bit_offset as _, false), "")?;

								// and finally or src & dst
								self.builder().build_or(src, dst_val, "")?
							};

							self.builder().build_store(dst, src)?;
						} else {
							self.builder().build_store(dst, src)?;
						}
					},
					_ => {
						let dst = self.resolve_inst(dst).into_pointer_value();
						self.builder().build_store(dst, src)?;
					},
				}
			},
			vtir::Opcode::FnParam { .. } => {
				let fun = self.cur_fn.unwrap();
				let param = fun.get_nth_param(self.cur_fn_param_idx).unwrap();
				self.cur_fn_param_idx += 1;
				self.vtir_inst_to_llvm_value.insert(*id, param.into());
			},
			vtir::Opcode::Return { value } => match self.cur_fn_abi.as_ref().unwrap().ret_mode {
				abi::RetMode::Direct => {
					if let Some(value) = value
						&& vtir.type_of(&self.compilation_unit.values, value) != self.compilation_unit.values.common.void_t
					{
						let value = self.resolve_inst(value);
						let value: BasicValueEnum = value.try_into().unwrap();
						self.builder().build_return(Some(&value))?;
					} else {
						self.builder().build_return(None)?;
					};
				},
				abi::RetMode::SretFirstParam(_) => {
					let ret_ptr = self.cur_fn.unwrap().get_nth_param(0).unwrap();
					if let Some(value) = value
						&& vtir.type_of(&self.compilation_unit.values, value) != self.compilation_unit.values.common.void_t
					{
						let value = self.resolve_inst(value);
						let value: BasicValueEnum = value.try_into().unwrap();
						self.builder().build_store(ret_ptr.into_pointer_value(), value)?;
					}
					self.builder().build_return(None)?;
				},
			},
			vtir::Opcode::FnCall { callee, params } => {
				let fn_ty_idx = vtir.type_of(&self.compilation_unit.values, callee);
				let fn_ty = self.compilation_unit.values.index_to_key(fn_ty_idx).as_type_fn();
				let callconv = fn_ty.callconv;
				let llvm_fn_ty = self.lowerer.lower_type(fn_ty_idx).into_function_type();
				let callee_param_tys = llvm_fn_ty.get_param_types();

				let (params, ret_ptr) = {
					let mut values = Vec::<BasicMetadataValueEnum>::with_capacity(params.len());

					// if we have a ret ptr (sret), it is the first param
					// use fn_returns_by_ref_in_first_param instead of recomputing the ABI which is costlier
					// TODO(zino): consider wrapping LLVM types with compiler-side ABI metadata.
					let ret_ptr = if abi::fn_returns_by_ref_in_first_param(self.lowerer, fn_ty) {
						let ret_ty: BasicTypeEnum = self.lowerer.lower_type(fn_ty.ret_ty).try_into().unwrap();
						let ret_ptr = self.builder().build_alloca(ret_ty, "sret_ret_ptr")?;
						values.push(ret_ptr.into());
						Some((ret_ptr, ret_ty))
					} else {
						None
					};

					for (i, param) in params.iter().enumerate() {
						let param_val = self.resolve_inst(param);
						// Win64 ABI: lower struct args to match declaration (integer for small, pointer for large).
						let final_val = if let Some(BasicMetadataTypeEnum::IntType(int_ty)) = callee_param_tys.get(i) {
							if let AnyValueEnum::StructValue(sv) = param_val {
								let alloca = self.builder().build_alloca(sv.get_type(), "")?;
								self.builder().build_store(alloca, sv)?;
								self.builder().build_load(*int_ty, alloca, "")?.as_any_value_enum()
							} else {
								param_val
							}
						} else if let Some(BasicMetadataTypeEnum::PointerType(_)) = callee_param_tys.get(i) {
							// Large struct (> 8 bytes): store to alloca, pass pointer.
							if let AnyValueEnum::StructValue(sv) = param_val {
								let alloca = self.builder().build_alloca(sv.get_type(), "")?;
								self.builder().build_store(alloca, sv)?;
								alloca.as_any_value_enum()
							} else {
								param_val
							}
						} else {
							param_val
						};
						values.push(final_val.try_into().unwrap());
					}
					(values, ret_ptr)
				};

				let callee = self.resolve_inst(callee);
				let val = match callee {
					AnyValueEnum::FunctionValue(callee_fn) => self.builder().build_direct_call(callee_fn, &params, "")?,
					AnyValueEnum::PointerValue(fn_ptr) => self.builder().build_indirect_call(llvm_fn_ty, fn_ptr, &params, "")?,
					_ => unreachable!("FnCall callee must lower to a function or function pointer"),
				};

				if let Some(callconv) = callconv {
					val.set_call_convention(self.lowerer.llvm_callconv_id(callconv) as u32);
				}

				// if we have a ret_ptr, the actual val we returns for this call is a load to this pointer
				let val = if let Some((ret_ptr, ret_ty)) = ret_ptr {
					self.builder().build_load(ret_ty, ret_ptr, "sret_ret_ptr_load")?.as_any_value_enum()
				} else {
					val.as_any_value_enum()
				};

				self.vtir_inst_to_llvm_value.insert(*id, val);
			},
			// unary
			vtir::Opcode::BoolNot { op } => {
				let op = self.resolve_inst(op).into_int_value();
				let val = self.builder().build_not(op, "bool.not")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			// arithmtics
			vtir::Opcode::Add { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

					let lhs = lhs.into_int_value();
					let rhs = rhs.into_int_value();

					let val = if signed {
						self.builder().build_int_nsw_add(lhs, rhs, "")?
					} else {
						self.builder().build_int_nuw_add(lhs, rhs, "")?
					};

					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_add(lhs.into_float_value(), rhs.into_float_value(), "")?;

					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::AddSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let add_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.sadd.sat" } else { "llvm.uadd.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let add_sat_fn = add_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let val = self.builder().build_call(add_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::AddWrap { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = self.builder().build_int_add(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Sub { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

					let lhs = lhs.into_int_value();
					let rhs = rhs.into_int_value();

					let val = if signed {
						self.builder().build_int_nsw_sub(lhs, rhs, "")?
					} else {
						self.builder().build_int_nuw_sub(lhs, rhs, "")?
					};

					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::SubSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let sub_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.ssub.sat" } else { "llvm.usub.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let sub_sat_fn = sub_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let value = self.builder().build_call(sub_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
			},
			vtir::Opcode::SubWrap { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = self.builder().build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "")?;

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Mul { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

					let lhs = lhs.into_int_value();
					let rhs = rhs.into_int_value();

					let val = if signed {
						self.builder().build_int_nsw_mul(lhs, rhs, "")?
					} else {
						self.builder().build_int_nuw_mul(lhs, rhs, "")?
					};

					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::MulSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let mul_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.smul.sat" } else { "llvm.umul.sat" }).unwrap();

				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_ty = lhs_int.get_type();
				let _rhs_ty = rhs_int.get_type();

				let mul_sat_fn = mul_sat_intrinsic.get_declaration(&self.lowerer.module, &[lhs_ty.into()]).unwrap();

				let val = self.builder().build_call(mul_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::MulWrap { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = self.builder().build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Div { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					let val = if signed {
						self.builder()
							.build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "")?
					};
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_div(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::Rem { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);

					let val = if signed {
						self.builder()
							.build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "")?
					};
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				} else {
					assert!(lhs.get_type().is_float_type());
					let val = self.builder().build_float_rem(lhs.into_float_value(), rhs.into_float_value(), "")?;
					self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
				}
			},
			vtir::Opcode::Lt { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);

					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SLT, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::ULT, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OLT, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Lte { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SLE, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::ULE, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OLE, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Gt { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SGT, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::UGT, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OGT, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Gte { lhs, rhs } => {
				let ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					let signed = self.compilation_unit.values.type_is_int_signed(ty);
					if signed {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::SGE, lhs.into_int_value(), rhs.into_int_value(), "")?
					} else {
						self.builder()
							.build_int_compare(inkwell::IntPredicate::UGE, lhs.into_int_value(), rhs.into_int_value(), "")?
					}
				} else {
					assert!(lhs.get_type().is_float_type());
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OGE, lhs.into_float_value(), rhs.into_float_value(), "")?
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BoolAnd { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = self.builder().build_and(lhs.into_int_value(), rhs.into_int_value(), "")?;

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BoolOr { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = self.builder().build_or(lhs.into_int_value(), rhs.into_int_value(), "")?;

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			// bitwise
			vtir::Opcode::Shl { lhs, rhs } | vtir::Opcode::ShlWrap { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self.builder().build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::ShlSat { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let shl_sat_intrinsic = intrinsics::Intrinsic::find(if signed { "llvm.sshl.sat" } else { "llvm.ushl.sat" }).unwrap();
				let lhs_int = lhs.into_int_value();
				let rhs_int = rhs.into_int_value();
				let lhs_llvm_ty = lhs_int.get_type();
				let shl_sat_fn = shl_sat_intrinsic
					.get_declaration(&self.lowerer.module, &[lhs_llvm_ty.into()])
					.unwrap();
				let val = self.builder().build_call(shl_sat_fn, &[lhs_int.into(), rhs_int.into()], "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Shr { lhs, rhs } | vtir::Opcode::ShrWrap { lhs, rhs } => {
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self
					.builder()
					.build_right_shift(lhs.into_int_value(), rhs.into_int_value(), signed, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::ShrSat { lhs, rhs } => {
				// Saturating right shift is the same as regular right shift
				let lhs_ty = vtir.type_of(&self.compilation_unit.values, lhs);
				let signed = self.compilation_unit.values.type_is_int_signed(lhs_ty);

				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self
					.builder()
					.build_right_shift(lhs.into_int_value(), rhs.into_int_value(), signed, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BitAnd { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self.builder().build_and(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BitOr { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self.builder().build_or(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BitXor { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);
				let val = self.builder().build_xor(lhs.into_int_value(), rhs.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::BitNot { op } => {
				let op = self.resolve_inst(op);
				let val = self.builder().build_not(op.into_int_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},

			vtir::Opcode::Eq { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::EQ, lhs.into_int_value(), rhs.into_int_value(), "")?
				} else if lhs.get_type().is_float_type() {
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "")?
				} else if lhs.get_type().is_pointer_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::EQ, lhs.into_pointer_value(), rhs.into_pointer_value(), "")?
				} else {
					unreachable!("eq not implemented for {lhs} and {rhs}")
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Neq { lhs, rhs } => {
				let lhs = self.resolve_inst(lhs);
				let rhs = self.resolve_inst(rhs);

				let val = if lhs.get_type().is_int_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::NE, lhs.into_int_value(), rhs.into_int_value(), "")?
				} else if lhs.get_type().is_float_type() {
					self.builder()
						.build_float_compare(inkwell::FloatPredicate::ONE, lhs.into_float_value(), rhs.into_float_value(), "")?
				} else if lhs.get_type().is_pointer_type() {
					self.builder()
						.build_int_compare(inkwell::IntPredicate::NE, lhs.into_pointer_value(), rhs.into_pointer_value(), "")?
				} else {
					unreachable!()
				};

				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},

			// structs
			vtir::Opcode::StructInit { struct_ty, fields } => {
				let is_packed_struct = matches!(
					self.compilation_unit.values.index_to_key_value(*struct_ty),
					(value::Key::TypeStruct(_), value::Value::Struct(s)) if s.as_ref().is_packed()
				);
				if is_packed_struct {
					let r#struct = self.compilation_unit.values.index_to_value(*struct_ty).as_struct();
					let r#struct = r#struct.as_ref();
					let struct_ty_llvm = self.lowerer.lower_type(*struct_ty).into_int_type();

					let struct_int = struct_ty_llvm.const_int(0, false);
					let (_, struct_int) = fields.iter().try_fold((0u32, struct_int), |(cur_bit_offset, struct_int), field| {
						let field_ty = vtir.type_of(&self.compilation_unit.values, field);
						let field_ty_int = ctx.custom_width_int_type(self.compilation_unit.values.type_bit_size(field_ty));
						let struct_int: IntValue = {
							let field: BasicValueEnum = self.resolve_inst(field).try_into().unwrap();
							let field = self.builder().build_bit_cast(field, field_ty_int, "")?.into_int_value();
							let field = self.builder().build_int_z_extend(field, struct_ty_llvm, "")?;
							let field = self
								.builder()
								.build_left_shift(field, struct_ty_llvm.const_int(cur_bit_offset as _, false), "")?;
							self.builder().build_or(struct_int, field, "")?
						};
						Ok::<(u32, inkwell::values::IntValue<'_>), BuilderError>((
							cur_bit_offset + self.compilation_unit.values.type_bit_size(field_ty),
							struct_int,
						))
					})?;

					self.vtir_inst_to_llvm_value.insert(*id, struct_int.as_any_value_enum());
				} else {
					let struct_ty_llvm = self.lowerer.lower_type(*struct_ty);
					let mut value = struct_ty_llvm.into_struct_type().get_poison().into();
					for (i, field) in fields.iter().enumerate() {
						let field: BasicValueEnum = match self.resolve_inst(field) {
							AnyValueEnum::FunctionValue(fun) => fun.as_global_value().as_pointer_value().as_basic_value_enum(),
							value => value.try_into().unwrap(),
						};
						value = self.builder().build_insert_value(value, field, i as _, "")?;
					}
					self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
				}
			},
			vtir::Opcode::StructFieldValue {
				struct_ty,
				field_idx,
				ret_ty,
			} => {
				let ty = vtir.type_of(&self.compilation_unit.values, struct_ty);

				// TypeEffect is not a Value::Struct. Handle it as a plain non-packed struct extract.
				let is_packed = match self.compilation_unit.values.index_to_key(ty) {
					value::Key::TypeStruct(_) => {
						let s = self.compilation_unit.values.index_to_value(ty).as_struct();
						matches!(s.as_ref().layout, value::StructLayout::Packed { .. })
					},
					_ => false,
				};

				if is_packed {
					let ty = self.compilation_unit.values.index_to_value(ty).as_struct();
					let ty = ty.as_ref();
					if let value::StructLayout::Packed { packed_fields, .. } = ty.layout {
						let field_info = &packed_fields[*field_idx];
						let struct_value = self.resolve_inst(struct_ty).into_int_value();

						let field = if field_info.offset > 0 {
							self.builder().build_right_shift(
								struct_value,
								ctx.custom_width_int_type(struct_value.get_type().get_bit_width())
									.const_int(field_info.offset as u64, false),
								false,
								"",
							)?
						} else {
							struct_value
						};

						let field = {
							let _ty: BasicTypeEnum = self.lowerer.lower_type(*ret_ty).try_into().unwrap();
							let field_ty_int = ctx.custom_width_int_type(self.compilation_unit.values.type_bit_size(*ret_ty));
							self.builder().build_int_truncate(field, field_ty_int, "")?
						};

						self.vtir_inst_to_llvm_value.insert(*id, field.as_any_value_enum());
					} else {
						unreachable!()
					}
				} else {
					let struct_value = self.resolve_inst(struct_ty).into_struct_value();
					let field = self.builder().build_extract_value(struct_value, *field_idx as u32, "")?;
					self.vtir_inst_to_llvm_value.insert(*id, field.as_any_value_enum());
				}
			},
			vtir::Opcode::StructFieldPtr {
				struct_ptr,
				field_idx,
				ret_ty: _,
			} => {
				let struct_ty = {
					let struct_ptr_ty = vtir.type_of(&self.compilation_unit.values, struct_ptr);
					let ptr_ty = self.compilation_unit.values.index_to_key(struct_ptr_ty).as_type_ptr();
					ptr_ty.pointee_ty
				};
				let r#struct = self.compilation_unit.values.index_to_value(struct_ty).as_struct();
				let r#struct = r#struct.as_ref();

				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type(struct_ty).try_into().unwrap();
				let ptr = self.resolve_inst(struct_ptr).into_pointer_value();

				// if the struct is packed, returns directly the ptr to the struct
				// the offset is encoded into the ret_ty type already so other insts should handle packed structs through that
				let ptr = if r#struct.is_packed() {
					ptr
				} else {
					self.builder().build_struct_gep(pointee_ty, ptr, *field_idx as u32, "")?
				};
				self.vtir_inst_to_llvm_value.insert(*id, ptr.as_any_value_enum());
			},

			// unions
			vtir::Opcode::UnionInit {
				union_ty,
				field_idx,
				value: payload_value,
			} => {
				let union_llvm_ty = self.lowerer.lower_type(*union_ty);
				let union_val_ty = self.compilation_unit.values.index_to_value(*union_ty).as_union();
				let union_val_ty = union_val_ty.as_ref();

				if let Some(tag_ty) = union_val_ty.tag_ty {
					// Tagged union: { tag, [N x i8] }
					let struct_ty = union_llvm_ty.into_struct_type();
					let mut value = struct_ty.get_poison().into();

					// Set tag value = field_idx
					let tag_llvm_ty = self.lowerer.lower_type(tag_ty).into_int_type();
					let tag_val = tag_llvm_ty.const_int(*field_idx as u64, false);
					value = self.builder().build_insert_value(value, tag_val, 0, "")?;

					// Set payload if present
					if let Some(payload) = payload_value {
						let payload_val: inkwell::values::BasicValueEnum = self.resolve_inst(payload).try_into().unwrap();
						// Alloca the union, store payload into it via bitcast
						let alloca = self.builder().build_alloca(struct_ty, "")?;
						let store_val: inkwell::values::BasicValueEnum = value.as_any_value_enum().try_into().unwrap();
						self.builder().build_store(alloca, store_val)?;
						let payload_ptr = self.builder().build_struct_gep(struct_ty, alloca, 1, "")?;
						let payload_ptr = self
							.builder()
							.build_bit_cast(payload_ptr, ctx.ptr_type(inkwell::AddressSpace::default()), "")?;
						self.builder().build_store(payload_ptr.into_pointer_value(), payload_val)?;
						let result = self.builder().build_load(struct_ty, alloca, "")?;
						self.vtir_inst_to_llvm_value.insert(*id, result.as_any_value_enum());
					} else {
						self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
					}
				} else {
					// Bare union: [N x i8]
					let array_ty = union_llvm_ty.into_array_type();
					if let Some(payload) = payload_value {
						let payload_val: inkwell::values::BasicValueEnum = self.resolve_inst(payload).try_into().unwrap();
						let alloca = self.builder().build_alloca(array_ty, "")?;
						let ptr = self
							.builder()
							.build_bit_cast(alloca, ctx.ptr_type(inkwell::AddressSpace::default()), "")?;
						self.builder().build_store(ptr.into_pointer_value(), payload_val)?;
						let result = self.builder().build_load(array_ty, alloca, "")?;
						self.vtir_inst_to_llvm_value.insert(*id, result.as_any_value_enum());
					} else {
						let value = array_ty.const_zero();
						self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
					}
				}
			},
			vtir::Opcode::UnionTag { union_val, tag_ty } => {
				let union_value = self.resolve_inst(union_val).into_struct_value();
				let tag = self.builder().build_extract_value(union_value, 0, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, tag.as_any_value_enum());
			},
			vtir::Opcode::UnionFieldValue {
				union_val,
				field_idx,
				ret_ty,
			} => {
				let union_ty_idx = vtir.type_of(&self.compilation_unit.values, union_val);
				let union_llvm_ty = self.lowerer.lower_type(union_ty_idx);
				let ret_llvm_ty: inkwell::types::BasicTypeEnum = self.lowerer.lower_type(*ret_ty).try_into().unwrap();

				let union_value: inkwell::values::BasicValueEnum = self.resolve_inst(union_val).try_into().unwrap();

				// Alloca the union, GEP to payload, bitcast, load
				let alloca = self
					.builder()
					.build_alloca(inkwell::types::BasicTypeEnum::try_from(union_llvm_ty).unwrap(), "")?;
				self.builder().build_store(alloca, union_value)?;

				let union_ty_ref = self.compilation_unit.values.index_to_value(union_ty_idx).as_union();
				let union_ty_ref = union_ty_ref.as_ref();

				let payload_ptr = if union_ty_ref.tag_ty.is_some() {
					// Tagged: GEP to field 1 (payload)
					self.builder().build_struct_gep(union_llvm_ty.into_struct_type(), alloca, 1, "")?
				} else {
					// Bare: the whole thing is the payload
					alloca
				};

				let cast_ptr = self
					.builder()
					.build_bit_cast(payload_ptr, ctx.ptr_type(inkwell::AddressSpace::default()), "")?;
				let result = self.builder().build_load(ret_llvm_ty, cast_ptr.into_pointer_value(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, result.as_any_value_enum());
			},

			// builtins
			vtir::Opcode::UnsafeIntCast { src, dst_ty } => {
				let src = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let src = self.resolve_inst(src);
					src.into_int_value()
				};
				let is_signed = self.compilation_unit.values.type_is_int_signed(*dst_ty);

				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = self
					.builder()
					.build_int_cast_sign_flag(src, dst_ty.into_int_type(), is_signed, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::UnsafeFloatCast { src, dst_ty } => {
				let src = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let _src_ty = self.lowerer.lower_type(src_ty);
					let src = self.resolve_inst(src);
					src.into_float_value()
				};
				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = self.builder().build_float_cast(src, dst_ty.into_float_type(), "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::IntToFloat { src, dst_ty } => {
				let (src, is_signed) = {
					let src_ty = vtir.type_of(&self.compilation_unit.values, src);
					let src = self.resolve_inst(src);
					let is_signed = self.compilation_unit.values.type_is_int_signed(src_ty);
					(src.into_int_value(), is_signed)
				};
				let dst_ty = self.lowerer.lower_type(*dst_ty);
				let val = if is_signed {
					self.builder().build_signed_int_to_float(src, dst_ty.into_float_type(), "")?
				} else {
					self.builder().build_unsigned_int_to_float(src, dst_ty.into_float_type(), "")?
				};
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::SizeOf { of } => {
				let of = self.lowerer.lower_type(of.as_interned());
				let val = of.size_of().expect("@size_of called on non-sized type");
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Zeroed { ty } => {
				let ty = self.lowerer.lower_type(*ty);
				let ty: BasicTypeEnum<'_> = ty.try_into().unwrap();
				self.vtir_inst_to_llvm_value.insert(*id, ty.const_zero().as_any_value_enum());
			},
			vtir::Opcode::BitCast { src, dst_ty } => {
				let src: BasicValueEnum = self.resolve_inst(src).try_into().unwrap();
				let dst_ty: BasicTypeEnum = self.lowerer.lower_type(*dst_ty).try_into().unwrap();
				let val = self.builder().build_bit_cast(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::Undefined { ty } => {
				let ty = self.lowerer.lower_type(*ty);
				let ty: BasicTypeEnum<'_> = ty.try_into().unwrap();

				let undef_value = match ty {
					BasicTypeEnum::IntType(int_ty) => int_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::FloatType(float_ty) => float_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::PointerType(ptr_ty) => ptr_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::ArrayType(array_ty) => array_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::StructType(struct_ty) => struct_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::VectorType(vec_ty) => vec_ty.get_undef().as_any_value_enum(),
					BasicTypeEnum::ScalableVectorType(svec_ty) => svec_ty.get_undef().as_any_value_enum(),
				};
				self.vtir_inst_to_llvm_value.insert(*id, undef_value);
			},
			vtir::Opcode::SliceFromRawParts { slice_ty, ptr, len } => {
				let slice_ty = self.lowerer.lower_type(*slice_ty).into_struct_type();
				let ptr: BasicValueEnum = self.resolve_inst(ptr).try_into().unwrap();
				let len: BasicValueEnum = self.resolve_inst(len).try_into().unwrap();
				let undef = slice_ty.get_undef();
				let with_ptr = self.builder().build_insert_value(undef, ptr, 0, "slice.ptr").unwrap();
				let with_len = self.builder().build_insert_value(with_ptr, len, 1, "slice.len").unwrap();
				self.vtir_inst_to_llvm_value.insert(*id, with_len.as_any_value_enum());
			},
			vtir::Opcode::SlicePtr { slice, ptr_ty } => {
				let slice = self.resolve_inst(slice).into_struct_value();
				let ptr = self.builder().build_extract_value(slice, 0, "slice.ptr").unwrap();
				self.vtir_inst_to_llvm_value.insert(*id, ptr.as_any_value_enum());
			},
			vtir::Opcode::SliceLen { slice } => {
				let slice = self.resolve_inst(slice).into_struct_value();
				let len = self.builder().build_extract_value(slice, 1, "slice.len").unwrap();
				self.vtir_inst_to_llvm_value.insert(*id, len.as_any_value_enum());
			},
			vtir::Opcode::PtrToInt { src, dst_ty } => {
				let src = self.resolve_inst(src).into_pointer_value();
				let dst_ty = self.lowerer.lower_type(*dst_ty).into_int_type();
				let val = self.builder().build_ptr_to_int(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::IntToPtr { src, dst_ty } => {
				let src = self.resolve_inst(src).into_int_value();
				let dst_ty = self.lowerer.lower_type(*dst_ty).into_pointer_type();
				let val = self.builder().build_int_to_ptr(src, dst_ty, "")?;
				self.vtir_inst_to_llvm_value.insert(*id, val.as_any_value_enum());
			},
			vtir::Opcode::SliceCopyNonoverlapping { slice_ty, src, dst } => {
				let src_pointee_ty = self.compilation_unit.values.index_to_key(*slice_ty).as_type_slice().pointee_ty;
				let src_pointee_ty = self.lowerer.lower_type(src_pointee_ty);

				let src = self.resolve_inst(src).into_struct_value();
				let dst = self.resolve_inst(dst).into_struct_value();
				let src_ptr = self.builder().build_extract_value(src, 0, "slice.ptr")?.into_pointer_value();
				let src_len = self.builder().build_extract_value(src, 1, "slice.len").unwrap().into_int_value();
				let src_len_in_bytes = self
					.builder()
					.build_int_mul(src_len, src_pointee_ty.size_of().unwrap(), "src_len_in_bytes")?;
				let dst_ptr = self.builder().build_extract_value(dst, 0, "slice.ptr")?.into_pointer_value();
				let value = self.builder().build_memcpy(dst_ptr, 4, src_ptr, 4, src_len_in_bytes).unwrap();
				self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
			},
			vtir::Opcode::SliceElemPtr { slice, index, elem_ptr_ty } => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*elem_ptr_ty).as_type_ptr().pointee_ty;
				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type(pointee_ty).try_into().unwrap();

				let src = self.resolve_inst(slice).into_struct_value();
				let src_ptr = self
					.builder()
					.build_extract_value(src, 0, "slice.ptr")
					.unwrap()
					.into_pointer_value();
				let index = self.resolve_inst(index).into_int_value();
				let value = unsafe {
					self.builder()
						.build_in_bounds_gep(pointee_ty, src_ptr, &[index], "slice.elem.ptr")?
				};
				self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
			},
			vtir::Opcode::PtrElemPtr {
				array_ptr,
				index,
				elem_ptr_ty,
			} => {
				let pointee_ty = self.compilation_unit.values.index_to_key(*elem_ptr_ty).as_type_ptr().pointee_ty;
				let pointee_ty: BasicTypeEnum = self.lowerer.lower_type(pointee_ty).try_into().unwrap();
				let array_ptr = self.resolve_inst(array_ptr).into_pointer_value();
				let index = self.resolve_inst(index).into_int_value();
				let zero = ctx.i32_type().const_zero();
				let value = unsafe {
					self.builder()
						.build_in_bounds_gep(pointee_ty, array_ptr, &[index], "ptr.array.elem.ptr")?
				};
				self.vtir_inst_to_llvm_value.insert(*id, value.as_any_value_enum());
			},

			vtir::Opcode::DbgSrcLoc { line, col } => {
				if let Some(di_gen) = self.di_gen.as_mut() {
					let location = di_gen.di_ctx.builder.create_debug_location(
						self.lowerer.ctx,
						*line as _,
						*col as _,
						di_gen.di_lexical_block_stack.last().unwrap().as_debug_info_scope(),
						None,
					);
					self.builder().set_current_debug_location(location);
				}
			},
			vtir::Opcode::Branch {
				cond,
				then_body,
				else_body,
			} => {
				let insert_bb = self.builder().get_insert_block().unwrap();
				let then_block = ctx.insert_basic_block_after(insert_bb, "then");
				let else_block = ctx.insert_basic_block_after(then_block, "else");

				let cond = self.resolve_inst(cond).into_int_value();
				let branch = self.builder().build_conditional_branch(cond, then_block, else_block)?;

				// then
				self.builder().position_at_end(then_block);
				self.lower_body(vtir, then_block, then_body);

				// else
				self.builder().position_at_end(else_block);
				self.lower_body(vtir, else_block, else_body);

				self.vtir_inst_to_llvm_value.insert(*id, branch.as_any_value_enum());
			},
			vtir::Opcode::Switch { operand, cases, else_body } => {
				let insert_bb = self.builder().get_insert_block().unwrap();
				let operand_val = self.resolve_inst(operand).into_int_value();

				// Create basic blocks: one per case body + else
				let else_bb = ctx.insert_basic_block_after(insert_bb, "switch_else");
				let mut case_bbs = Vec::with_capacity(cases.len());
				let mut prev_bb = else_bb;
				for i in 0..cases.len() {
					let bb = ctx.insert_basic_block_after(prev_bb, &format!("switch_case_{i}"));
					case_bbs.push(bb);
					prev_bb = bb;
				}

				// Build LLVM switch cases: (value, basic_block) pairs
				let mut switch_cases = Vec::new();
				for (i, case) in cases.iter().enumerate() {
					for item in case.items.iter() {
						let item_val = self.resolve_inst(item).into_int_value();
						switch_cases.push((item_val, case_bbs[i]));
					}
				}

				self.builder().build_switch(operand_val, else_bb, &switch_cases)?;

				// Lower each case body
				for (i, case) in cases.iter().enumerate() {
					self.builder().position_at_end(case_bbs[i]);
					self.lower_body(vtir, case_bbs[i], case.body);
				}

				// Lower else body
				self.builder().position_at_end(else_bb);
				if else_body.is_empty() {
					// Exhaustive switch: else is unreachable
					self.builder().build_unreachable()?;
				} else {
					self.lower_body(vtir, else_bb, else_body);
				}

				// For exhaustive switches (no else body), the else BB has an
				// unreachable terminator, so we need a merge block for any
				// subsequent instructions. For non-exhaustive switches, the
				// else body's lowering leaves the builder positioned correctly.
				if else_body.is_empty() {
					let merge_bb = ctx.insert_basic_block_after(prev_bb, "switch_merge");
					self.builder().position_at_end(merge_bb);
				}
			},
			vtir::Opcode::Abort => {
				let trap_fn = self.lowerer.intrins.trap.get_declaration(&self.lowerer.module, &[]).unwrap();
				self.builder().build_call(trap_fn, &[], "@abort")?;
			},
			vtir::Opcode::Unreachable => {
				self.builder().build_unreachable()?;
			},

			// invalid insts
			vtir::Opcode::StackAllocInferred { .. } => {
				unreachable!("StackAllocInferred should be replaced at uir2tir time");
			},
		};
		Ok(())
	}

	fn lower_body(
		&mut self,
		vtir: &Vtir,
		parent_bb: BasicBlock<'ctx>,
		insts: &[vtir::InstructionRef],
	) {
		for inst in insts.iter() {
			let id = inst.as_id().unwrap();
			let inst = &vtir.instructions[id];
			self.lower_body_inst(vtir, parent_bb, &id, inst).unwrap();
		}
	}

	fn lower_fn_body(
		&mut self,
		interned_fn_value: value::Index,
		body: &Vtir,
	) {
		let fn_value = self.compilation_unit.values.index_to_key(interned_fn_value).as_fn();
		let fn_ty = self.compilation_unit.values.index_to_key(fn_value.ty).as_type_fn();
		let fn_abi = abi::compute_fn_abi(self.lowerer, fn_ty);

		if fn_ty.external {
			let fn_value = self
				.resolve_inst(&InstructionRef::Interned(interned_fn_value))
				.into_function_value();
			fn_value.set_linkage(Linkage::External);
			return;
		}

		let fn_value = self
			.resolve_inst(&InstructionRef::Interned(interned_fn_value))
			.into_function_value();
		let block = self.lowerer.ctx.append_basic_block(fn_value, "entry");
		self.builder().position_at_end(block);

		if let Some(di_gen) = self.di_gen.as_mut() {
			let di_builder = &di_gen.di_ctx.builder;
			let di_cu = &di_gen.di_ctx.cu;

			// no debug info for external or builtins
			if !fn_ty.external {
				let mangled_name = fn_value.get_name().to_string_lossy();
				let di_sp_type = di_builder.create_subroutine_type(di_gen.di_file, None, &[], DIFlags::PUBLIC);
				let di_mangled_name = format!("{} {}", fn_value.print_to_string().to_string_lossy(), mangled_name);
				let di_sp = di_builder.create_function(
					di_gen.di_file.as_debug_info_scope(),
					&mangled_name,
					Some(&di_mangled_name),
					di_gen.di_file,
					0,
					di_sp_type,
					true,
					true,
					0,
					DIFlags::PROTOTYPED,
					false,
				);
				fn_value.set_subprogram(di_sp);
			}

			let di_parent_scope = fn_value
				.get_subprogram()
				.map(|s| s.as_debug_info_scope())
				.unwrap_or(di_cu.as_debug_info_scope());

			di_gen
				.di_lexical_block_stack
				.push(di_builder.create_lexical_block(di_parent_scope, di_gen.di_file, 0, 0));
		}

		self.cur_fn = Some(fn_value);
		self.cur_fn_param_idx = match fn_abi.ret_mode {
			abi::RetMode::Direct => 0,
			abi::RetMode::SretFirstParam(_) => 1,
		};
		self.cur_fn_abi = Some(fn_abi);
		self.lower_body(body, block, body.main_body);
		self.cur_fn = None;

		self.builder().unset_current_debug_location();
		if let Some(di) = self.di_gen.as_mut() {
			di.di_lexical_block_stack.pop();
		}
	}
}

pub fn initialize(build_opts: &Build) -> inkwell::context::Context {
	// required for tests with cargo test
	// not great since the first build_opts win but eh we should require them here tbh only for debugging
	static WAS_INITIALIZED: OnceLock<()> = OnceLock::new();
	WAS_INITIALIZED.get_or_init(|| {
		let empty_arg = c"".as_ptr();
		let args = [
			c"vifc".as_ptr(),
			c"-x86-asm-syntax=intel".as_ptr(),
			if build_opts.dump_llvm_timings {
				c"-time-passes".as_ptr()
			} else {
				empty_arg
			},
		];

		// We need to initialize targets before CLI opts parsing to have cl::opt flags registered
		Target::initialize_native(&InitializationConfig {
			asm_parser: true,
			asm_printer: true,
			base: true,
			disassembler: false,
			info: true,
			machine_code: true,
		});

		// SAFETY: todo
		unsafe { LLVMParseCommandLineOptions(args.len() as i32, args.as_ptr(), core::ptr::null()) };
	});

	inkwell::context::Context::create()
}

struct DebugInfoCtx<'ctx> {
	builder: inkwell::debug_info::DebugInfoBuilder<'ctx>,
	cu: inkwell::debug_info::DICompileUnit<'ctx>,
	module_to_file: FxHashMap<ModuleId, inkwell::debug_info::DIFile<'ctx>>,
}

pub struct Lowerer<'ctx> {
	compilation_unit: &'ctx CompilationUnit,
	ctx: &'ctx inkwell::context::Context,
	builder: inkwell::builder::Builder<'ctx>,
	module: inkwell::module::Module<'ctx>,
	target_machine: inkwell::targets::TargetMachine,
	interned_value_to_llvm_type: FxHashMap<value::Index, AnyTypeEnum<'ctx>>,
	interned_value_to_llvm_value: FxHashMap<value::Index, AnyValueEnum<'ctx>>,
	attributes: LlvmAttributes,
	intrins: LlvmIntrins,
	di: Option<DebugInfoCtx<'ctx>>,
	winapi_callconv: llvm_sys::LLVMCallConv,
}
impl<'ctx> Lowerer<'ctx> {
	fn llvm_callconv_id(
		&self,
		callconv: CallingConvention,
	) -> llvm_sys::LLVMCallConv {
		match callconv {
			CallingConvention::C => llvm_sys::LLVMCallConv::LLVMCCallConv,
			CallingConvention::Fast => llvm_sys::LLVMCallConv::LLVMFastCallConv,
			CallingConvention::Cold => llvm_sys::LLVMCallConv::LLVMColdCallConv,
			CallingConvention::Winapi => self.winapi_callconv,
		}
	}

	pub fn new(
		ctx: &'ctx inkwell::context::Context,
		compilation_unit: &'ctx CompilationUnit,
		build_opts: &Build,
	) -> Self {
		let triple = TargetTriple::create(&compilation_unit.resolved_target.triple.to_string());
		let target = { Target::from_triple(&triple).unwrap() };

		let cpu = TargetMachine::get_host_cpu_name();
		let features = TargetMachine::get_host_cpu_features();

		let target_machine = target
			.create_target_machine(
				&triple,
				cpu.to_string_lossy().as_ref(),
				features.to_string_lossy().as_ref(),
				match build_opts.opt {
					0 => inkwell::OptimizationLevel::None,
					1 => inkwell::OptimizationLevel::Less,
					2 => inkwell::OptimizationLevel::Default,
					3 => inkwell::OptimizationLevel::Aggressive,
					_ => unreachable!(),
				},
				inkwell::targets::RelocMode::Default,
				inkwell::targets::CodeModel::Default,
			)
			.unwrap();

		let builder = ctx.create_builder();
		let module = ctx.create_module("vifmod");
		let data_layout = target_machine.get_target_data().get_data_layout();
		module.set_data_layout(&data_layout);
		module.set_triple(&triple);

		// On Windows we want to emit CodeView data for PDB-based debuggers
		let u32_one = ctx.i32_type().const_int(1, false);

		#[cfg(target_os = "windows")]
		module.add_metadata_flag(
			"CodeView",
			inkwell::module::FlagBehavior::Error,
			ctx.metadata_node(&[u32_one.into()]),
		);

		let di = if build_opts.debug_info {
			let modules = compilation_unit.modules.read();
			let root_mod_path = modules[compilation_unit.root_module].path.clone();
			let (builder, cu) = module.create_debug_info_builder(
				true,
				inkwell::debug_info::DWARFSourceLanguage::C,
				root_mod_path.file_name().unwrap(),
				root_mod_path.parent().unwrap().as_str(),
				"vifc",
				false,
				"",
				0,
				"",
				inkwell::debug_info::DWARFEmissionKind::Full,
				0,
				false,
				false,
				"",
				"",
			);

			let mut module_to_file = FxHashMap::default();
			for (id, module) in compilation_unit.modules.read().iter().enumerate() {
				let path = &module.path;
				let file = builder.create_file(path.file_name().unwrap(), path.parent().unwrap().as_str());
				module_to_file.insert(ModuleId::from(id), file);
			}

			Some(DebugInfoCtx {
				builder,
				cu,
				module_to_file,
			})
		} else {
			None
		};

		let winapi_callconv = if compilation_unit.resolved_target.winapi_uses_stdcall {
			llvm_sys::LLVMCallConv::LLVMX86StdcallCallConv
		} else {
			llvm_sys::LLVMCallConv::LLVMCCallConv
		};

		Self {
			compilation_unit,
			ctx,
			builder,
			module,
			target_machine,
			interned_value_to_llvm_type: Default::default(),
			interned_value_to_llvm_value: Default::default(),
			attributes: LlvmAttributes::new(ctx),
			intrins: LlvmIntrins::new(ctx),
			di,
			winapi_callconv,
		}
	}

	fn build_slice(
		&mut self,
		slice_ty: &StructType<'ctx>,
		ptr: BasicValueEnum<'ctx>,
		len: BasicValueEnum<'ctx>,
	) -> AnyValueEnum<'ctx> {
		let undef = slice_ty.get_undef();
		let with_ptr = self.builder.build_insert_value(undef, ptr, 0, "slice.ptr").unwrap();
		let with_len = self.builder.build_insert_value(with_ptr, len, 1, "slice.len").unwrap();
		with_len.as_any_value_enum()
	}

	fn lower_interned_value(
		&mut self,
		val: value::Index,
	) -> AnyValueEnum<'ctx> {
		if let Some(value) = self.interned_value_to_llvm_value.get(&val) {
			return *value;
		}

		let key = self.compilation_unit.values.index_to_key(val);
		let llvm_val: AnyValueEnum = match key {
			value::Key::Str { value, slice_ty } => {
				let slice_ty = self.lower_type(*slice_ty).into_struct_type();
				let ptr = self
					.builder
					.build_global_string_ptr(core::str::from_utf8(value).expect("todo change"), "")
					.unwrap()
					.as_basic_value_enum();
				let len = self
					.ctx
					.ptr_sized_int_type(&self.target_machine.get_target_data(), None)
					.const_int(value.len().try_into().unwrap(), false)
					.as_basic_value_enum();
				self.build_slice(&slice_ty, ptr, len)
			},
			value::Key::Int { ty, value } => {
				let ty = self.lower_type(*ty);
				let str = value.to_string();
				ty.into_int_type()
					.const_int_from_string(&str, inkwell::types::StringRadix::Decimal)
					.unwrap()
					.into()
			},
			value::Key::Float { ty, value } => {
				let ty = self.lower_type(*ty);
				let value = **value as f64; // TODO(zino): f128 casted to f64 not good
				ty.into_float_type().const_float(value).into()
			},
			value::Key::Bool(b) => self.ctx.bool_type().const_int(*b as u64, false).into(),
			value::Key::Ptr(p) => match p.kind {
				value::PtrKind::Value(v) => self.lower_interned_value(v),
				_ => unreachable!("{p:?}"),
			},
			value::Key::EnumTag { val: v, .. } => self.lower_interned_value(*v),
			value::Key::Fn(fun) => self.lower_decl_fn(fun.owner_decl).as_any_value_enum(),
			value::Key::NullPtr => self.ctx.ptr_type(AddressSpace::default()).const_null().as_any_value_enum(),
			value::Key::Aggregate { ty, values } => {
				let r#struct = self.compilation_unit.values.index_to_value(*ty).as_struct();
				let r#struct = r#struct.as_ref();
				if r#struct.is_packed() {
					todo!()
				} else {
					let struct_ty_llvm = self.lower_type(*ty).into_struct_type();
					let values = values
						.iter()
						.map(|value| self.lower_interned_value(*value).try_into().unwrap())
						.collect::<Vec<_>>();
					let value = struct_ty_llvm.const_named_struct(&values);
					value.as_any_value_enum()
				}
			},
			_ => unreachable!("{:?} is not a value and therefore cannot be lowered into a LLVM value", key),
		};

		self.interned_value_to_llvm_value.insert(val, llvm_val);

		llvm_val
	}

	fn lower_type(
		&mut self,
		index: value::Index,
	) -> inkwell::types::AnyTypeEnum<'ctx> {
		if let Some(ty) = self.interned_value_to_llvm_type.get(&index) {
			*ty
		} else {
			let ty = match self.compilation_unit.values.index_to_key_value(index) {
				(value::Key::TypeVoid | value::Key::TypeNever, _) => self.ctx.void_type().into(),
				(value::Key::TypeInt { bits, .. }, _) => self.ctx.custom_width_int_type(*bits as u32).into(),
				(value::Key::Int { ty, .. }, _) => self.lower_type(*ty),
				(value::Key::TypeUsize | value::Key::TypeIsize, _) => self
					.ctx
					.ptr_sized_int_type(&self.target_machine.get_target_data(), None)
					.as_any_type_enum(),
				(value::Key::TypeF16, _) => self.ctx.f16_type().as_any_type_enum(),
				(value::Key::TypeF32, _) => self.ctx.f32_type().as_any_type_enum(),
				(value::Key::TypeF64, _) => self.ctx.f64_type().as_any_type_enum(),
				(value::Key::TypeF128, _) => self.ctx.f128_type().as_any_type_enum(),
				(value::Key::TypeBool, _) => self.ctx.bool_type().as_any_type_enum(),
				(value::Key::TypePtr(_), _) => self.ctx.ptr_type(inkwell::AddressSpace::default()).as_any_type_enum(),
				(value::Key::TypeSlice(_), _) => {
					let ptr_type = self.ctx.ptr_type(inkwell::AddressSpace::default());
					let len_type = self.ctx.ptr_sized_int_type(&self.target_machine.get_target_data(), None);
					self.ctx.struct_type(&[ptr_type.into(), len_type.into()], false).as_any_type_enum()
				},
				(value::Key::TypeArray(array), _) => {
					let elem_ty: BasicTypeEnum = self.lower_type(array.elem_ty).try_into().unwrap();
					elem_ty.array_type(array.len.try_into().unwrap()).as_any_type_enum()
				},
				(value::Key::TypeStruct(_), value::Value::Struct(struct_ty)) => {
					let struct_ty = struct_ty.as_ref();
					if let &value::StructLayout::Packed { storage_bits, .. } = &struct_ty.layout {
						self.ctx.custom_width_int_type(storage_bits).as_any_type_enum()
					} else {
						let field_types = struct_ty
							.fields
							.iter()
							.map(|f| self.lower_type(f.ty).try_into().unwrap())
							.collect::<Vec<_>>();
						let struct_ty = self.ctx.opaque_struct_type(&struct_ty.name);
						struct_ty.set_body(&field_types, false);
						struct_ty.as_any_type_enum()
					}
				},
				(value::Key::TypeEnum(_), value::Value::Enum(r#enum)) => self.lower_type(r#enum.tag_ty),
				(value::Key::TypeUnion(_), value::Value::Union(union_ty)) => {
					let union_ty = union_ty.as_ref();
					let target_data = self.target_machine.get_target_data();

					// Find the largest field size
					let mut max_size = 0u64;
					for field in union_ty.fields {
						if let Some(field_ty) = field.ty {
							let llvm_ty = self.lower_type(field_ty);
							let basic_ty: inkwell::types::BasicTypeEnum = llvm_ty.try_into().unwrap();
							let size = target_data.get_store_size(&basic_ty);
							max_size = max_size.max(size);
						}
					}

					if let Some(tag_ty) = union_ty.tag_ty {
						// Tagged union: { tag, [max_size x i8] }
						let tag_llvm = self.lower_type(tag_ty);
						let tag_basic: inkwell::types::BasicTypeEnum = tag_llvm.try_into().unwrap();
						let payload = self.ctx.i8_type().array_type(max_size as u32);
						self.ctx.struct_type(&[tag_basic, payload.into()], false).as_any_type_enum()
					} else {
						// Bare union: [max_size x i8]
						self.ctx.i8_type().array_type(max_size as u32).as_any_type_enum()
					}
				},
				(value::Key::TypeFn(_), _) => {
					let fn_ty = self.compilation_unit.values.index_to_key(index).as_type_fn();
					let params_tys: Vec<BasicMetadataTypeEnum> = fn_ty
						.params
						.iter()
						.enumerate()
						.filter(|(i, _)| !fn_ty.comptime_params[*i])
						.map(|(i, _)| {
							let llvm_ty = self.lower_type(fn_ty.params[i]);
							if fn_ty.callconv == Some(CallingConvention::C) {
								if let AnyTypeEnum::StructType(st) = llvm_ty {
									let bits = self.target_machine.get_target_data().get_bit_size(&st);
									match bits {
										8 => self.ctx.i8_type().into(),
										16 => self.ctx.i16_type().into(),
										32 => self.ctx.i32_type().into(),
										64 => self.ctx.i64_type().into(),
										_ => self.ctx.ptr_type(inkwell::AddressSpace::default()).into(),
									}
								} else {
									llvm_ty.try_into().unwrap()
								}
							} else {
								llvm_ty.try_into().unwrap()
							}
						})
						.collect();
					match self.lower_type(fn_ty.ret_ty) {
						AnyTypeEnum::IntType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::ArrayType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::FloatType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::ScalableVectorType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::PointerType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::VoidType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::StructType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::VectorType(t) => t.fn_type(&params_tys, fn_ty.var_args),
						AnyTypeEnum::FunctionType(t) => t,
					}
					.as_any_type_enum()
				},
				_ => unreachable!(
					"cannot lower type {:?}, is a comptime type",
					self.compilation_unit.values.index_to_key(index)
				),
			};
			self.interned_value_to_llvm_type.insert(index, ty);
			ty
		}
	}

	fn lower_decl_fn(
		&mut self,
		decl: DeclId,
	) -> AnyValueEnum<'ctx> {
		let (fn_ty_idx, name) = self.compilation_unit.decls.with_mut(|decls| {
			let decl = &decls[decl];
			let ty = match &decl.analysis_state {
				DeclAnalysisState::TypeKnown(ty) => *ty,
				DeclAnalysisState::Analysed { value } => self.compilation_unit.values.type_of_interned(*value),
				_ => {
					unreachable!("encountered a invalid decl in codegen: {decl:?}");
				},
			};

			(ty, decl.name)
		});

		let fn_ty = self.compilation_unit.values.index_to_key(fn_ty_idx).as_type_fn();
		let abi = abi::compute_fn_abi(self, fn_ty);
		let ty = match abi.ret_ty {
			inkwell::types::AnyTypeEnum::IntType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::ArrayType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::FloatType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::ScalableVectorType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::FunctionType(t) => t,
			inkwell::types::AnyTypeEnum::PointerType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::VoidType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::StructType(t) => t.fn_type(&abi.params, fn_ty.var_args),
			inkwell::types::AnyTypeEnum::VectorType(t) => t.fn_type(&abi.params, fn_ty.var_args),
		};

		// TODO(zino): handle extern declarations more explicitly.
		let is_main = &*name == "main";

		// TODO(ldubos): generate a stable unique name for generic function instantiations.
		let mangled_name: String = name.to_string();

		let llvm_fn_value = if fn_ty.external {
			if let Some(existing) = self.module.get_function(&mangled_name) {
				existing
			} else {
				self.module.add_function(&mangled_name, ty, Some(Linkage::External))
			}
		} else if false {
			/// todo was is generic
			if let Some(existing) = self.module.get_function(&mangled_name) {
				existing
			} else {
				self.module.add_function(&mangled_name, ty, Some(Linkage::Private))
			}
		} else {
			self.module.add_function(
				&mangled_name,
				ty,
				if is_main { Some(Linkage::External) } else { Some(Linkage::Private) },
			)
		};

		if let Some(callconv) = fn_ty.callconv {
			llvm_fn_value.set_call_conventions(self.llvm_callconv_id(callconv) as u32);
		}

		// attributes
		if let abi::RetMode::SretFirstParam(ret_ty) = abi.ret_mode {
			let sret_attr = self.ctx.create_type_attribute(self.attributes.sret.get_enum_kind_id(), ret_ty);
			llvm_fn_value.add_attribute(AttributeLoc::Param(0), sret_attr);
		}
		llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.nounwind);
		if fn_ty.ret_ty == self.compilation_unit.values.common.never_t {
			llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.noreturn);
		}
		match fn_ty.inline {
			ast::Inline::Always => {
				llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.alwaysinline);
			},
			ast::Inline::Never => {
				llvm_fn_value.add_attribute(AttributeLoc::Function, self.attributes.noinline);
			},
			ast::Inline::None => {},
		}

		// for windows debugging we need the unwind table
		llvm_fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, self.attributes.uwtable_sync);

		llvm_fn_value.into()
	}

	pub fn lower_function(
		&mut self,
		compilation_unit: &CompilationUnit,
		fun: value::Index,
		vtir: &Vtir,
		build_opts: &Build,
	) {
		let module = {
			let fun = self.compilation_unit.values.index_to_key(fun).as_fn();
			self.compilation_unit.decls.lock()[fun.owner_decl].module
		};

		let di = self.di.take();

		let mut fn_lower_ctx = FnLowerCtx {
			compilation_unit,
			module,
			vtir_inst_to_llvm_value: Default::default(),
			cur_fn: None,
			cur_fn_abi: None,
			cur_fn_param_idx: 0,
			di_gen: di.map(|di| DebugInfoGen {
				di_file: di.module_to_file[&module],
				di_ctx: di,
				di_lexical_block_stack: vec![],
			}),
			vtir_block_to_break_list: Default::default(),
			lowerer: self,
		};
		fn_lower_ctx.lower_fn_body(fun, vtir);

		self.di = fn_lower_ctx.di_gen.map(|di| di.di_ctx);
	}

	pub fn finish(
		mut self,
		build_opts: &Build,
	) -> Result<inkwell::memory_buffer::MemoryBuffer, ()> {
		if let Some(di) = self.di {
			di.builder.finalize();
		}

		// validate and dump any LLVM IR errors
		{
			if let Err(err) = self.module.verify() {
				for err in err.to_string_lossy().split('\n') {
					eprintln!("{}", err);
				}
				self.module.print_to_stderr();
				return Err(());
			}
		}

		// Run LLVM passes to transform IR depending on opt levels
		let pass_opts = PassBuilderOptions::create();
		pass_opts.set_loop_unrolling(true);
		pass_opts.set_loop_vectorization(true);
		pass_opts.set_loop_slp_vectorization(true);

		self.module
			.run_passes(
				match build_opts.opt {
					0 => "default<O0>",
					1 => "default<O1>",
					2 => "default<O2>",
					3 => "default<O3>",
					_ => unreachable!(),
				},
				&self.target_machine,
				PassBuilderOptions::create(),
			)
			.unwrap();

		if build_opts.dump_llvm_ir {
			println!("--- START OF LLVM IR DUMP ---");
			self.module.print_to_stderr();
			println!("--- END OF LLVM IR DUMP ---");
		}

		if build_opts.dump_asm {
			println!("--- START OF ASM DUMP ---");
			let mb = self
				.target_machine
				.write_to_memory_buffer(&self.module, FileType::Assembly)
				.unwrap();
			println!("{}", String::from_utf8_lossy(mb.as_slice()).into_owned());
			println!("--- END OF ASM DUMP ---");
		}

		Ok(self.target_machine.write_to_memory_buffer(&self.module, FileType::Object).unwrap())
	}
}
