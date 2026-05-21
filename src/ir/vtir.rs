//! VTIR (typed IR) - The typed intermediate representation.
//!
//! VTIR is produced after semantic analysis of VUIR. It contains type information
//! and is ready for code generation.

pub mod opcodes;

use std::sync::SyncView;

use bumpalo::Bump;
// Re-export commonly used types
pub use opcodes::*;
use rustc_hash::FxHashMap;

use crate::{
	common::{
		IndexVec,
		RcLinearAllocator,
	},
	value::{
		self,
		ValueStore,
	},
};

#[allow(clippy::upper_case_acronyms)]
#[doc(hidden)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct VTIR;

impl super::id::IRMarker for VTIR {}

pub type InstructionId = super::id::InstructionId<VTIR>;
pub type InstructionRef = super::id::InstructionRef<VTIR>;

#[derive(Debug)]
pub struct Vtir {
	pub instructions: IndexVec<InstructionId, Opcode>,
	pub instructions_payload_allocator: Box<SyncView<Bump>>,
	pub main_body: &'static [InstructionRef],
}

impl Vtir {
	pub fn type_of(
		&self,
		values: &ValueStore,
		inst: &InstructionRef,
	) -> value::Index {
		type_of(values, &self.instructions, inst)
	}

	pub fn pretty_print(
		&self,
		stream: &mut dyn std::io::Write,
	) -> std::io::Result<()> {
		struct Printer<'a> {
			vtir: &'a Vtir,
			stream: &'a mut dyn std::io::Write,
			indent: String,
		}

		impl<'a> Printer<'a> {
			fn print_body(
				&mut self,
				body: &[InstructionRef],
			) -> std::io::Result<()> {
				for inst in body {
					self.print_indent()?;
					self.print_inst_ref(inst)?;
					writeln!(self.stream)?;
				}
				Ok(())
			}

			fn print_inst_ref(
				&mut self,
				r: &InstructionRef,
			) -> std::io::Result<()> {
				match r {
					InstructionRef::Instruction(id) => self.print_inst(id),
					_ => write!(self.stream, "{r:?}"),
				}
			}

			fn print_inst(
				&mut self,
				id: &InstructionId,
			) -> std::io::Result<()> {
				write!(self.stream, "{id} = ")?;
				match &self.vtir.instructions[id] {
					Opcode::Branch {
						cond,
						then_body,
						else_body,
					} => {
						write!(self.stream, "Branch {{ ")?;
						self.push_indent();

						writeln!(self.stream, "cond = {:?},", cond)?;

						self.print_indent()?;
						writeln!(self.stream, "then_body = {{")?;

						self.push_indent();
						self.print_body(then_body)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}},")?;

						self.print_indent()?;
						writeln!(self.stream, "else_body = {{")?;
						self.push_indent();
						self.print_body(else_body)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}}")?;

						self.pop_indent();
						self.print_indent()?;
						writeln!(self.stream, "}}")?;
					},
					Opcode::Block { instructions, ret_ty, .. } => {
						write!(self.stream, "Block {{ ")?;
						self.push_indent();

						writeln!(self.stream, "ret_ty = {:?},", ret_ty)?;

						self.print_indent()?;
						writeln!(self.stream, "instructions = {{")?;

						self.push_indent();
						self.print_body(instructions)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}}")?;

						self.pop_indent();
						self.print_indent()?;
						writeln!(self.stream, "}}")?;
					},
					Opcode::Loop { instructions, ret_ty, .. } => {
						write!(self.stream, "Loop {{ ")?;
						self.push_indent();

						writeln!(self.stream, "ret_ty = {:?},", ret_ty)?;

						self.print_indent()?;
						writeln!(self.stream, "instructions = {{")?;

						self.push_indent();
						self.print_body(instructions)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}}")?;

						self.pop_indent();
						self.print_indent()?;
						writeln!(self.stream, "}}")?;
					},
					opcode => {
						write!(self.stream, "{:?}", opcode)?;
					},
				};
				Ok(())
			}

			fn push_indent(&mut self) {
				self.indent.push(' ');
				self.indent.push(' ');
			}

			fn pop_indent(&mut self) {
				self.indent.pop();
				self.indent.pop();
			}

			fn print_indent(&mut self) -> std::io::Result<()> {
				write!(self.stream, "{}", self.indent)
			}

			#[inline(always)]
			fn pretty_print(mut self) -> std::io::Result<()> {
				self.push_indent();
				self.print_body(self.vtir.main_body)?;
				self.pop_indent();
				Ok(())
			}
		}

		Printer {
			vtir: self,
			stream,
			indent: "".into(),
		}
		.pretty_print()
	}
}
