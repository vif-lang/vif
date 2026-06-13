//! VUIR (unresolved IR) - The first IR produced after AST generation.
//!
//! VUIR is inspired by Zig ZIR, a bytecode meant for interpretation.
//! It represents unresolved/untyped intermediate code before semantic analysis.

pub mod from_ast;
pub mod opcodes;

use std::{
	pin::Pin,
	sync::SyncView,
};

use bumpalo::Bump;
use internment::Intern;
// Re-export commonly used types
pub use opcodes::*;

use crate::common::{
	IndexVec,
	RcLinearAllocator,
	Span,
};

#[allow(clippy::upper_case_acronyms)]
#[doc(hidden)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct VUIR;

impl super::id::IRMarker for VUIR {}

pub type InstructionId = super::id::InstructionId<VUIR>;
pub type InstructionRef = super::id::InstructionRef<VUIR>;

#[derive(Debug)]
pub struct Import {
	pub path: Intern<[u8]>,
	pub span: Span,
}

#[derive(Debug)]
pub struct Vuir {
	pub instructions: IndexVec<InstructionId, Opcode>,
	pub instructions_payload_allocator: Box<SyncView<Bump>>,
	pub imports: &'static [Import],
}

impl Vuir {
	pub fn pretty_print(
		&self,
		stream: &mut dyn std::io::Write,
	) -> std::io::Result<()> {
		struct Printer<'a> {
			vuir: &'a Vuir,
			stream: &'a mut dyn std::io::Write,
			indent: String,
		}

		impl<'a> Printer<'a> {
			fn print_declaration(
				&mut self,
				id: InstructionId,
			) -> std::io::Result<()> {
				let Opcode::Declaration(decl) = &self.vuir.instructions[id] else {
					unreachable!()
				};
				write!(self.stream, "decl {id} '{}' = ", decl.name)?;
				writeln!(self.stream, "{{")?;

				self.push_indent();
				self.print_body(decl.value)?;
				self.pop_indent();

				self.print_indent()?;
				writeln!(self.stream, "}}")?;
				Ok(())
			}

			fn print_body(
				&mut self,
				body: &[InstructionId],
			) -> std::io::Result<()> {
				for inst in body {
					self.print_indent()?;
					self.print_inst(*inst)?;
					writeln!(self.stream)?;
				}
				Ok(())
			}

			fn print_inst_ref(
				&mut self,
				r: InstructionRef,
			) -> std::io::Result<()> {
				match r {
					InstructionRef::Instruction(id) => self.print_inst(id),
					_ => write!(self.stream, "{r:?}"),
				}
			}

			fn print_inst(
				&mut self,
				id: InstructionId,
			) -> std::io::Result<()> {
				write!(self.stream, "{id} = ")?;
				match &self.vuir.instructions[id] {
					Opcode::BlockComptime { instructions } => {
						write!(self.stream, "BlockComptime")?;
						writeln!(self.stream, "{{")?;
						self.push_indent();
						self.print_body(instructions)?;
						self.pop_indent();

						self.print_indent()?;
						write!(self.stream, "}}")?;
					},
					Opcode::DeclFn {
						ret_ty,
						params,
						body,
						builtin,
						callconv,
						..
					} => {
						self.push_indent();

						write!(self.stream, "Func (builtin: {:?}) {{ ret_ty = ", builtin)?;
						self.print_inst(*ret_ty)?;

						if let Some(callconv) = callconv {
							writeln!(self.stream, ", callconv= ")?;
							self.print_inst(*callconv)?;
						}

						writeln!(self.stream, ", params = {{")?;
						{
							self.push_indent();
							self.print_body(params)?;
							self.pop_indent();
							self.print_indent()?;
							write!(self.stream, "}}")?;
						}

						writeln!(self.stream, ", body = {{")?;
						{
							self.push_indent();
							self.print_body(body)?;
							self.pop_indent();
							self.print_indent()?;
							writeln!(self.stream, "}}")?;
						}

						self.pop_indent();
						self.print_indent()?;
						write!(self.stream, "}}")?;
					},
					Opcode::DeclFnParam {
						name,
						type_body,
						comptime,
						generic,
						span,
					} => {
						self.push_indent();

						writeln!(
							self.stream,
							"DeclFnParam {{ name: {:?}, comptime: {:?}, generic: {:?}, type_body = {{",
							name, generic, comptime
						)?;

						{
							self.push_indent();
							self.print_body(type_body)?;
							self.pop_indent();
						}

						self.pop_indent();
						self.print_indent()?;
						writeln!(self.stream, "}}, span = {:?} }}", span)?;
					},
					Opcode::FnCallWithFieldPtrReceiver {
						field_ptr,
						field_name,
						generic_args,
						args,
						ret_ty,
						span,
					} => {
						self.push_indent();

						write!(
							self.stream,
							"FnCallWithFieldPtrReceiver {{ field_ptr: {:?}, field_name: {:?}, generic_args: {:?}, args = [",
							field_ptr, field_name, generic_args
						)?;

						for arg in args.iter() {
							writeln!(self.stream, "name = {:?}, body = {{", arg.name)?;

							self.push_indent();
							self.print_body(arg.body)?;
							self.pop_indent();

							write!(self.stream, "}}, span = {:?},", arg.span)?;
						}

						self.pop_indent();
						self.print_indent()?;
						writeln!(self.stream, "], ret_ty = {:?}, span = {:?} }}", ret_ty, span)?;
					},
					Opcode::DeclStruct { fields, decls, .. } => {
						self.push_indent();

						writeln!(self.stream, "struct {{")?;
						for field in fields {
							self.print_indent()?;
							writeln!(self.stream, "field `{}`: {:?}", field.name, field.ty)?;
						}
						for decl in decls {
							self.print_indent()?;
							self.print_declaration(*decl)?;
						}
						writeln!(self.stream, "}}")?;

						self.pop_indent();
					},
					Opcode::Block { instructions, .. } => {
						writeln!(self.stream, "Block {{")?;

						self.push_indent();
						self.print_body(instructions)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}}")?;
					},
					Opcode::Loop { instructions, .. } => {
						writeln!(self.stream, "Loop {{")?;

						self.push_indent();
						self.print_body(instructions)?;
						self.pop_indent();

						self.print_indent()?;
						writeln!(self.stream, "}}")?;
					},
					Opcode::Branch {
						cond,
						then_body,
						else_body,
						..
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
				self.print_inst(InstructionId::FILE_MODULE)
			}
		}

		Printer {
			vuir: self,
			stream,
			indent: "".into(),
		}
		.pretty_print()
	}
}
