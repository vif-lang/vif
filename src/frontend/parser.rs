use core::hint::{
	likely,
	unlikely,
	unreachable_unchecked,
};

use internment::Intern;
use rustc_hash::FxHashSet;
use sorted_insert::SortedInsertBy;

use super::{
	ast::*,
	lexer::*,
};
use crate::{
	assume,
	common::{
		COMMON_INTERNS,
		RcLinearAllocator,
		Span,
		diagnostic::*,
	},
	compile_unit::module::ModuleId,
	frontend::IdentKind,
};

// =============================================================================
//                              Infix Operators
// =============================================================================

/// Infix operator with binding powers.
#[derive(Copy, Clone, Debug)]
pub struct InfixOp {
	/// Left binding power (determines associativity).
	pub lbp: u8,
	/// Right binding power.
	pub rbp: u8,
	/// AST constructor for this operator.
	pub kind: InfixOpKind,
}

/// The kind of infix operation.
#[derive(Copy, Clone, Debug)]
pub enum InfixOpKind {
	Add,
	AddSat,
	AddWrap,
	Sub,
	SubSat,
	SubWrap,
	Mul,
	MulSat,
	MulWrap,
	Pow,
	PowSat,
	PowWrap,
	Div,
	Rem,
	Shl,
	ShlSat,
	ShlWrap,
	Shr,
	ShrSat,
	ShrWrap,
	BitAnd,
	BitOr,
	BitXor,
	Lt,
	Lte,
	Gt,
	Gte,
	BoolAnd,
	BoolOr,
	Eq,
	Neq,
}

impl InfixOp {
	#[inline(always)]
	const fn left(
		lbp: u8,
		kind: InfixOpKind,
	) -> Self {
		Self { lbp, rbp: lbp + 1, kind }
	}

	#[inline(always)]
	const fn right(
		lbp: u8,
		kind: InfixOpKind,
	) -> Self {
		Self { lbp, rbp: lbp, kind }
	}
}

// =============================================================================
//                              Parsing Context
// =============================================================================

/// Determines the parsing mode: type-expression context or value-expression context.
///
/// - `Ty`: Parses expressions in a type context (uses `parse_ty_expr`, `parse_ty_statement`).
/// - `Value`: Parses expressions in a value context (uses `parse_expr`, `parse_statement`).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Ctx {
	/// Type-expression context.
	Ty,
	/// Value-expression context.
	Value,
}

// =============================================================================
//                                   Parser
// =============================================================================

struct CommonTypes {
	generic_ty: &'static Type,
	bool_ty: &'static Type,
	void_ty: &'static Type,
	type_ty: &'static Type,
	any_ty: &'static Type,
	anyint_ty: &'static Type,
	anyfloat_ty: &'static Type,
	anyerror_ty: &'static Type,
	never_ty: &'static Type,
	f16_ty: &'static Type,
	f32_ty: &'static Type,
	f64_ty: &'static Type,
	f128_ty: &'static Type,
	usize_ty: &'static Type,
	isize_ty: &'static Type,
	u8_ty: &'static Type,
	u16_ty: &'static Type,
	u32_ty: &'static Type,
	u64_ty: &'static Type,
	u128_ty: &'static Type,
	i8_ty: &'static Type,
	i16_ty: &'static Type,
	i32_ty: &'static Type,
	i64_ty: &'static Type,
	i128_ty: &'static Type,
}

pub struct Parser {
	errors: Vec<Diagnostic>,
	next_id: u32,
	tokens: Vec<Token, RcLinearAllocator>,
	data: ModuleData,
	offset: usize,
	module_id: ModuleId,
	common_types: CommonTypes,
	linear_alloc: RcLinearAllocator,
}

impl Parser {
	#[inline(always)]
	pub fn new(
		code: &str,
		module_id: ModuleId,
	) -> Self {
		// Heuristic: ~1 token per 5 bytes of source on average.
		// This accounts for whitespace, multi-char operators, identifiers, etc.
		let estimated_tokens = (code.len() / 5).max(256);
		let linear_alloc = RcLinearAllocator::new(bumpalo::Bump::with_capacity(estimated_tokens * 2 * core::mem::size_of::<Token>()));

		let mut tokens = Vec::with_capacity_in(estimated_tokens, linear_alloc.clone());
		let mut lexer = Lexer::new(code, module_id);

		loop {
			let token = lexer.next();
			let is_eof = token.is_eof();

			tokens.push(token);

			if unlikely(is_eof) {
				break;
			}
		}

		let mut data = ModuleData::new();
		let common_types = CommonTypes {
			generic_ty: data.push(&Type::Generic),
			bool_ty: data.push(&Type::Bool),
			void_ty: data.push(&Type::Void),
			type_ty: data.push(&Type::Type),
			any_ty: data.push(&Type::Any),
			anyint_ty: data.push(&Type::Anyint),
			anyfloat_ty: data.push(&Type::Anyfloat),
			anyerror_ty: data.push(&Type::Anyerror),
			never_ty: data.push(&Type::Never),
			f16_ty: data.push(&Type::Float(FloatSuffix::F16)),
			f32_ty: data.push(&Type::Float(FloatSuffix::F32)),
			f64_ty: data.push(&Type::Float(FloatSuffix::F64)),
			f128_ty: data.push(&Type::Float(FloatSuffix::F128)),
			usize_ty: data.push(&Type::Int(IntSuffix::Usize)),
			isize_ty: data.push(&Type::Int(IntSuffix::Isize)),
			u8_ty: data.push(&Type::Int(IntSuffix::U(8))),
			u16_ty: data.push(&Type::Int(IntSuffix::U(16))),
			u32_ty: data.push(&Type::Int(IntSuffix::U(32))),
			u64_ty: data.push(&Type::Int(IntSuffix::U(64))),
			u128_ty: data.push(&Type::Int(IntSuffix::U(128))),
			i8_ty: data.push(&Type::Int(IntSuffix::I(8))),
			i16_ty: data.push(&Type::Int(IntSuffix::I(16))),
			i32_ty: data.push(&Type::Int(IntSuffix::I(32))),
			i64_ty: data.push(&Type::Int(IntSuffix::I(64))),
			i128_ty: data.push(&Type::Int(IntSuffix::I(128))),
		};

		Self {
			errors: lexer.take_errors(),
			next_id: 0,
			tokens,
			data,
			offset: 0,
			module_id,
			common_types,
			linear_alloc,
		}
	}

	#[inline(always)]
	fn diag_span(
		&self,
		span: Span,
	) -> DiagSpan {
		DiagSpan {
			module: self.module_id,
			span,
		}
	}

	#[inline(always)]
	pub fn parse_module(&mut self) -> Result<Module, Vec<Diagnostic>> {
		let mut module = Module {
			kind: ModuleKind::None,
			data: ModuleData::zeroed(),
		};

		if unlikely(self.peek().is_eof()) {
			module.kind = ModuleKind::StructDecl(StructTy {
				generics: &[],
				params: &[],
				fields: &[],
				associated_items: &[],
				packed: false,
				linear: false,
				span: self.peek().span,
			});
			core::mem::swap(&mut module.data, &mut self.data);

			// Empty file, return empty struct.
			return Ok(module);
		}

		let struct_decl = match self.parse_struct_decl(true, false, false, self.peek().span) {
			Ok(decl) => decl,
			Err(err) => {
				self.push_error(err);
				return Err(core::mem::take(&mut self.errors));
			},
		};

		if !self.errors.is_empty() {
			return Err(core::mem::take(&mut self.errors));
		}

		module.kind = ModuleKind::StructDecl(struct_decl);
		core::mem::swap(&mut module.data, &mut self.data);
		Ok(module)
	}

	fn parse_struct_decl(
		&mut self,
		is_root: bool,
		is_packed: bool,
		is_linear: bool,
		start_span: Span,
	) -> Result<StructTy, Diagnostic> {
		let (generics, params) = if self.peek().kind == TokenKind::LParen {
			let (generics, params, ..) = self.parse_fn_decl(true)?;

			if generics.is_empty() && params.is_empty() {
				self.push_error(
					Diagnostic::error()
						.with_message("struct generic parameters cannot be empty")
						.with_label(
							Label::primary()
								.with_span(self.diag_span(Span::new(self.prev().span.start() - 1..self.prev().span.end())))
								.with_message("empty parameters list"),
						)
						.with_note("remove `()` or add generic parameters and/or associated const parameters"),
				);
			}

			(generics, params)
		} else {
			(self.data.push_slice(&[]), self.data.push_slice(&[]))
		};

		if !is_root {
			match self.eat_expect(TokenTag::LBrace) {
				Ok(_) => {},
				Err(err) => {
					self.eat_until(TokenTag::RBrace);
					return Err(err);
				},
			}
		}

		let mut fields = Vec::new_in(self.linear_alloc.clone());
		let mut associated_items = Vec::new_in(self.linear_alloc.clone());

		let mut prev_field_have_comma = true;

		while likely((is_root || self.peek().kind != TokenTag::RBrace) && self.peek().kind != TokenTag::Eof) {
			if !prev_field_have_comma {
				// Avoid repeating the error for following entries
				prev_field_have_comma = true;

				let tok = self.peek();
				self.push_error(self.diag_expected_token(TokenTag::Comma, tok));
			}

			let start_span = self.peek().span;
			let is_pub = self.eat_if(TokenTag::KwPub).is_some();

			let (ext, inline) = self.parse_decl_modifiers();

			let is_comptime = self.eat_if(TokenTag::KwComptime).is_some();
			let is_const = self.eat_if(TokenTag::KwConst).is_some();

			if self.eat_if(TokenTag::KwFn).is_some() {
				match self.parse_associated_function(is_pub, is_comptime, ext, inline, start_span, false) {
					Ok(item) => associated_items.push(item),
					Err(err) => self.push_error(err),
				}
				continue;
			}

			if is_const {
				match self.parse_associated_const(is_pub, start_span) {
					Ok(item) => associated_items.push(item),
					Err(err) => self.push_error(err),
				}
				continue;
			}

			match self.peek().kind {
				TokenKind::Ident { .. } => match self.parse_field_def(is_pub, start_span) {
					Ok((field, have_trailing_comma)) => {
						fields.push(field);
						prev_field_have_comma = have_trailing_comma;
					},
					Err(err) => self.push_error(err),
				},
				_ => {
					let tok = self.bump();
					self.push_error(self.diag_unexpected_token(&tok));
				},
			}
		}

		if !is_root {
			self.eat_expect(TokenTag::RBrace)?;
		}

		let fields = self.data.push_slice(&fields);
		let associated_items = self.data.push_slice(&associated_items);
		let end_span = self.peek().span;

		Ok(StructTy {
			generics,
			params,
			fields,
			associated_items,
			packed: is_packed,
			linear: is_linear,
			span: (start_span, end_span).into(),
		})
	}

	fn parse_enum_decl(
		&mut self,
		is_root: bool,
		is_packed: bool,
		is_linear: bool,
		start_span: Span,
	) -> Result<EnumTy, Diagnostic> {
		let tag_ty = if self.peek().kind == TokenKind::LParen {
			self.eat_expect(TokenTag::LParen)?;
			let expr = self.expect_expr()?;
			self.eat_expect(TokenTag::RParen)?;
			Some(self.data.push(&expr))
		} else {
			None
		};

		if !is_root {
			match self.eat_expect(TokenTag::LBrace) {
				Ok(_) => {},
				Err(err) => {
					self.eat_until(TokenTag::RBrace);
					return Err(err);
				},
			}
		}

		let mut variants = Vec::new_in(self.linear_alloc.clone());
		let mut associated_items = Vec::new_in(self.linear_alloc.clone());

		let mut prev_variant_have_comma = true;

		while likely((is_root || self.peek().kind != TokenTag::RBrace) && self.peek().kind != TokenTag::Eof) {
			if !prev_variant_have_comma {
				// Avoid repeating the error for following entries
				prev_variant_have_comma = true;

				let tok = self.peek();
				self.push_error(self.diag_expected_token(TokenTag::Comma, tok));
			}

			let start_span = self.peek().span;
			let is_pub = self.eat_if(TokenTag::KwPub).is_some();

			let (ext, inline) = self.parse_decl_modifiers();

			// is this a fn ?
			let is_const = self.eat_if(TokenTag::KwConst).is_some();
			if self.eat_if(TokenTag::KwFn).is_some() {
				match self.parse_associated_function(is_pub, is_const, ext, inline, start_span, false) {
					Ok(item) => associated_items.push(item),
					Err(err) => self.push_error(err),
				}
				continue;
			}

			// no, a field
			match self.peek().kind {
				TokenKind::Ident { .. } => match self.parse_enum_variant_def(start_span) {
					Ok((field, have_trailing_comma)) => {
						variants.push(field);
						prev_variant_have_comma = have_trailing_comma;
					},
					Err(err) => self.push_error(err),
				},
				_ => {
					let tok = self.bump();
					self.push_error(self.diag_unexpected_token(&tok));
				},
			}
		}

		if !is_root {
			self.eat_expect(TokenTag::RBrace)?;
		}

		let variants = self.data.push_slice(&variants);
		let associated_items = self.data.push_slice(&associated_items);
		let end_span = self.peek().span;

		Ok(EnumTy {
			tag_ty,
			variants,
			associated_items,
			linear: is_linear,
			span: (start_span, end_span).into(),
		})
	}

	fn parse_enum_variant_def(
		&mut self,
		start_span: Span,
	) -> Result<(EnumVariantDef, bool), Diagnostic> {
		let id = self.next_id();
		let ident = match self.parse_ident() {
			Ok(id) => id,
			Err(err) => {
				self.eat_until2(TokenTag::Colon, TokenTag::Comma);
				self.push_error(err);
				Ident {
					symbol: COMMON_INTERNS.empty_str,
					kind: IdentKind::User,
					span: self.prev().span,
				}
			},
		};

		// has tag value ?
		let value = if self.eat_if(TokenTag::Eq).is_some() {
			let expr = self.expect_expr()?;
			Some(self.data.push(&expr))
		} else {
			None
		};

		let have_trailing_comma = self.eat_if(TokenTag::Comma).is_some();
		let span = (start_span, self.prev().span).into();
		Ok((EnumVariantDef { id, ident, value, span }, have_trailing_comma))
	}

	fn parse_union_decl(
		&mut self,
		is_root: bool,
		is_linear: bool,
		start_span: Span,
	) -> Result<UnionTy, Diagnostic> {
		let tag = if self.peek().kind == TokenKind::LParen {
			self.eat_expect(TokenTag::LParen)?;
			if self.eat_if(TokenTag::KwEnum).is_some() {
				self.eat_expect(TokenTag::RParen)?;
				UnionTagKind::AutoEnum
			} else {
				let expr = self.expect_expr()?;
				self.eat_expect(TokenTag::RParen)?;
				UnionTagKind::Enum(self.data.push(&expr))
			}
		} else {
			UnionTagKind::Bare
		};

		if !is_root {
			match self.eat_expect(TokenTag::LBrace) {
				Ok(_) => {},
				Err(err) => {
					self.eat_until(TokenTag::RBrace);
					return Err(err);
				},
			}
		}

		let mut fields = Vec::new_in(self.linear_alloc.clone());
		let mut associated_items = Vec::new_in(self.linear_alloc.clone());

		let mut prev_field_have_comma = true;

		while likely((is_root || self.peek().kind != TokenTag::RBrace) && self.peek().kind != TokenTag::Eof) {
			if !prev_field_have_comma {
				prev_field_have_comma = true;

				let tok = self.peek();
				self.push_error(self.diag_expected_token(TokenTag::Comma, tok));
			}

			let start_span = self.peek().span;
			let is_pub = self.eat_if(TokenTag::KwPub).is_some();

			let (ext, inline) = self.parse_decl_modifiers();

			let is_const = self.eat_if(TokenTag::KwConst).is_some();
			if self.eat_if(TokenTag::KwFn).is_some() {
				match self.parse_associated_function(is_pub, is_const, ext, inline, start_span, false) {
					Ok(item) => associated_items.push(item),
					Err(err) => self.push_error(err),
				}
				continue;
			}

			if is_const {
				match self.parse_associated_const(is_pub, start_span) {
					Ok(item) => associated_items.push(item),
					Err(err) => self.push_error(err),
				}
				continue;
			}

			match self.peek().kind {
				TokenKind::Ident { .. } => match self.parse_union_field_def(start_span) {
					Ok((field, have_trailing_comma)) => {
						fields.push(field);
						prev_field_have_comma = have_trailing_comma;
					},
					Err(err) => self.push_error(err),
				},
				_ => {
					let tok = self.bump();
					self.push_error(self.diag_unexpected_token(&tok));
				},
			}
		}

		if !is_root {
			self.eat_expect(TokenTag::RBrace)?;
		}

		let fields = self.data.push_slice(&fields);
		let associated_items = self.data.push_slice(&associated_items);
		let end_span = self.peek().span;

		Ok(UnionTy {
			tag,
			fields,
			associated_items,
			linear: is_linear,
			span: (start_span, end_span).into(),
		})
	}

	fn parse_union_field_def(
		&mut self,
		start_span: Span,
	) -> Result<(UnionFieldDef, bool), Diagnostic> {
		let id = self.next_id();
		let ident = match self.parse_ident() {
			Ok(id) => id,
			Err(err) => {
				self.eat_until2(TokenTag::Colon, TokenTag::Comma);
				self.push_error(err);
				Ident {
					symbol: COMMON_INTERNS.empty_str,
					kind: IdentKind::User,
					span: self.prev().span,
				}
			},
		};

		let ty = if self.eat_if(TokenTag::Colon).is_some() {
			Some(self.expect_ty_expr().map(|ty| self.data.push(&ty))?)
		} else {
			None
		};

		let have_trailing_comma = self.eat_if(TokenTag::Comma).is_some();
		let span = (start_span, self.prev().span).into();
		Ok((UnionFieldDef { id, ident, ty, span }, have_trailing_comma))
	}

	fn parse_field_def(
		&mut self,
		is_pub: bool,
		start_span: Span,
	) -> Result<(FieldDef, bool), Diagnostic> {
		let id = self.next_id();
		let ident = match self.parse_ident() {
			Ok(id) => id,
			Err(err) => {
				self.eat_until2(TokenTag::Colon, TokenTag::Comma);
				self.push_error(err);
				Ident {
					symbol: COMMON_INTERNS.empty_str,
					kind: IdentKind::User,
					span: self.prev().span,
				}
			},
		};

		match self.eat_expect(TokenTag::Colon) {
			Ok(_) => {},
			Err(err) => {
				self.eat_until(TokenTag::Comma);
				return Err(err);
			},
		}
		let ty = self.expect_ty_expr().map(|ty| self.data.push(&ty))?;

		let default = if self.eat_if(TokenTag::Eq).is_some() {
			self.expect_expr().map(|expr| self.data.push(&expr)).map(Some)?
		} else {
			None
		};

		let have_trailing_comma = self.eat_if(TokenTag::Comma).is_some();

		let span = (start_span, self.prev().span).into();
		Ok((
			FieldDef {
				id,
				ident,
				ty,
				is_pub,
				default,
				span,
			},
			have_trailing_comma,
		))
	}

	fn parse_associated_const(
		&mut self,
		is_pub: bool,
		start_span: Span,
	) -> Result<AssociatedItem, Diagnostic> {
		let id = self.next_id();
		let var_binding = self.parse_var_binding()?;

		let span = (start_span, self.prev().span).into();
		Ok(AssociatedItem {
			id,
			kind: AssociatedItemKind::Const(var_binding),
			is_pub,
			span,
		})
	}

	fn parse_associated_function(
		&mut self,
		is_pub: bool,
		is_comptime: bool,
		ext: Extern,
		inline: Inline,
		start_span: Span,
		allow_bodyless: bool,
	) -> Result<AssociatedItem, Diagnostic> {
		let id = self.next_id();
		let ident = self.parse_ident()?;

		if ident.is_generic() {
			return Err(Diagnostic::error()
				.with_message("associated function name cannot be a generic identifier")
				.with_label(Label::primary().with_span(self.diag_span(ident.span))));
		}

		let (generics, params, callconv, ret_ty, variadic) = self.parse_fn_decl(false)?;
		let ret_ty = ret_ty.unwrap();

		let requires = if self.eat_if(TokenTag::KwRequires).is_some() {
			let mut requires = Vec::new_in(self.linear_alloc.clone());

			loop {
				let expr = self.expect_expr()?;
				requires.push(expr);

				if self.eat_if(TokenTag::Comma).is_none() {
					break;
				}
			}

			self.data.push_slice(&requires)
		} else {
			self.data.push_slice(&[])
		};

		let has_body = self.peek().kind == TokenTag::LBrace;

		if !has_body && (ext == Extern::None) && !allow_bodyless {
			self.eat_expect(TokenTag::Semicolon)?;
			return Err(self.diag_expected_token(TokenTag::LBrace, self.prev()));
		}

		if has_body && (ext != Extern::None) {
			self.push_error(
				Diagnostic::error()
					.with_message("extern functions cannot have a body")
					.with_label(Label::primary().with_span(self.diag_span(ident.span)))
					.with_note("remove the body or remove the `extern` modifier"),
			);
		}

		let block = if has_body {
			self.parse_block_impl(None, false, self.peek().span, Ctx::Value).map(Some)?
		} else {
			self.eat_expect(TokenTag::Semicolon)?;
			None
		};

		Ok(AssociatedItem {
			id,
			is_pub,
			kind: AssociatedItemKind::Fn(Fn {
				ident,
				comptime: is_comptime,
				requires,
				generics,
				params,
				callconv,
				inline,
				variadic,
				ext,
				ret_ty,
				block,
			}),
			span: (start_span, self.prev().span).into(),
		})
	}

	#[allow(clippy::type_complexity)]
	fn parse_fn_decl(
		&mut self,
		is_type_decl: bool,
	) -> Result<
		(
			&'static [Generic],
			&'static [FnParam],
			Option<&'static Expr>,
			Option<&'static Expr>,
			bool,
		),
		Diagnostic,
	> {
		match self.eat_expect(TokenTag::LParen) {
			Ok(_) => {},
			Err(err) => {
				self.push_error(err);
			},
		}

		let mut last_param = None;

		let mut generics = Vec::new_in(self.linear_alloc.clone());
		let mut generics_set: FxHashSet<Ident> = FxHashSet::default();
		let mut params = Vec::new_in(self.linear_alloc.clone());

		loop {
			let token = self.peek();

			if matches!(token.kind, TokenKind::RParen | TokenKind::Ellipsis) || unlikely(token.is_eof()) {
				break;
			}

			let id = self.next_id();

			let comptime = self.eat_if(TokenTag::KwComptime).is_some();

			// it can be tempting to write const instead of comptime or write `comptime const`
			if unlikely(self.peek().kind == TokenTag::KwConst) {
				self.push_error(
					Diagnostic::error()
						.with_message("function parameters are `const` by default")
						.with_label(
							Label::primary()
								.with_span(self.diag_span(self.peek().span))
								.with_message("remove the `const` modifier"),
						)
						.with_note("perhaps you meant `comptime` instead to have a comptime-known parameter"),
				);
				self.bump();
			}

			let ident = self.parse_ident()?;

			if unlikely(ident.is_generic()) {
				if let Some(last_param_span) = last_param {
					self.push_error(
						Diagnostic::error()
							.with_message("positional generic parameters must come before typed parameters")
							.with_label(Label::secondary().with_span(self.diag_span(last_param_span)))
							.with_label(Label::primary().with_span(self.diag_span(ident.span)))
							.with_note("move this generic parameter before the typed parameters"),
					);
				}

				if unlikely(comptime) {
					self.push_error(
						Diagnostic::error()
							.with_message("positional generic parameters cannot be comptime")
							.with_label(
								Label::primary()
									.with_span(self.diag_span(self.prev_nth(1).span))
									.with_message("remove the `comptime` modifier"),
							),
					);
				}

				if !generics_set.insert(ident) {
					self.push_error(
						Diagnostic::error()
							.with_message("duplicate positional generic parameter")
							.with_label(Label::primary().with_span(self.diag_span(ident.span)))
							.with_note("positional generic parameters must have unique names"),
					);
				} else {
					generics.push(ident);
				}
			} else {
				last_param = Some(ident.span);

				if unlikely(is_type_decl && !comptime) {
					self.push_error(
						Diagnostic::error()
							.with_message("type parameters must be comptime")
							.with_label(Label::primary().with_span(self.diag_span(self.peek().span)))
							.with_note("add the `const` modifier to the type parameter"),
					);
				}

				self.eat_expect(TokenTag::Colon)?;
				let ty = self.expect_ty_expr().map(|ty| self.data.push(&ty))?;

				if let Some(ident) = ty.as_generic_ident() {
					// Here we don't error on duplicate generic idents,
					// because they might be used multiple on "concrete" types.
					if generics_set.insert(*ident) {
						generics.push(*ident);
					}
				}

				let default = if unlikely(self.eat_if(TokenTag::Eq).is_some()) {
					let default_expr = self.expect_expr()?;
					let default_expr = self.data.push(&default_expr);
					Some(default_expr)
				} else {
					None
				};

				params.push(FnParam {
					id,
					comptime,
					ident,
					ty,
					default,
				});
			}

			if self.eat_if(TokenTag::Comma).is_none() {
				break;
			}
		}

		let params = self.data.push_slice(&params);

		let variadic = self.eat_if(TokenTag::Ellipsis).is_some();
		if unlikely(variadic && is_type_decl) {
			self.push_error(
				self.diag_unexpected_token(self.prev())
					.with_note("variadic parameters are not allowed here"),
			);
		}

		match self.eat_expect(TokenTag::RParen) {
			Ok(_) => {},
			Err(err) => {
				self.push_error(err);
			},
		}

		let callconv = if !is_type_decl && self.eat_if(TokenTag::DirCallconv).is_some() {
			self.eat_expect(TokenTag::LParen)?;
			let expr = self.expect_expr()?;
			let parsed = Some(self.data.push(&expr));

			self.eat_expect(TokenTag::RParen)?;
			parsed
		} else {
			None
		};

		let ret_ty = if !is_type_decl {
			let ret_ty = self.expect_ty_expr().map(|ty| self.data.push(&ty))?;

			if let Some(ident) = ret_ty.as_generic_ident() {
				// Here we don't error on duplicate either generic idents,
				// because they might be used multiple on "concrete" types.
				if generics_set.insert(*ident) {
					generics.push(*ident);
				}
			}

			Some(ret_ty)
		} else {
			None
		};

		// NOTE(ldubos): We should be careful with this ID thing, and we should ensure to
		// always set the correct `next_id` after using it.
		let mut id = self.next_id();
		let generics = self.data.push_slice_from(generics.iter().map(|&ident| {
			let generic = Generic { ident, id };
			id += 1;
			generic
		}));
		self.next_id = id.as_u32();

		Ok((generics, params, callconv, ret_ty, variadic))
	}

	// =========================================================================
	//                               Expressions
	// =========================================================================

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn expect_ty_expr(&mut self) -> Result<Expr, Diagnostic> {
		let expr = match self.parse_ty_expr()? {
			Some(expr) => expr,
			None => return Err(self.diag_expected_type_expr(self.prev(), self.peek())),
		};
		Ok(expr)
	}

	#[inline(always)]
	fn parse_ty_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		let tok = self.peek();
		let start_span = tok.span;

		let kind = match tok.kind {
			TokenKind::QuestionMark | TokenKind::Star | TokenKind::StarStar | TokenKind::LBracket => become self.parse_prefixed_ty_expr(),
			// error union: E!T
			_ => {
				let Some(err_ty) = self.parse_suffix_ty_expr()? else {
					return Ok(None);
				};
				if self.eat_if(TokenTag::Bang).is_none() {
					return Ok(Some(err_ty));
				}

				let err_ty = self.data.push(&err_ty);
				let ok_ty = self.expect_ty_expr()?;
				let ok_ty = self.data.push(&ok_ty);

				ExprKind::Type(self.data.push(&Type::ErrorUnion { err_ty, ok_ty }))
			},
		};

		let span = (start_span, self.prev().span).into();
		Ok(Some(Expr {
			id: self.next_id(),
			kind,
			span,
		}))
	}

	fn parse_prefixed_ty_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		#[derive(Copy, Clone)]
		enum PrefixTyLayer {
			Nullable {
				start_span: Span,
			},
			Ptr {
				start_span: Span,
				modifiers: PtrModifiers,
			},
			ManyPtr {
				start_span: Span,
				sentinel: Option<&'static Expr>,
				modifiers: PtrModifiers,
			},
			Slice {
				start_span: Span,
				sentinel: Option<&'static Expr>,
				modifiers: PtrModifiers,
			},
			Array {
				start_span: Span,
				size: Option<&'static Expr>,
				sentinel: Option<&'static Expr>,
				is_const: bool,
			},
		}

		let mut layers = Vec::with_capacity_in(16, self.linear_alloc.clone());
		loop {
			let start_span = self.peek().span;
			match self.peek().kind {
				TokenKind::QuestionMark => {
					self.offset += 1;
					layers.push(PrefixTyLayer::Nullable { start_span });
				},
				TokenKind::Star => {
					self.offset += 1;
					let modifiers = self.parse_ptr_modifiers()?;
					layers.push(PrefixTyLayer::Ptr { start_span, modifiers });
				},
				TokenKind::StarStar => {
					self.offset += 1;
					let modifiers = self.parse_ptr_modifiers()?;
					layers.push(PrefixTyLayer::Ptr { start_span, modifiers });
					layers.push(PrefixTyLayer::Ptr { start_span, modifiers });
				},
				TokenKind::LBracket => {
					self.offset += 1;
					match self.peek().kind {
						TokenKind::Star => {
							self.offset += 1;
							let sentinel = if self.eat_if(TokenTag::Colon).is_some() {
								let expr = self.expect_expr()?;
								Some(self.data.push(&expr))
							} else {
								None
							};
							self.eat_expect(TokenTag::RBracket)?;
							let modifiers = self.parse_ptr_modifiers()?;
							layers.push(PrefixTyLayer::ManyPtr {
								start_span,
								sentinel,
								modifiers,
							});
						},
						TokenKind::RBracket => {
							self.offset += 1;
							let modifiers = self.parse_ptr_modifiers()?;
							layers.push(PrefixTyLayer::Slice {
								start_span,
								sentinel: None,
								modifiers,
							});
						},
						TokenKind::Colon => {
							self.offset += 1;
							let sentinel = self.expect_expr()?;
							let sentinel = Some(self.data.push(&sentinel));
							self.eat_expect(TokenTag::RBracket)?;
							let modifiers = self.parse_ptr_modifiers()?;
							layers.push(PrefixTyLayer::Slice {
								start_span,
								sentinel,
								modifiers,
							});
						},
						_ => {
							let size = match self.peek().kind {
								TokenKind::Ident {
									symbol,
									kind: IdentKind::User,
								} if symbol.as_ref() == "_" => {
									self.offset += 1;
									None
								},
								_ => {
									let size = self.expect_expr()?;
									Some(self.data.push(&size))
								},
							};
							let sentinel = if self.eat_if(TokenTag::Colon).is_some() {
								let expr = self.expect_expr()?;
								Some(self.data.push(&expr))
							} else {
								None
							};
							self.eat_expect(TokenTag::RBracket)?;
							let is_const = self.eat_if(TokenTag::KwConst).is_some();
							layers.push(PrefixTyLayer::Array {
								start_span,
								size,
								sentinel,
								is_const,
							});
						},
					}
				},
				_ => break,
			}
		}

		let mut expr = self.expect_ty_expr()?;
		while let Some(layer) = layers.pop() {
			let ty = self.data.push(&expr);
			let (kind, start_span) = match layer {
				PrefixTyLayer::Nullable { start_span } => (ExprKind::Type(self.data.push(&Type::Nullable(ty))), start_span),
				PrefixTyLayer::Ptr { start_span, modifiers } => (ExprKind::Type(self.data.push(&Type::Ptr { ty, modifiers })), start_span),
				PrefixTyLayer::ManyPtr {
					start_span,
					sentinel,
					modifiers,
				} => (
					ExprKind::Type(self.data.push(&Type::ManyPtr { ty, sentinel, modifiers })),
					start_span,
				),
				PrefixTyLayer::Slice {
					start_span,
					sentinel,
					modifiers,
				} => (ExprKind::Type(self.data.push(&Type::Slice { ty, sentinel, modifiers })), start_span),
				PrefixTyLayer::Array {
					start_span,
					size,
					sentinel,
					is_const,
				} => (
					ExprKind::Type(self.data.push(&Type::Array {
						ty,
						is_const,
						size,
						sentinel,
					})),
					start_span,
				),
			};
			expr = Expr {
				id: self.next_id(),
				kind,
				span: (start_span, expr.span).into(),
			};
		}

		Ok(Some(expr))
	}

	fn parse_suffix_ty_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		let start_span = self.peek().span;
		let Some(mut expr) = self.parse_primary_ty_expr()? else {
			return Ok(None);
		};

		let mut just_called = false;
		loop {
			if let Some(suffix_op) = self.parse_suffix_ty_op(expr)? {
				expr = suffix_op;
				just_called = false;
				continue;
			}

			if just_called || self.peek().kind != TokenKind::LParen {
				return Ok(Some(expr));
			}

			let fn_call = self.parse_fn_call(expr)?;
			let span = (start_span, self.prev().span).into();
			expr = Expr {
				id: self.next_id(),
				kind: ExprKind::FnCall(self.data.push(&fn_call)),
				span,
			};
			just_called = true;
		}
	}

	fn parse_suffix_ty_op(
		&mut self,
		lhs: Expr,
	) -> Result<Option<Expr>, Diagnostic> {
		match self.peek().kind {
			TokenKind::LBracket => {
				self.offset += 1; // consume '['

				match self.peek().kind {
					TokenKind::DotDot => {
						self.offset += 1; // consume '..'

						match self.peek().kind {
							TokenKind::RBracket => {
								self.offset += 1; // consume ']'

								let index = Index {
									collection: self.data.push(&lhs),
									kind: IndexKind::Range { start: None, end: None },
								};
								let id = self.next_id();
								let index = self.data.push(&index);

								Ok(Some(Expr {
									id,
									kind: ExprKind::Index(index),
									span: (lhs.span, self.prev().span).into(),
								}))
							},
							_ => {
								let end = self.expect_expr()?;
								self.eat_expect(TokenTag::RBracket)?;

								let index = Index {
									collection: self.data.push(&lhs),
									kind: IndexKind::Range {
										start: None,
										end: Some(self.data.push(&end)),
									},
								};
								let id = self.next_id();
								let index = self.data.push(&index);

								Ok(Some(Expr {
									id,
									kind: ExprKind::Index(index),
									span: (lhs.span, self.prev().span).into(),
								}))
							},
						}
					},
					TokenKind::DotDotEq => {
						self.offset += 1; // consume '..='

						let end = self.expect_expr()?;
						self.eat_expect(TokenTag::RBracket)?;

						let index = Index {
							collection: self.data.push(&lhs),
							kind: IndexKind::RangeInclusive {
								start: None,
								end: self.data.push(&end),
							},
						};
						let id = self.next_id();
						let index = self.data.push(&index);
						Ok(Some(Expr {
							id,
							kind: ExprKind::Index(index),
							span: (lhs.span, self.prev().span).into(),
						}))
					},
					_ => {
						let start_or_index = self.expect_expr()?;
						let start_or_index = self.data.push(&start_or_index);

						match self.peek().kind {
							TokenKind::DotDot => {
								self.offset += 1; // consume '..'

								match self.peek().kind {
									TokenKind::RBracket => {
										self.offset += 1; // consume ']'

										let index = Index {
											collection: self.data.push(&lhs),
											kind: IndexKind::Range {
												start: Some(start_or_index),
												end: None,
											},
										};
										let id = self.next_id();
										let index = self.data.push(&index);

										Ok(Some(Expr {
											id,
											kind: ExprKind::Index(index),
											span: (lhs.span, self.prev().span).into(),
										}))
									},
									_ => {
										let end = self.expect_expr()?;
										self.eat_expect(TokenTag::RBracket)?;

										let index = Index {
											collection: self.data.push(&lhs),
											kind: IndexKind::Range {
												start: Some(start_or_index),
												end: Some(self.data.push(&end)),
											},
										};
										let id = self.next_id();
										let index = self.data.push(&index);

										Ok(Some(Expr {
											id,
											kind: ExprKind::Index(index),
											span: (lhs.span, self.prev().span).into(),
										}))
									},
								}
							},
							TokenKind::DotDotEq => {
								self.offset += 1; // consume '..='

								let end = self.expect_expr()?;
								self.eat_expect(TokenTag::RBracket)?;

								let index = Index {
									collection: self.data.push(&lhs),
									kind: IndexKind::RangeInclusive {
										start: Some(start_or_index),
										end: self.data.push(&end),
									},
								};
								let id = self.next_id();
								let index = self.data.push(&index);

								Ok(Some(Expr {
									id,
									kind: ExprKind::Index(index),
									span: (lhs.span, self.prev().span).into(),
								}))
							},
							TokenKind::RBracket => {
								self.offset += 1; // consume ']'

								let index = Index {
									collection: self.data.push(&lhs),
									kind: IndexKind::Index(start_or_index),
								};
								let id = self.next_id();
								let index = self.data.push(&index);

								Ok(Some(Expr {
									id,
									kind: ExprKind::Index(index),
									span: (lhs.span, self.prev().span).into(),
								}))
							},
							_ => {
								Err(self
									.diag_expected_one_of_token(&[TokenTag::DotDot, TokenTag::DotDotEq, TokenTag::RBracket], self.peek()))
							},
						}
					},
				}
			},
			TokenKind::DotStar => {
				self.offset += 1; // consume '.*'

				let deref = ExprKind::Deref(self.data.push(&lhs));

				Ok(Some(Expr {
					id: self.next_id(),
					kind: deref,
					span: (lhs.span, self.prev().span).into(),
				}))
			},
			TokenKind::DotQuestionMark => {
				self.offset += 1; // consume '.?'

				let unwrap = ExprKind::UnwrapNullable(self.data.push(&lhs));

				Ok(Some(Expr {
					id: self.next_id(),
					kind: unwrap,
					span: (lhs.span, self.prev().span).into(),
				}))
			},
			TokenKind::Dot => {
				self.offset += 1; // consume '.'

				match self.peek().kind {
					TokenKind::Ident {
						kind: IdentKind::User | IdentKind::UserEscaped,
						..
					} => {
						let ident = self.parse_user_ident()?;

						let field_access = FieldAccess {
							lhs: self.data.push(&lhs),
							field: self.data.push(&ident),
							span: (lhs.span, self.prev().span).into(),
						};

						Ok(Some(Expr {
							id: self.next_id(),
							kind: ExprKind::FieldAccess(self.data.push(&field_access)),
							span: (lhs.span, self.prev().span).into(),
						}))
					},
					// handle missplaced `?` for nullable unwrap: .?
					TokenKind::QuestionMark => {
						self.offset += 1; // consume '?'

						self.push_error(
							self.diag_unexpected_token(self.peek())
								.with_label(Label::secondary().with_span(self.diag_span((self.prev().span, self.peek().span).into())))
								.with_note("did you mean to use the nullable unwrap operator `.?`?"),
						);

						Ok(None)
					},
					// handle missplaced `*` for dereference: .*
					TokenKind::Star => {
						self.offset += 1; // consume '*'

						self.push_error(
							self.diag_unexpected_token(self.peek())
								.with_label(Label::secondary().with_span(self.diag_span((self.prev().span, self.peek().span).into())))
								.with_note("did you mean to use the dereference operator `.*`?"),
						);

						Ok(None)
					},
					// handle missplaced struct init syntax: .{ ... }
					TokenKind::LBrace => {
						self.push_error(
							self.diag_unexpected_token(self.peek())
								.with_label(Label::secondary().with_span(self.diag_span((self.prev().span, self.peek().span).into())))
								.with_note("did you mean to use a struct initialization expression?"),
						);

						Ok(None)
					},
					_ => Err(self.diag_unexpected_token(self.peek())),
				}
			},
			_ => Ok(None),
		}
	}

	fn parse_primary_ty_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		let start_span = self.peek().span;

		let kind = match self.peek().kind {
			// Primitive types
			TokenKind::TyUsize => {
				self.offset += 1;
				ExprKind::Type(self.common_types.usize_ty)
			},
			TokenKind::TyIsize => {
				self.offset += 1;
				ExprKind::Type(self.common_types.isize_ty)
			},
			TokenKind::TyU(bits) => {
				self.offset += 1;

				ExprKind::Type(match bits {
					8 => self.common_types.u8_ty,
					16 => self.common_types.u16_ty,
					32 => self.common_types.u32_ty,
					64 => self.common_types.u64_ty,
					128 => self.common_types.u128_ty,
					_ => self.data.push(&Type::Int(IntSuffix::U(bits))),
				})
			},
			TokenKind::TyI(bits) => {
				self.offset += 1;

				ExprKind::Type(match bits {
					8 => self.common_types.i8_ty,
					16 => self.common_types.i16_ty,
					32 => self.common_types.i32_ty,
					64 => self.common_types.i64_ty,
					128 => self.common_types.i128_ty,
					_ => self.data.push(&Type::Int(IntSuffix::I(bits))),
				})
			},
			TokenKind::TyF16 => {
				self.offset += 1;
				ExprKind::Type(self.common_types.f16_ty)
			},
			TokenKind::TyF32 => {
				self.offset += 1;
				ExprKind::Type(self.common_types.f32_ty)
			},
			TokenKind::TyF64 => {
				self.offset += 1;
				ExprKind::Type(self.common_types.f64_ty)
			},
			TokenKind::TyF128 => {
				self.offset += 1;
				ExprKind::Type(self.common_types.f128_ty)
			},
			TokenKind::TyBool => {
				self.offset += 1;
				ExprKind::Type(self.common_types.bool_ty)
			},
			TokenKind::TyVoid => {
				self.offset += 1;
				ExprKind::Type(self.common_types.void_ty)
			},
			TokenKind::TyNever => {
				self.offset += 1;
				ExprKind::Type(self.common_types.never_ty)
			},
			TokenKind::TyAny => {
				self.offset += 1;
				ExprKind::Type(self.common_types.any_ty)
			},
			TokenKind::TyAnyint => {
				self.offset += 1;
				ExprKind::Type(self.common_types.anyint_ty)
			},
			TokenKind::TyAnyfloat => {
				self.offset += 1;
				ExprKind::Type(self.common_types.anyfloat_ty)
			},
			TokenKind::TyAnyerror => {
				self.offset += 1;
				ExprKind::Type(self.common_types.anyerror_ty)
			},
			TokenKind::TyType => {
				self.offset += 1;
				ExprKind::Type(self.common_types.type_ty)
			},

			// Literals
			TokenKind::LitChar(chr) => {
				self.offset += 1;
				ExprKind::Lit(self.data.push(&Lit::Char(chr)))
			},
			TokenKind::LitStr(symbol) => {
				self.offset += 1;
				ExprKind::Lit(self.data.push(&Lit::Str(symbol)))
			},
			TokenKind::LitMultiLineStr(bytes) => {
				self.offset += 1;
				let mut buffer = Vec::new_in(self.linear_alloc.clone());
				buffer.extend_from_slice(&bytes);

				while let TokenKind::LitMultiLineStr(symbol) = self.peek().kind {
					self.offset += 1;
					buffer.push(b'\n');
					buffer.extend_from_slice(&symbol);
				}

				ExprKind::Lit(self.data.push(&Lit::Str(Intern::from(&buffer[..]))))
			},
			TokenKind::LitInt { symbol, radix } => {
				self.offset += 1;
				let suffix = self.maybe_parse_int_suffix();
				ExprKind::Lit(self.data.push(&Lit::Integer { symbol, radix, suffix }))
			},
			TokenKind::LitFloat { symbol } => {
				self.offset += 1;
				let suffix = self.maybe_parse_float_suffix();
				ExprKind::Lit(self.data.push(&Lit::Float { symbol, suffix }))
			},
			TokenKind::LitBool(value) => {
				self.offset += 1;
				ExprKind::Lit(self.data.push(&Lit::Bool(value)))
			},
			TokenKind::KwUndefined => {
				self.offset += 1;
				ExprKind::Undefined
			},
			TokenKind::Dot => {
				match self.peek_nth(1).kind {
					TokenKind::Ident {
						kind: IdentKind::User | IdentKind::UserEscaped,
						..
					} => {
						self.offset += 1; // consume '.'
						let TokenKind::Ident { symbol, .. } = self.bump().kind else {
							// SAFETY: we just checked that the next token is an identifier
							unsafe { unreachable_unchecked() }
						};

						ExprKind::Lit(self.data.push(&Lit::EnumVariant(symbol)))
					},
					TokenKind::LBrace => {
						self.offset += 1; // consume '.'
						let next = self.peek_nth(1).kind;
						if next == TokenKind::Dot || next == TokenKind::RBrace {
							let init = self.parse_struct_init_expr(None)?;
							ExprKind::StructInit(self.data.push(&init))
						} else {
							let init = self.parse_array_init_expr(None)?;
							ExprKind::ArrayInit(self.data.push(&init))
						}
					},
					_ => return self.parse_init_expr(),
				}
			},
			TokenKind::Ident {
				kind: IdentKind::Builtin, ..
			} => {
				let ident = self.parse_ident()?;
				let ident = Expr {
					id: self.next_id(),
					kind: ExprKind::Ident(self.data.push(&ident)),
					span: (start_span, self.prev().span).into(),
				};
				let fn_call = self.parse_fn_call(ident)?;
				ExprKind::FnCall(self.data.push(&fn_call))
			},
			TokenKind::DirLinear => {
				self.offset += 1; // consume 'linear'
				match self.peek().kind {
					TokenKind::DirPacked => {
						self.offset += 1; // consume 'packed'
						self.eat_expect(TokenTag::KwStruct)?;
						let struct_decl = self.parse_struct_decl(false, true, true, start_span)?;
						let struct_decl = self.data.push(&struct_decl);
						ExprKind::Type(self.data.push(&Type::Struct(struct_decl)))
					},
					TokenKind::KwStruct => {
						self.offset += 1; // consume 'struct'
						let struct_decl = self.parse_struct_decl(false, false, true, start_span)?;
						let struct_decl = self.data.push(&struct_decl);
						ExprKind::Type(self.data.push(&Type::Struct(struct_decl)))
					},
					TokenKind::KwEnum => {
						self.offset += 1; // consume 'enum'
						let enum_decl = self.parse_enum_decl(false, false, true, start_span)?;
						let enum_decl = self.data.push(&enum_decl);
						ExprKind::Type(self.data.push(&Type::Enum(enum_decl)))
					},
					TokenKind::KwUnion => {
						self.offset += 1; // consume 'union'
						let union_decl = self.parse_union_decl(false, true, start_span)?;
						let union_decl = self.data.push(&union_decl);
						ExprKind::Type(self.data.push(&Type::Union(union_decl)))
					},
					_ => {
						let tok = self.bump();
						return Err(self.diag_expected_token(TokenTag::KwStruct, &tok));
					},
				}
			},
			TokenKind::DirPacked => {
				self.offset += 1; // consume 'packed'
				self.eat_expect(TokenTag::KwStruct)?;

				let struct_decl = self.parse_struct_decl(false, true, false, start_span)?;
				let struct_decl = self.data.push(&struct_decl);
				ExprKind::Type(self.data.push(&Type::Struct(struct_decl)))
			},
			TokenKind::KwStruct => {
				self.offset += 1; // consume 'struct'

				let struct_decl = self.parse_struct_decl(false, false, false, start_span)?;
				let struct_decl = self.data.push(&struct_decl);
				ExprKind::Type(self.data.push(&Type::Struct(struct_decl)))
			},
			TokenKind::KwEnum => {
				self.offset += 1; // consume 'enum'

				let enum_decl = self.parse_enum_decl(false, false, false, start_span)?;
				let enum_decl = self.data.push(&enum_decl);
				ExprKind::Type(self.data.push(&Type::Enum(enum_decl)))
			},
			TokenKind::KwUnion => {
				self.offset += 1; // consume 'union'

				let union_decl = self.parse_union_decl(false, false, start_span)?;
				let union_decl = self.data.push(&union_decl);
				ExprKind::Type(self.data.push(&Type::Union(union_decl)))
			},
			// TODO: add error parsing
			TokenKind::LParen => become self.parse_group_ty_expr(),

			// Control flow
			TokenKind::Ident { kind: IdentKind::User, .. } if self.peek_nth(1).kind == TokenKind::Colon => {
				let ident = self.parse_user_ident()?;
				self.eat_expect(TokenTag::Colon)?;
				self.parse_labeled_control_flow(Some(ident), Ctx::Ty)?
			},
			TokenKind::KwIf => self.parse_if_impl(Ctx::Ty)?,
			TokenKind::KwWhile => self.parse_while_impl(None, false, Ctx::Ty)?,
			TokenKind::KwFor => self.parse_for_impl(None, false, Ctx::Ty)?,
			TokenKind::KwLoop => self.parse_loop_impl(None, Ctx::Ty)?,
			TokenKind::KwSwitch => self.parse_switch_impl(None, Ctx::Ty)?,
			TokenKind::DirInline => {
				self.offset += 1; // consume 'inline'
				self.parse_inline_loop(None, Ctx::Ty)?
			},
			TokenKind::Ident { .. } => {
				let generic_ident = self.parse_ident()?;
				ExprKind::Ident(self.data.push(&generic_ident))
			},
			_ => return Ok(None),
		};

		let span = (start_span, self.prev().span).into();
		Ok(Some(Expr {
			id: self.next_id(),
			kind,
			span,
		}))
	}

	fn parse_group_ty_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		let mut starts = Vec::with_capacity_in(16, self.linear_alloc.clone());
		while self.peek().kind == TokenKind::LParen {
			starts.push(self.peek().span);
			self.offset += 1;
		}

		let mut expr = self.expect_ty_expr()?;
		while let Some(start_span) = starts.pop() {
			self.eat_expect(TokenTag::RParen)?;
			let span = (start_span, self.prev().span).into();
			let inner = self.data.push(&expr);
			expr = Expr {
				id: self.next_id(),
				kind: ExprKind::Group(inner),
				span,
			};
		}
		Ok(Some(expr))
	}

	fn parse_ptr_modifiers(&mut self) -> Result<PtrModifiers, Diagnostic> {
		let mut saw_const = false;
		let mut saw_volatile = false;
		let mut saw_addrspace = false;

		let mut modifiers = PtrModifiers {
			is_const: false,
			is_volatile: false,
			addrspace: None,
		};

		loop {
			match self.peek().kind {
				TokenKind::KwConst => {
					if saw_const {
						self.errors.push(self.diag_redundant_qualifier("const", self.peek().span))
					}

					self.offset += 1;
					saw_const = true;
					modifiers.is_const = true;
				},
				TokenKind::DirVolatile => {
					if saw_volatile {
						self.errors.push(self.diag_redundant_qualifier("volatile", self.peek().span))
					}

					self.offset += 1;
					saw_volatile = true;
					modifiers.is_volatile = true;
				},
				TokenKind::DirAddrspace => {
					if saw_addrspace {
						self.errors.push(self.diag_redundant_qualifier("addrspace", self.peek().span))
					}

					self.offset += 1;
					saw_addrspace = true;
					self.eat_expect(TokenTag::LParen)?;
					let expr = self.expect_expr()?;
					self.eat_expect(TokenTag::RParen)?;
					modifiers.addrspace = Some(self.data.push(&expr));
				},
				_ => break,
			}
		}

		Ok(modifiers)
	}

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn expect_expr(&mut self) -> Result<Expr, Diagnostic> {
		let expr = match self.parse_expr()? {
			Some(expr) => expr,
			None => return Err(self.diag_expected_expression(self.peek())),
		};
		Ok(expr)
	}

	/// Like `expect_expr`, but does not allow `Type { ... }` init expressions.
	/// Used where `{` would be ambiguous with a block body (if/while/switch/for conditions).
	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn expect_expr_no_init(&mut self) -> Result<Expr, Diagnostic> {
		let expr = match self.parse_expr_precedence(0, false)? {
			Some(expr) => expr,
			None => return Err(self.diag_expected_expression(self.peek())),
		};
		Ok(expr)
	}

	#[inline(always)]
	fn parse_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		self.parse_expr_precedence(0, true)
	}

	fn parse_expr_precedence(
		&mut self,
		min_bp: u8,
		allow_init: bool,
	) -> Result<Option<Expr>, Diagnostic> {
		let Some(mut lhs) = self.parse_prefix_expr(allow_init)? else {
			return Ok(None);
		};

		loop {
			let start = self.offset;
			let current = self.peek().kind;

			let op = self.infix_op(current);

			let op = match op {
				Some(op) => op,
				None => {
					self.offset = start;
					break;
				},
			};

			if op.lbp < min_bp || self.peek().kind == TokenKind::Eq {
				self.offset = start;
				break;
			}

			let rhs = match self.parse_expr_precedence(op.rbp, allow_init)? {
				Some(expr) => expr,
				None => return Err(self.diag_expected_expression(self.peek())),
			};
			let span = (lhs.span, rhs.span).into();
			let lhs_ref = self.data.push(&lhs);
			let rhs_ref = self.data.push(&rhs);
			let kind = self.bin_op(op.kind, lhs_ref, rhs_ref);

			lhs = Expr {
				id: self.next_id(),
				kind,
				span,
			};
		}

		Ok(Some(lhs))
	}

	#[inline(always)]
	fn infix_op(
		&mut self,
		kind: TokenKind,
	) -> Option<InfixOp> {
		use InfixOpKind::*;

		let (lbp, op_kind) = match kind {
			TokenKind::Plus => (50, Add),
			TokenKind::PlusPipe => (50, AddSat),
			TokenKind::PlusPercent => (50, AddWrap),
			TokenKind::Minus => (50, Sub),
			TokenKind::MinusPipe => (50, SubSat),
			TokenKind::MinusPercent => (50, SubWrap),
			TokenKind::Star => (60, Mul),
			TokenKind::StarPipe => (60, MulSat),
			TokenKind::StarPercent => (60, MulWrap),
			TokenKind::StarStar => (70, Pow),
			TokenKind::StarStarPipe => (70, PowSat),
			TokenKind::StarStarPercent => (70, PowWrap),
			TokenKind::Slash => (60, Div),
			TokenKind::Percent => (60, Rem),
			TokenKind::Amp => (30, BitAnd),
			TokenKind::Pipe => (20, BitOr),
			TokenKind::Caret => (25, BitXor),
			TokenKind::Lt => (35, Lt),
			TokenKind::LtEq => (35, Lte),
			TokenKind::Gt => (35, Gt),
			TokenKind::GtEq => (35, Gte),
			TokenKind::KwAnd => (15, BoolAnd),
			TokenKind::KwOr => (10, BoolOr),
			TokenKind::EqEq => (30, Eq),
			TokenKind::BangEq => (30, Neq),
			TokenKind::LtLt => (40, Shl),
			TokenKind::LtLtPipe => (40, ShlSat),
			TokenKind::LtLtPercent => (40, ShlWrap),
			TokenKind::GtGt => (40, Shr),
			TokenKind::GtGtPipe => (40, ShrSat),
			TokenKind::GtGtPercent => (40, ShrWrap),
			_ => return None,
		};

		self.offset += 1;
		Some(InfixOp::left(lbp, op_kind))
	}

	#[inline(always)]
	fn bin_op(
		&mut self,
		op: InfixOpKind,
		lhs: &'static Expr,
		rhs: &'static Expr,
	) -> ExprKind {
		use InfixOpKind::*;

		let bin_op = self.data.push(&BinOp { lhs, rhs });
		match op {
			Add => ExprKind::Add(bin_op),
			AddSat => ExprKind::AddSat(bin_op),
			AddWrap => ExprKind::AddWrap(bin_op),
			Sub => ExprKind::Sub(bin_op),
			SubSat => ExprKind::SubSat(bin_op),
			SubWrap => ExprKind::SubWrap(bin_op),
			Mul => ExprKind::Mul(bin_op),
			MulSat => ExprKind::MulSat(bin_op),
			MulWrap => ExprKind::MulWrap(bin_op),
			Pow => ExprKind::Pow(bin_op),
			PowSat => ExprKind::PowSat(bin_op),
			PowWrap => ExprKind::PowWrap(bin_op),
			Div => ExprKind::Div(bin_op),
			Rem => ExprKind::Rem(bin_op),
			Shl => ExprKind::Shl(bin_op),
			ShlSat => ExprKind::ShlSat(bin_op),
			ShlWrap => ExprKind::ShlWrap(bin_op),
			Shr => ExprKind::Shr(bin_op),
			ShrSat => ExprKind::ShrSat(bin_op),
			ShrWrap => ExprKind::ShrWrap(bin_op),
			BitAnd => ExprKind::BitAnd(bin_op),
			BitOr => ExprKind::BitOr(bin_op),
			BitXor => ExprKind::BitXor(bin_op),
			Lt => ExprKind::Lt(bin_op),
			Lte => ExprKind::Lte(bin_op),
			Gt => ExprKind::Gt(bin_op),
			Gte => ExprKind::Gte(bin_op),
			BoolAnd => ExprKind::BoolAnd(bin_op),
			BoolOr => ExprKind::BoolOr(bin_op),
			Eq => ExprKind::Eq(bin_op),
			Neq => ExprKind::Neq(bin_op),
		}
	}

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn expect_prefix_expr(
		&mut self,
		allow_init: bool,
	) -> Result<Expr, Diagnostic> {
		let expr = match self.parse_prefix_expr(allow_init)? {
			Some(expr) => expr,
			None => return Err(self.diag_expected_expression(self.peek())),
		};
		Ok(expr)
	}

	fn parse_prefix_expr(
		&mut self,
		allow_init: bool,
	) -> Result<Option<Expr>, Diagnostic> {
		#[derive(Copy, Clone)]
		enum PrefixExprLayer {
			Neg(Span),
			Pos(Span),
			Not(Span),
			BitNot(Span),
			AddressOf(Span),
			Try(Span),
		}

		let mut layers = Vec::with_capacity_in(16, self.linear_alloc.clone());
		loop {
			let start_span = self.peek().span;
			match self.peek().kind {
				TokenKind::Minus => {
					self.offset += 1;
					layers.push(PrefixExprLayer::Neg(start_span));
				},
				TokenKind::Plus => {
					self.offset += 1;
					layers.push(PrefixExprLayer::Pos(start_span));
				},
				TokenKind::Bang => {
					self.offset += 1;
					layers.push(PrefixExprLayer::Not(start_span));
				},
				TokenKind::Tilde => {
					self.offset += 1;
					layers.push(PrefixExprLayer::BitNot(start_span));
				},
				TokenKind::Amp => {
					self.offset += 1;
					layers.push(PrefixExprLayer::AddressOf(start_span));
				},
				TokenKind::KwTry => {
					self.offset += 1;
					layers.push(PrefixExprLayer::Try(start_span));
				},
				_ => break,
			}
		}

		if layers.is_empty() {
			become self.parse_primary_expr(allow_init)
		}

		let mut expr = match self.parse_primary_expr(allow_init)? {
			Some(expr) => expr,
			None => return Err(self.diag_expected_expression(self.peek())),
		};

		while let Some(layer) = layers.pop() {
			let kind = match layer {
				PrefixExprLayer::Neg(_) => ExprKind::Neg(self.data.push(&expr)),
				PrefixExprLayer::Pos(_) => ExprKind::Pos(self.data.push(&expr)),
				PrefixExprLayer::Not(_) => ExprKind::Not(self.data.push(&expr)),
				PrefixExprLayer::BitNot(_) => ExprKind::BitNot(self.data.push(&expr)),
				PrefixExprLayer::AddressOf(_) => ExprKind::AddressOf(self.data.push(&expr)),
				PrefixExprLayer::Try(_) => ExprKind::Try(self.data.push(&expr)),
			};
			let start_span = match layer {
				PrefixExprLayer::Neg(start_span)
				| PrefixExprLayer::Pos(start_span)
				| PrefixExprLayer::Not(start_span)
				| PrefixExprLayer::BitNot(start_span)
				| PrefixExprLayer::AddressOf(start_span)
				| PrefixExprLayer::Try(start_span) => start_span,
			};
			expr = Expr {
				id: self.next_id(),
				kind,
				span: (start_span, expr.span).into(),
			};
		}

		Ok(Some(expr))
	}

	fn parse_primary_expr(
		&mut self,
		allow_init: bool,
	) -> Result<Option<Expr>, Diagnostic> {
		let token = self.peek();
		let start_span = token.span;

		let kind = match token.kind {
			// Groups
			TokenKind::LParen => become self.parse_group_expr(allow_init),
			TokenKind::LBrace => self
				.parse_block_impl(None, false, start_span, Ctx::Value)
				.map(|block| ExprKind::Block(self.data.push(&block)))?,

			// Control flow
			TokenKind::Ident { kind: IdentKind::User, .. } if self.peek_nth(1).kind == TokenKind::Colon => {
				let ident = self.parse_user_ident()?;
				self.eat_expect(TokenTag::Colon)?;
				self.parse_labeled_control_flow(Some(ident), Ctx::Value)?
			},
			TokenKind::KwIf => self.parse_if_impl(Ctx::Value)?,
			TokenKind::KwWhile => self.parse_while_impl(None, false, Ctx::Value)?,
			TokenKind::KwFor => self.parse_for_impl(None, false, Ctx::Value)?,
			TokenKind::KwLoop => self.parse_loop_impl(None, Ctx::Value)?,
			TokenKind::KwSwitch => self.parse_switch_impl(None, Ctx::Value)?,
			TokenKind::DirInline => {
				self.offset += 1; // consume 'inline'
				self.parse_inline_loop(None, Ctx::Value)?
			},
			_ => return if allow_init { self.parse_init_expr() } else { self.parse_ty_expr() },
		};

		let span = (start_span, self.prev().span).into();
		Ok(Some(Expr {
			id: self.next_id(),
			kind,
			span,
		}))
	}

	fn parse_group_expr(
		&mut self,
		allow_init: bool,
	) -> Result<Option<Expr>, Diagnostic> {
		let mut starts = Vec::with_capacity_in(16, self.linear_alloc.clone());
		while self.peek().kind == TokenKind::LParen {
			starts.push(self.peek().span);
			self.offset += 1;
		}

		let mut expr = if allow_init {
			self.expect_expr()?
		} else {
			self.expect_expr_no_init()?
		};

		while let Some(start_span) = starts.pop() {
			self.eat_expect(TokenTag::RParen)?;
			let span = (start_span, self.prev().span).into();
			let inner = self.data.push(&expr);
			expr = Expr {
				id: self.next_id(),
				kind: ExprKind::Group(inner),
				span,
			};
		}

		Ok(Some(expr))
	}

	fn parse_init_expr(&mut self) -> Result<Option<Expr>, Diagnostic> {
		let Some(lhs) = self.parse_ty_expr()? else {
			return Ok(None);
		};

		if self.peek().kind != TokenKind::LBrace {
			return Ok(Some(lhs));
		}

		let next = self.peek_nth(1).kind;
		if next == TokenKind::Dot || next == TokenKind::RBrace {
			let init = self.parse_struct_init_expr(Some(lhs))?;
			return Ok(Some(Expr {
				id: self.next_id(),
				kind: ExprKind::StructInit(self.data.push(&init)),
				span: (lhs.span, self.prev().span).into(),
			}));
		}

		// Array init: `Type { value, value }`
		let init = self.parse_array_init_expr(Some(lhs))?;
		Ok(Some(Expr {
			id: self.next_id(),
			kind: ExprKind::ArrayInit(self.data.push(&init)),
			span: (lhs.span, self.prev().span).into(),
		}))
	}

	// =========================================================================
	//                     Unified Control Flow Implementations
	// =========================================================================

	/// Parse labeled control flow constructs (e.g., `label: for`, `label: while`, etc.).
	#[inline(always)]
	fn parse_labeled_control_flow(
		&mut self,
		label: Option<Ident>,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		match self.peek().kind {
			TokenKind::DirInline => {
				self.offset += 1; // consume 'inline'
				self.parse_inline_loop(label, ctx)
			},
			TokenKind::KwFor => self.parse_for_impl(label, false, ctx),
			TokenKind::KwWhile => self.parse_while_impl(label, false, ctx),
			TokenKind::KwLoop => self.parse_loop_impl(label, ctx),
			TokenKind::KwSwitch => self.parse_switch_impl(label, ctx),
			TokenKind::LBrace => {
				let block = self.parse_block_impl(label, false, self.peek().span, ctx)?;
				Ok(ExprKind::Block(self.data.push(&block)))
			},
			_ => Err(self.diag_expected_one_of_token(
				&[
					TokenTag::KwFor,
					TokenTag::KwWhile,
					TokenTag::KwLoop,
					TokenTag::KwSwitch,
					TokenTag::LBrace,
				],
				self.peek(),
			)),
		}
	}

	/// Parse inline loop constructs (`inline for` or `inline while`).
	#[inline(always)]
	fn parse_inline_loop(
		&mut self,
		label: Option<Ident>,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		match self.peek().kind {
			TokenKind::KwFor => self.parse_for_impl(label, true, ctx),
			TokenKind::KwWhile => self.parse_while_impl(label, true, ctx),
			_ => Err(self.diag_expected_inlinable(self.prev())),
		}
	}

	#[cfg_attr(debug_assertions, track_caller)]
	fn parse_if_impl(
		&mut self,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		let start_span = self.peek().span;

		self.eat_expect(TokenTag::KwIf)?;
		let cond = self.expect_expr_ctx_no_init(ctx)?;
		let cond = self.data.push(&cond);

		let then_block = self.parse_block_impl(None, false, self.peek().span, ctx)?;
		let then_block = self.data.push(&then_block);

		let mut branches = Vec::new_in(self.linear_alloc.clone());

		while self.eat_if(TokenTag::KwElse).is_some() {
			if self.eat_if(TokenTag::KwIf).is_some() {
				let elif_cond = self.expect_expr_ctx_no_init(ctx)?;
				let elif_cond = self.data.push(&elif_cond);

				let elif_block = self.parse_block_impl(None, false, self.peek().span, ctx)?;
				let elif_block = self.data.push(&elif_block);

				branches.push((
					self.next_id(),
					Some(elif_cond),
					elif_block,
					(elif_cond.span, elif_block.span).into(),
				));
			} else {
				let else_block = self.parse_block_impl(None, false, self.peek().span, ctx)?;
				let else_block = self.data.push(&else_block);

				branches.push((self.next_id(), None, else_block, else_block.span));
				break;
			}
		}

		// Build the else_block chain from the inside out (reverse iteration)
		let mut else_block: Option<&'static ElseBlock> = None;

		for (id, branch_cond, branch_block, span) in branches.into_iter().rev() {
			match branch_cond {
				Some(elif_cond) => {
					// This is an else-if: wrap in an If struct, then in ElseBlock::If
					let inner_if = If {
						id,
						cond: elif_cond,
						then_block: branch_block,
						else_block,
						span,
					};
					else_block = Some(self.data.push(&ElseBlock::If(inner_if)));
				},
				None => {
					else_block = Some(self.data.push(&ElseBlock::Block(*branch_block)));
				},
			}
		}

		let id = self.next_id();
		let if_expr = self.data.push(&If {
			id,
			cond,
			then_block,
			else_block,
			span: (start_span, self.prev().span).into(),
		});

		Ok(ExprKind::If(if_expr))
	}

	/// Unified implementation for `while` expressions.
	fn parse_while_impl(
		&mut self,
		label: Option<Ident>,
		inline: bool,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		let start_span = self.peek().span;

		self.eat_expect(TokenTag::KwWhile)?;
		let cond = self.expect_expr_ctx_no_init(ctx)?;
		let cond = self.data.push(&cond);

		let body = self.parse_block_impl(label, false, self.peek().span, ctx)?;
		let body = self.data.push(&body);

		let id = self.next_id();
		let while_expr = self.data.push(&While {
			id,
			inline,
			cond,
			body,
			span: (start_span, self.prev().span).into(),
		});

		Ok(ExprKind::While(while_expr))
	}

	/// Unified implementation for `for` expressions.
	fn parse_for_impl(
		&mut self,
		label: Option<Ident>,
		inline: bool,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		let start_span = self.peek().span;

		self.eat_expect(TokenTag::KwFor)?;
		let iter_var = self.expect_expr_ctx(ctx)?;
		let iter_var = self.data.push(&iter_var);

		self.eat_expect(TokenTag::KwIn)?;
		let iterable = self.expect_expr_ctx_no_init(ctx)?;
		let iterable = self.data.push(&iterable);

		let body = self.parse_block_impl(label, false, self.peek().span, ctx)?;
		let body = self.data.push(&body);

		let id = self.next_id();
		let for_expr = self.data.push(&For {
			id,
			inline,
			iter_var,
			iterable,
			body,
			span: (start_span, self.prev().span).into(),
		});

		Ok(ExprKind::For(for_expr))
	}

	/// Unified implementation for `loop` expressions.
	fn parse_loop_impl(
		&mut self,
		label: Option<Ident>,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		self.eat_expect(TokenTag::KwLoop)?;

		let body = self.parse_block_impl(label, false, self.peek().span, ctx)?;
		let body = self.data.push(&body);

		Ok(ExprKind::Loop(body))
	}

	/// Unified implementation for `switch` expressions.
	fn parse_switch_impl(
		&mut self,
		label: Option<Ident>,
		ctx: Ctx,
	) -> Result<ExprKind, Diagnostic> {
		let start_span = self.peek().span;

		self.eat_expect(TokenTag::KwSwitch)?;
		let expr = self.expect_expr_ctx_no_init(ctx)?;
		let expr = self.data.push(&expr);

		let mut cases = Vec::new_in(self.linear_alloc.clone());

		self.eat_expect(TokenTag::LBrace)?;
		while likely(!matches!(self.peek().kind, TokenKind::RBrace | TokenKind::KwElse)) {
			let case_start_span = self.peek().span;

			let (patterns, _) = self.eat_delimited_ctx(TokenTag::Comma, |Token { kind, .. }| !matches!(kind, TokenKind::FatArrow), ctx)?;

			self.eat_expect(TokenTag::FatArrow)?;

			// Parse optional capture: |ident|
			let capture = if self.eat_if(TokenTag::Pipe).is_some() {
				let ident = self.parse_ident()?;
				self.eat_expect(TokenTag::Pipe)?;
				Some(ident)
			} else {
				None
			};

			let stmt = self.parse_statement_impl(ctx)?;
			let case_end_span = self.prev().span;

			let id = self.next_id();

			cases.push(SwitchCase {
				id,
				patterns: self.data.push_slice(&patterns),
				capture,
				stmt: self.data.push(&stmt),
				span: (case_start_span, case_end_span).into(),
			});
		}

		let cases = self.data.push_slice(&cases);

		let (else_capture, else_stmt) = if self.eat_if(TokenTag::KwElse).is_some() {
			if self.eat_if(TokenTag::FatArrow).is_some() {
				let stmt = self.parse_statement_impl(ctx)?;
				(None, Some(self.data.push(&stmt)))
			} else {
				let capture = self.expect_expr_ctx(ctx)?;
				let capture = self.data.push(&capture);

				self.eat_expect(TokenTag::FatArrow)?;

				let stmt = self.parse_statement_impl(ctx)?;

				(Some(capture), Some(self.data.push(&stmt)))
			}
		} else {
			(None, None)
		};

		self.eat_expect(TokenTag::RBrace)?;

		let id = self.next_id();
		let switch_expr = self.data.push(&Switch {
			id,
			label,
			expr,
			cases,
			else_capture,
			else_stmt,
			span: (start_span, self.prev().span).into(),
		});

		Ok(ExprKind::Switch(switch_expr))
	}

	/// Unified implementation for block expressions.
	#[cfg_attr(debug_assertions, track_caller)]
	fn parse_block_impl(
		&mut self,
		label: Option<Ident>,
		is_const: bool,
		start_span: Span,
		ctx: Ctx,
	) -> Result<Block, Diagnostic> {
		let mut saw_implicit_return = false;

		let stmts = self.eat_group(TokenTag::LBrace, TokenTag::RBrace, |parser| {
			if unlikely(saw_implicit_return) {
				parser.push_error(parser.diag_missing_semicolon(parser.prev()));
			}

			let stmt = parser.parse_statement_impl(ctx)?;

			if let StatementKind::ImplicitReturn(expr) = &stmt.kind
				&& unlikely(!expr.is_control_flow())
			{
				saw_implicit_return = true;
			}

			Ok(stmt)
		})?;

		let end_span = self.prev().span;

		Ok(Block {
			id: self.next_id(),
			label,
			is_const,
			stmts: self.data.push_slice(&stmts),
			span: (start_span, end_span).into(),
		})
	}

	/// Unified implementation for statements.
	fn parse_statement_impl(
		&mut self,
		ctx: Ctx,
	) -> Result<Statement, Diagnostic> {
		let id = self.next_id();
		let start_span = self.peek().span;

		let kind = match self.peek().kind {
			TokenKind::KwReturn => {
				self.offset += 1;

				let expr = if self.peek().kind != TokenTag::Semicolon {
					let expr = self.expect_expr_ctx(ctx)?;
					Some(self.data.push(&expr))
				} else {
					None
				};

				self.eat_expect(TokenTag::Semicolon)?;
				StatementKind::Return(expr)
			},
			TokenKind::KwDefer => {
				self.offset += 1;

				let expr = self.expect_expr_ctx(ctx)?;
				self.eat_expect(TokenTag::Semicolon)?;
				StatementKind::Defer(self.data.push(&expr))
			},
			TokenKind::KwErrdefer => {
				self.offset += 1;

				let expr = self.expect_expr_ctx(ctx)?;
				self.eat_expect(TokenTag::Semicolon)?;
				StatementKind::Errdefer(self.data.push(&expr))
			},
			TokenKind::KwContinue => {
				self.offset += 1;

				let label = if self.eat_if(TokenTag::Colon).is_some() {
					self.parse_ident().map(Some)?
				} else {
					None
				};

				let value = if self.eat_if(TokenTag::Semicolon).is_none() && self.peek().kind != TokenKind::Comma {
					let value = self.expect_expr()?;
					let value = self.data.push(&value);
					self.eat_if(TokenTag::Semicolon);
					Some(value)
				} else {
					None
				};

				StatementKind::Continue { label, value }
			},
			TokenKind::KwBreak => {
				self.offset += 1;

				let label = if self.eat_if(TokenTag::Colon).is_some() {
					self.parse_ident().map(Some)?
				} else {
					None
				};

				let value = if self.eat_if(TokenTag::Semicolon).is_none() && self.peek().kind != TokenKind::Comma {
					let value = self.expect_expr()?;
					let value = self.data.push(&value);
					self.eat_if(TokenTag::Semicolon);
					Some(value)
				} else {
					None
				};

				StatementKind::Break { label, value }
			},
			TokenKind::KwVar => {
				self.offset += 1;

				let var_binding = self.parse_var_binding()?;
				StatementKind::Var(self.data.push(&var_binding))
			},
			TokenKind::KwConst => {
				self.offset += 1;

				let var_binding = self.parse_var_binding()?;
				StatementKind::Const(self.data.push(&var_binding))
			},
			TokenKind::KwComptime => {
				self.offset += 1;
				match self.peek().kind {
					TokenKind::KwConst | TokenKind::KwVar => {
						self.offset += 1;
					},
					_ => {
						let token = self.peek();
						let err = self.diag_expected_one_of_token(&[TokenTag::KwConst, TokenTag::KwVar], token);
						return Err(err);
					},
				}
				let var_binding = self.parse_var_binding()?;
				StatementKind::ComptimeVarBinding(self.data.push(&var_binding))
			},
			_ => {
				let expr = self.expect_expr_ctx(ctx)?;
				let expr = self.data.push(&expr);

				if let Some(op) = self.maybe_assign_op() {
					let rhs = self.expect_expr_ctx(ctx)?;
					let rhs = self.data.push(&rhs);

					self.eat_expect(TokenTag::Semicolon)?;

					StatementKind::Assign { lhs: expr, op, rhs }
				} else {
					match self.peek().kind {
						TokenKind::Semicolon => {
							self.offset += 1; // consume ';'
							StatementKind::Expr(expr)
						},
						kind => {
							if expr.is_control_flow() && !matches!(kind, TokenKind::RBrace | TokenKind::Comma) {
								StatementKind::Expr(expr)
							} else {
								StatementKind::ImplicitReturn(expr)
							}
						},
					}
				}
			},
		};

		let span = (start_span, self.prev().span).into();
		Ok(Statement { id, kind, span })
	}

	// =========================================================================
	//                      Context-Aware Expression Helpers
	// =========================================================================

	/// Expect and parse an expression based on context.
	#[inline(always)]
	fn expect_expr_ctx(
		&mut self,
		ctx: Ctx,
	) -> Result<Expr, Diagnostic> {
		match ctx {
			Ctx::Ty => self.expect_ty_expr(),
			Ctx::Value => self.expect_expr(),
		}
	}

	/// Like `expect_expr_ctx`, but does not allow `Type { ... }` init expressions.
	/// Used where `{` would be ambiguous with a block body (if/while/switch/for conditions).
	#[inline(always)]
	fn expect_expr_ctx_no_init(
		&mut self,
		ctx: Ctx,
	) -> Result<Expr, Diagnostic> {
		match ctx {
			Ctx::Ty => self.expect_ty_expr(),
			Ctx::Value => self.expect_expr_no_init(),
		}
	}

	/// Eat a delimited list of expressions based on context.
	fn eat_delimited_ctx(
		&mut self,
		delim_tok: TokenTag,
		predicate: impl FnMut(&Token) -> bool,
		ctx: Ctx,
	) -> Result<(Vec<Expr, RcLinearAllocator>, bool), Diagnostic> {
		match ctx {
			Ctx::Ty => self.eat_delimited(delim_tok, predicate, Self::expect_ty_expr),
			Ctx::Value => self.eat_delimited(delim_tok, predicate, Self::expect_expr),
		}
	}

	fn parse_struct_init_expr(
		&mut self,
		struct_ty: Option<Expr>,
	) -> Result<StructInit, Diagnostic> {
		let fields = self
			.eat_delimited_group(TokenTag::LBrace, TokenTag::RBrace, TokenTag::Comma, Self::parse_struct_field_init)
			.map(|v| self.data.push_slice(&v))?;

		Ok(StructInit {
			ty: struct_ty.map(|expr| self.data.push(&expr)),
			fields,
		})
	}

	fn parse_array_init_expr(
		&mut self,
		array_ty: Option<Expr>,
	) -> Result<ArrayInit, Diagnostic> {
		let elements = self
			.eat_delimited_group(TokenTag::LBrace, TokenTag::RBrace, TokenTag::Comma, Self::expect_expr)
			.map(|v| self.data.push_slice(&v))?;

		Ok(ArrayInit {
			ty: array_ty.map(|expr| self.data.push(&expr)),
			elements,
		})
	}

	fn parse_struct_field_init(&mut self) -> Result<FieldInit, Diagnostic> {
		self.eat_expect(TokenTag::Dot)?;
		let ident = self.parse_ident()?;
		self.eat_expect(TokenTag::Eq)?;
		let value = self.expect_expr()?;
		let value = self.data.push(&value);
		Ok(FieldInit { ident, value })
	}

	fn parse_fn_call(
		&mut self,
		callee: Expr,
	) -> Result<FnCall, Diagnostic> {
		match self.eat_expect(TokenTag::LParen) {
			Ok(_) => {},
			Err(err) => {
				self.push_error(err);
			},
		}

		let mut last_arg = None;
		let mut last_named_arg = None;
		let mut generics = Vec::new_in(self.linear_alloc.clone());
		let mut args = Vec::new_in(self.linear_alloc.clone());

		loop {
			let token = self.peek();

			if token.kind == TokenKind::RParen || unlikely(token.is_eof()) {
				break;
			}

			// Check for named/generic arguments by peeking ahead for `ident:` pattern.
			// We must do this **BEFORE** calling parse_expr(), because parse_expr() would
			// interpret `name:` as a labeled block/loop and fail.
			let is_named_or_generic = matches!(
				(&token.kind, self.peek_nth(1).kind),
				(
					TokenKind::Ident {
						kind: IdentKind::User | IdentKind::UserEscaped,
						..
					},
					TokenKind::Colon
				) | (
					TokenKind::Ident {
						kind: IdentKind::Generic,
						..
					},
					TokenKind::Colon
				)
			);

			if is_named_or_generic {
				// Parse the identifier and consume the colon
				let ident = self.parse_ident()?;
				self.eat_expect(TokenTag::Colon)?;

				if ident.is_generic() {
					// Generic argument
					if let Some(last_arg) = last_arg.or(last_named_arg) {
						self.push_error(
							Diagnostic::error()
								.with_message("generic arguments must come before regular arguments")
								.with_label(Label::secondary().with_span(self.diag_span(last_arg)))
								.with_label(Label::primary().with_span(self.diag_span(ident.span)))
								.with_note("move this generic arg before this argument"),
						);
					}

					let ty = self.expect_ty_expr()?;
					generics.push(GenericArg {
						ident,
						value: self.data.push(&ty),
					});
				} else {
					if last_arg.is_some() {
						self.push_error(
							Diagnostic::error()
								.with_message("named arguments cannot be mixed with positional arguments")
								.with_label(Label::primary().with_span(self.diag_span(ident.span))),
						);
					}

					// Named argument
					let value = self.expect_expr()?;
					let value = self.data.push(&value);

					last_named_arg = Some(ident.span);

					args.push(Arg::Named { name: ident, value });
				}
			} else {
				// Positional argument
				let expr = self.parse_expr()?;
				let expr = if unlikely(expr.is_none()) {
					break;
				} else {
					// SAFETY: we just checked that expr is Some
					unsafe { expr.unwrap_unchecked() }
				};

				if let Some(last_named_arg_span) = last_named_arg {
					self.push_error(
						Diagnostic::error()
							.with_message("positional arguments cannot come after named arguments")
							.with_label(
								Label::secondary()
									.with_span(self.diag_span(last_named_arg_span))
									.with_message("named argument here"),
							)
							.with_label(Label::primary().with_span(self.diag_span(expr.span))),
					);
				}

				last_arg = Some(expr.span);
				args.push(Arg::Positional(self.data.push(&expr)));
			}

			if self.eat_if(TokenTag::Comma).is_none() {
				break;
			}
		}

		self.eat_expect(TokenTag::RParen)?;

		let args = self.data.push_slice(&args);
		let generics = self.data.push_slice(&generics);

		Ok(FnCall {
			callee: self.data.push(&callee),
			generics,
			args,
		})
	}

	// =========================================================================
	//                                Statements
	// =========================================================================

	fn maybe_assign_op(&mut self) -> Option<AssignOp> {
		match self.bump().kind {
			TokenKind::Eq => Some(AssignOp::Assign),
			TokenKind::PlusEq => Some(AssignOp::Add),
			TokenKind::PlusPipeEq => Some(AssignOp::AddSat),
			TokenKind::PlusPercentEq => Some(AssignOp::AddWrap),
			TokenKind::MinusEq => Some(AssignOp::Sub),
			TokenKind::MinusPipeEq => Some(AssignOp::SubSat),
			TokenKind::MinusPercentEq => Some(AssignOp::SubWrap),
			TokenKind::StarEq => Some(AssignOp::Mul),
			TokenKind::StarPipeEq => Some(AssignOp::MulSat),
			TokenKind::StarPercentEq => Some(AssignOp::MulWrap),
			TokenKind::SlashEq => Some(AssignOp::Div),
			TokenKind::LtLtEq => Some(AssignOp::Shl),
			TokenKind::LtLtPipeEq => Some(AssignOp::ShlSat),
			TokenKind::LtLtPercentEq => Some(AssignOp::ShlWrap),
			TokenKind::GtGtEq => Some(AssignOp::Shr),
			TokenKind::GtGtPipeEq => Some(AssignOp::ShrSat),
			TokenKind::GtGtPercentEq => Some(AssignOp::ShrWrap),
			TokenKind::PercentEq => Some(AssignOp::Rem),
			TokenKind::AmpEq => Some(AssignOp::BitAnd),
			TokenKind::PipeEq => Some(AssignOp::BitOr),
			TokenKind::CaretEq => Some(AssignOp::BitXor),
			TokenKind::AmpAmpEq => Some(AssignOp::BoolAnd),
			TokenKind::PipePipeEq => Some(AssignOp::BoolOr),
			_ => {
				self.offset -= 1;
				None
			},
		}
	}

	fn parse_var_binding(&mut self) -> Result<VarBinding, Diagnostic> {
		let id = self.next_id();
		let name = match self.parse_user_ident() {
			Ok(id) => id,
			Err(err) => {
				self.eat_until3(TokenTag::Colon, TokenTag::Eq, TokenTag::Semicolon);
				self.push_error(err);
				Ident {
					symbol: COMMON_INTERNS.empty_str,
					kind: IdentKind::User,
					span: self.prev().span,
				}
			},
		};

		let ty = if self.eat_if(TokenTag::Colon).is_some() {
			Some(match self.expect_ty_expr() {
				Ok(ty) => self.data.push(&ty),
				Err(err) => {
					self.eat_until2(TokenTag::Eq, TokenTag::Semicolon);
					self.push_error(err);
					let id = self.next_id();
					let kind = ExprKind::Type(self.common_types.any_ty);
					let span = self.prev().span;
					self.data.push(&Expr { id, kind, span })
				},
			})
		} else {
			None
		};

		match self.eat_expect(TokenTag::Eq) {
			Ok(_) => {},
			Err(err) => {
				self.eat_until(TokenTag::Semicolon);
				self.offset += 1;
				return Err(err);
			},
		};

		let val = match self.expect_expr() {
			Ok(val) => val,
			Err(err) => {
				self.eat_until(TokenTag::Semicolon);
				self.eat_if(TokenTag::Semicolon);
				return Err(err);
			},
		};
		let val = self.data.push(&val);
		self.eat_expect(TokenTag::Semicolon)?;

		Ok(VarBinding { id, name, ty, val })
	}

	#[inline(always)]
	fn maybe_parse_int_suffix(&mut self) -> Option<IntSuffix> {
		Some(match self.bump().kind {
			TokenKind::TyUsize => IntSuffix::Usize,
			TokenKind::TyIsize => IntSuffix::Isize,
			TokenKind::TyU(bits) => IntSuffix::U(bits),
			TokenKind::TyI(bits) => IntSuffix::I(bits),
			_ => {
				self.offset -= 1;
				return None;
			},
		})
	}

	#[inline(always)]
	fn maybe_parse_float_suffix(&mut self) -> Option<FloatSuffix> {
		Some(match self.bump().kind {
			TokenKind::TyF16 => FloatSuffix::F16,
			TokenKind::TyF32 => FloatSuffix::F32,
			TokenKind::TyF64 => FloatSuffix::F64,
			TokenKind::TyF128 => FloatSuffix::F128,
			_ => {
				self.offset -= 1;
				return None;
			},
		})
	}

	/// Parse an identifier, ensuring it is not a reserved keyword or invalid name.
	#[inline(always)]
	fn parse_ident(&mut self) -> Result<Ident, Diagnostic> {
		let TokenKind::Ident { symbol, kind } = self.eat_expect(TokenTag::Ident)? else {
			// SAFETY: We already know this is an ident token.
			unsafe { unreachable_unchecked() }
		};

		Ok(Ident {
			symbol,
			kind,
			span: self.prev().span,
		})
	}

	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn parse_user_ident(&mut self) -> Result<Ident, Diagnostic> {
		let ident = self.parse_ident()?;

		if unlikely(!ident.is_user()) {
			return Err(self.diag_expected_user_ident(&ident));
		}

		Ok(ident)
	}

	fn parse_decl_modifiers(&mut self) -> (Extern, Inline) {
		let mut ext: Extern = Extern::None;
		let mut inline: Inline = Inline::None;

		loop {
			match self.peek().kind {
				TokenKind::KwExtern => {
					if unlikely(ext != Extern::None) {
						self.push_error(
							Diagnostic::error()
								.with_message("duplicate `extern` modifier")
								.with_label(Label::primary().with_span(self.diag_span(self.peek().span))),
						);
					}

					self.bump();

					ext = if let Ok(TokenKind::LitStr(symbol)) = self.peek_expect(TokenTag::LitStr) {
						self.bump();

						let str = core::str::from_utf8(&symbol).unwrap_or("");
						Extern::Explicit(Intern::from(str))
					} else {
						Extern::Implicit
					};
				},
				TokenKind::DirInline => {
					if unlikely(inline != Inline::None) {
						self.push_error(
							Diagnostic::error()
								.with_message("duplicate inline modifier")
								.with_label(Label::primary().with_span(self.diag_span(self.peek().span))),
						);
					}

					self.bump();
					inline = Inline::Always;
				},
				TokenKind::DirNoinline => {
					if unlikely(inline != Inline::None) {
						self.push_error(
							Diagnostic::error()
								.with_message("duplicate inline modifier")
								.with_label(Label::primary().with_span(self.diag_span(self.peek().span))),
						);
					}

					self.bump();
					inline = Inline::Never;
				},
				TokenKind::KwFn => break,
				_ => break,
			}
		}

		(ext, inline)
	}

	// =========================================================================
	//                                  Helpers
	// =========================================================================

	/// Get the next unique node ID.
	#[inline]
	fn next_id(&mut self) -> NodeId {
		let id = self.next_id;
		self.next_id += 1;
		NodeId::from_u32(id)
	}

	/// Advance to the next token and return the previous one.
	fn bump(&mut self) -> Token {
		assume!(self.offset < self.tokens.len());

		// SAFETY: `assume!` above guarantees offset is within bounds.
		let token = *unsafe { self.tokens.get_unchecked(self.offset) };
		if likely(!token.is_eof()) {
			self.offset += 1;
		}
		token
	}

	/// Get the previous token.
	#[inline(always)]
	fn prev(&self) -> &Token {
		assume!(self.offset > 0);

		// SAFETY: We assume that offset is always > 0 when calling this method.
		unsafe { self.tokens.get_unchecked(self.offset - 1) }
	}

	#[inline(always)]
	fn prev_nth(
		&self,
		nth: usize,
	) -> &Token {
		let idx = if likely(self.offset > nth) { self.offset - nth - 1 } else { 0 };

		// SAFETY: idx is guaranteed to be within bounds because we use 0
		// as a fallback when offset < nth + 1
		unsafe { self.tokens.get_unchecked(idx) }
	}

	/// Get the current token.
	#[inline(always)]
	fn peek(&self) -> &Token {
		assume!(self.offset < self.tokens.len());

		// SAFETY: We assume that offset is always < tokens.len() when calling this method.
		unsafe { self.tokens.get_unchecked(self.offset) }
	}

	/// Get the next nth token.
	#[inline(always)]
	fn peek_nth(
		&self,
		nth: usize,
	) -> &Token {
		let len = self.tokens.len();
		let idx = if likely(self.offset + nth < len) {
			self.offset + nth
		} else {
			len - 1
		};

		// SAFETY: idx is guaranteed to be within bounds because we use len - 1
		// as a fallback when offset + nth >= len
		unsafe { self.tokens.get_unchecked(idx) }
	}

	/// Expect the current token to be of the given kind.
	#[cfg_attr(debug_assertions, track_caller)]
	fn peek_expect(
		&self,
		tag: TokenTag,
	) -> Result<TokenKind, Diagnostic> {
		let token = self.peek();

		if unlikely(token.kind != tag) {
			Err(self.diag_expected_token(tag, token))
		} else {
			Ok(token.kind)
		}
	}

	/// Expect the current token to be of the given kind, and advance to the next token.
	#[cfg_attr(debug_assertions, track_caller)]
	fn eat_expect(
		&mut self,
		tag: TokenTag,
	) -> Result<TokenKind, Diagnostic> {
		let token = self.bump();

		if unlikely(token.kind != tag) {
			Err(self.diag_expected_token(tag, &token))
		} else {
			Ok(token.kind)
		}
	}

	/// If the current token is of the given kind, advance to the next token and return it.
	/// Otherwise, return `None`.
	fn eat_if(
		&mut self,
		tok: TokenTag,
	) -> Option<Token> {
		if self.peek().kind == tok { Some(self.bump()) } else { None }
	}

	/// Eat tokens until a token of the given kind is found or EOF is reached.
	/// Returns the kind of the found token.
	fn eat_until(
		&mut self,
		tag: TokenTag,
	) -> TokenKind {
		loop {
			let token = self.peek();

			if token.kind == tag || unlikely(token.is_eof()) {
				break token.kind;
			}

			self.offset += 1;
		}
	}

	/// Eat tokens until a token of one of the given kinds is found or EOF is reached.
	/// Returns the kind of the found token.
	fn eat_until2(
		&mut self,
		tag1: TokenTag,
		tag2: TokenTag,
	) -> TokenKind {
		loop {
			let token = self.peek();

			if token.kind == tag1 || token.kind == tag2 || unlikely(token.is_eof()) {
				break token.kind;
			}

			self.offset += 1;
		}
	}

	/// Eat tokens until a token of one of the given kinds is found or EOF is reached.
	/// Returns the kind of the found token.
	fn eat_until3(
		&mut self,
		tag1: TokenTag,
		tag2: TokenTag,
		tag3: TokenTag,
	) -> TokenKind {
		loop {
			let token = self.peek();

			if token.kind == tag1 || token.kind == tag2 || token.kind == tag3 || unlikely(token.is_eof()) {
				break token.kind;
			}

			self.offset += 1;
		}
	}

	#[inline]
	#[cfg_attr(debug_assertions, track_caller)]
	fn eat_group<T>(
		&mut self,
		opening_tok: TokenTag,
		closing_tok: TokenTag,
		mut producer: impl FnMut(&mut Self) -> Result<T, Diagnostic>,
	) -> Result<Vec<T, RcLinearAllocator>, Diagnostic> {
		match self.eat_expect(opening_tok) {
			Ok(_) => {},
			Err(err) => self.push_error(err),
		}

		if self.eat_if(closing_tok).is_some() {
			return Ok(Vec::new_in(self.linear_alloc.clone()));
		}

		let mut items = Vec::new_in(self.linear_alloc.clone());

		loop {
			let token = self.peek();

			if token.kind == closing_tok || unlikely(token.is_eof()) {
				break;
			}

			let start_offset = self.offset;
			match producer(self) {
				Ok(item) => items.push(item),
				Err(err) => {
					self.push_error(err);
					if self.offset == start_offset {
						self.offset += 1;
						if self.eat_until2(TokenTag::Semicolon, closing_tok) == TokenKind::Semicolon {
							self.offset += 1
						}
					}
				},
			}
		}

		self.eat_expect(closing_tok)?;
		Ok(items)
	}

	/// Eat a delimited list of items enclosed in opening and closing tokens,
	/// separated by a delimiter token.
	#[cfg_attr(debug_assertions, track_caller)]
	fn eat_delimited_group<T>(
		&mut self,
		opening_tok: TokenTag,
		closing_tok: TokenTag,
		delim_tok: TokenTag,
		mut producer: impl FnMut(&mut Self) -> Result<T, Diagnostic>,
	) -> Result<Vec<T, RcLinearAllocator>, Diagnostic> {
		match self.eat_expect(opening_tok) {
			Ok(_) => {},
			Err(err) => self.push_error(err),
		}

		if self.eat_if(closing_tok).is_some() {
			return Ok(Vec::new_in(self.linear_alloc.clone()));
		}

		let mut items = Vec::new_in(self.linear_alloc.clone());

		loop {
			let token = self.peek();

			if token.kind == closing_tok || unlikely(token.is_eof()) {
				break;
			}

			let start_offset = self.offset;
			match producer(self) {
				Ok(item) => items.push(item),
				Err(err) => {
					self.push_error(err);
					if self.offset == start_offset {
						self.offset += 1;
						match self.eat_until2(delim_tok, closing_tok) {
							token if token == delim_tok => self.offset += 1,
							_ => {},
						}
					}
				},
			}

			if self.eat_if(delim_tok).is_none() {
				break;
			}
		}

		self.eat_expect(closing_tok)?;
		Ok(items)
	}

	/// Eat a delimited list of items separated by a delimiter token.
	/// Continues eating items as long as the predicate returns true or
	/// a trailing delimiter is present.
	///
	/// Returns a tuple containing the list of items and a boolean indicating
	/// whether a trailing delimiter was present.
	fn eat_delimited<T>(
		&mut self,
		delim_tok: TokenTag,
		mut predicate: impl FnMut(&Token) -> bool,
		mut producer: impl FnMut(&mut Self) -> Result<T, Diagnostic>,
	) -> Result<(Vec<T, RcLinearAllocator>, bool), Diagnostic> {
		let mut items = Vec::new_in(self.linear_alloc.clone());

		loop {
			let token = self.peek();

			if !predicate(token) || unlikely(token.is_eof()) {
				break;
			}

			items.push(producer(self)?);

			if self.eat_if(delim_tok).is_none() {
				break;
			}
		}

		Ok((items, self.eat_if(delim_tok).is_some()))
	}

	// =========================================================================
	//                                 Errors
	// =========================================================================

	/// Push an error to the parser's error list.
	fn push_error(
		&mut self,
		diag: Diagnostic,
	) {
		self.errors
			.sorted_insert_by(diag, |a, b| match (a.primary_labels.first(), b.primary_labels.first()) {
				(Some(a), Some(b)) => a.span.le(&b.span),
				(None, Some(_)) => false,
				_ => true,
			});
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_unexpected_token(
		&self,
		token: &Token,
	) -> Diagnostic {
		if let Some(diag) = self.diag_unknown_directive(token) {
			return diag;
		}

		Diagnostic::error()
			.with_message(format!("unexpected token {}", token.kind))
			.with_label(Label::primary().with_span(self.diag_span(token.span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_token(
		&self,
		expected: TokenTag,
		found: &Token,
	) -> Diagnostic {
		if let Some(diag) = self.diag_unknown_directive(found) {
			return diag;
		}

		Diagnostic::error()
			.with_message(format!("expected '{}' found '{}'", expected, found.kind))
			.with_label(Label::primary().with_span(self.diag_span(found.span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_one_of_token(
		&self,
		expected: &[TokenTag],
		found: &Token,
	) -> Diagnostic {
		if let Some(diag) = self.diag_unknown_directive(found) {
			return diag;
		}

		let expected = match expected {
			[first] => format!("'{}'", first),
			[first, second] => format!("'{}' or '{}'", first, second),
			[first, middle @ .., last] => {
				let mut s = format!("'{}'", first);
				for kind in middle {
					s.push_str(&format!(", '{}'", kind));
				}
				s.push_str(&format!(" or '{}'", last));
				s
			},
			_ => {
				panic!("diag_expected_any_of_token called with an empty slice");
			},
		};

		Diagnostic::error()
			.with_message(format!("expected {} found '{}'", expected, found.kind))
			.with_label(Label::primary().with_span(self.diag_span(found.span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_redundant_qualifier(
		&self,
		qualifier: &str,
		span: Span,
	) -> Diagnostic {
		Diagnostic::error()
			.with_message(format!("redundant qualifier '{qualifier}'"))
			.with_label(Label::primary().with_span(self.diag_span(span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_type_expr(
		&self,
		prev: &Token,
		token: &Token,
	) -> Diagnostic {
		if let Some(diag) = self.diag_unknown_directive(token) {
			return diag;
		}

		Diagnostic::error()
			.with_message("expected type expression")
			.with_label(Label::secondary().with_span(self.diag_span(prev.span)))
			.with_label(Label::primary().with_span(self.diag_span(token.span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_expression(
		&self,
		token: &Token,
	) -> Diagnostic {
		if let Some(diag) = self.diag_unknown_directive(token) {
			return diag;
		}

		Diagnostic::error()
			.with_message("expected expression")
			.with_label(Label::primary().with_span(self.diag_span(token.span)))
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_unknown_directive(
		&self,
		token: &Token,
	) -> Option<Diagnostic> {
		let TokenKind::DirectiveIdent { symbol } = token.kind else {
			return None;
		};

		Some(
			Diagnostic::error()
				.with_message(format!("unknown directive `#{symbol}`"))
				.with_label(Label::primary().with_span(self.diag_span(token.span)))
				.with_note("known directives are `#inline`, `#noinline`, `#callconv`, `#linear`, `#packed`, `#volatile`, and `#addrspace`"),
		)
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_missing_semicolon(
		&self,
		token: &Token,
	) -> Diagnostic {
		Diagnostic::error()
			.with_message("missing semicolon")
			.with_label(Label::primary().with_span(self.diag_span(token.span)))
	}

	#[cold]
	#[inline(always)]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_inlinable(
		&self,
		token: &Token,
	) -> Diagnostic {
		self.diag_expected_one_of_token(&[TokenTag::KwWhile, TokenTag::KwFor], token)
	}

	#[cold]
	#[cfg_attr(debug_assertions, track_caller)]
	fn diag_expected_user_ident(
		&self,
		ident: &Ident,
	) -> Diagnostic {
		Diagnostic::error()
			.with_message(format!("expected user identifier, found '{}'", ident.symbol))
			.with_label(Label::primary().with_span(self.diag_span(ident.span)))
	}
}
