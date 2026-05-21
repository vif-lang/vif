use core::hint::{
	likely,
	unlikely,
	unreachable_unchecked,
};

use internment::Intern;

use super::{
	IdentKind,
	Radix,
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
};

pub mod char_class {
	pub const IDENT_START: u8 = 1 << 0;
	pub const IDENT_CONT: u8 = IDENT_START | DIGIT;
	pub const DIGIT: u8 = 1 << 1;
	pub const BIN_DIGIT: u8 = 1 << 2;
	pub const OCT_DIGIT: u8 = 1 << 3;
	pub const HEX_DIGIT: u8 = 1 << 4;
	pub const WHITESPACE: u8 = 1 << 5;
	pub const UNDERSCORE: u8 = 1 << 6;
	pub const SIGN: u8 = 1 << 7;

	pub static CHAR_TABLE: [u8; 256] = {
		let mut table = [0u8; 256];
		let mut i = 0usize;
		while i < 256 {
			let b = i as u8;
			let mut flags = 0u8;

			if matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'_') {
				flags |= IDENT_START;
			}

			if b.is_ascii_digit() {
				flags |= DIGIT;
			}

			if matches!(b, b'0'..=b'1') {
				flags |= BIN_DIGIT;
			}

			if matches!(b, b'0'..=b'7') {
				flags |= OCT_DIGIT;
			}

			if b.is_ascii_hexdigit() {
				flags |= HEX_DIGIT;
			}

			if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
				flags |= WHITESPACE;
			}

			if b == b'_' {
				flags |= UNDERSCORE;
			}

			if matches!(b, b'+' | b'-') {
				flags |= SIGN;
			}

			table[i] = flags;
			i += 1;
		}
		table
	};

	#[inline(always)]
	pub const fn is_ident_start(b: u8) -> bool {
		CHAR_TABLE[b as usize] & IDENT_START != 0
	}

	#[inline(always)]
	pub const fn is_ident_cont(b: u8) -> bool {
		CHAR_TABLE[b as usize] & IDENT_CONT != 0
	}

	#[inline(always)]
	pub const fn is_digit(b: u8) -> bool {
		CHAR_TABLE[b as usize] & DIGIT != 0
	}

	#[inline(always)]
	pub const fn is_digit_or_sign(b: u8) -> bool {
		CHAR_TABLE[b as usize] & (DIGIT | SIGN) != 0
	}

	#[inline(always)]
	pub const fn is_digit_or_uscore(b: u8) -> bool {
		CHAR_TABLE[b as usize] & (DIGIT | UNDERSCORE) != 0
	}

	#[inline(always)]
	pub const fn is_bin_digit_or_uscore(b: u8) -> bool {
		CHAR_TABLE[b as usize] & (BIN_DIGIT | UNDERSCORE) != 0
	}

	#[inline(always)]
	pub const fn is_oct_digit_or_uscore(b: u8) -> bool {
		CHAR_TABLE[b as usize] & (OCT_DIGIT | UNDERSCORE) != 0
	}

	#[inline(always)]
	pub const fn is_hex_digit_or_uscore(b: u8) -> bool {
		CHAR_TABLE[b as usize] & (HEX_DIGIT | UNDERSCORE) != 0
	}

	#[inline(always)]
	pub const fn is_whitespace(b: u8) -> bool {
		CHAR_TABLE[b as usize] & WHITESPACE != 0
	}
}

macro_rules! tokens {
	($(
		$(#[doc = $doc:tt])*
		$(@$kw:ident)? $variant:ident
			$(($($tuple_field:ty),*))?
			$({ $($struct_field:ident: $struct_field_type:ty),* $(,)? })?
				= $display:literal
	),+ $(,)?) => {
		pub const AVG_KEYWORD_LEN: usize = const {
			let mut total_len = 0;
			let mut index = 0;
			$(
				if tokens!(@is_keyword: $($kw)*) {
					total_len += $display.len();
					index += 1;
				}
			)*
			total_len / index
		};

		#[allow(unused)]
		#[repr(u8)]
		#[derive(Debug)]
		pub enum TokenTag {
			$(
				$(#[doc = $doc])*
				$variant,
			)+
		}

		impl PartialEq for TokenTag {
			#[inline(always)]
			fn eq(
				&self,
				other: &Self,
			) -> bool {
				core::mem::discriminant(self) == core::mem::discriminant(other)
			}

			#[inline(always)]
			fn ne(
				&self,
				other: &Self,
			) -> bool {
				core::mem::discriminant(self) != core::mem::discriminant(other)
			}
		}

		impl Eq for TokenTag {}

		impl Clone for TokenTag {
			#[inline(always)]
			fn clone(&self) -> Self {
				*self
			}
		}

		impl Copy for TokenTag {}

		#[allow(unused)]
		#[repr(u8)]
		#[derive(Debug)]
		pub enum TokenKind {
			$(
				$(#[doc = $doc])*
				$variant $( ( $($tuple_field),* ) )? $({ $($struct_field: $struct_field_type),* })?,
			)+
		}

		impl PartialEq for TokenKind {
			#[inline(always)]
			fn eq(
				&self,
				other: &Self,
			) -> bool {
				core::mem::discriminant(self) == core::mem::discriminant(other)
			}

			#[inline(always)]
			fn ne(
				&self,
				other: &Self,
			) -> bool {
				core::mem::discriminant(self) != core::mem::discriminant(other)
			}
		}

		impl Eq for TokenKind {}

		impl Clone for TokenKind {
			#[inline(always)]
			fn clone(&self) -> Self {
				*self
			}
		}

		impl Copy for TokenKind {}

		impl TokenKind {
			#[inline(always)]
			pub const fn tag(&self) -> TokenTag {
				let discriminant = core::mem::discriminant(self);

				// SAFETY: TokenTag is mapped 1:1 to TokenKind and have the same size
				unsafe {
					core::mem::transmute::<_, TokenTag>(discriminant)
				}
			}
		}

		impl PartialEq<TokenTag> for TokenKind {
			#[inline(always)]
			fn eq(&self, other: &TokenTag) -> bool {
				self.tag() == *other
			}

			#[inline(always)]
			fn ne(&self, other: &TokenTag) -> bool {
				self.tag() != *other
			}
		}

		impl std::fmt::Display for TokenTag {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				match self {
					$(
						TokenTag::$variant => f.write_str($display),
					)+
				}
			}
		}

		impl std::fmt::Display for TokenKind {
			#[inline(always)]
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
				self.tag().fmt(f)
			}
		}

		const MIN_KEYWORD_LEN: usize = {
			let mut min = usize::MAX;
			$(
				if tokens!(@is_keyword: $($kw)*) {
					let len = $display.len();
					if len < min {
						min = len;
					}
				}
			)*
			min
		};

		const MAX_KEYWORD_LEN: usize = {
			let mut max = 0;
			$(
				if tokens!(@is_keyword: $($kw)*) {
					let len = $display.len();
					if len > max {
						max = len;
					}
				}
			)*
			max
		};
	};
	(@is_keyword: keyword) => { true };
	(@is_keyword: $($kw:tt)*) => { false };
}

tokens! {
	/// `-`
	Minus = "-",
	/// `+`
	Plus = "+",
	/// `*`
	Star = "*",
	/// `/`
	Slash = "/",
	/// `\`
	Backslash = "\\",
	/// `%`
	Percent = "%",
	/// `^`
	Caret = "^",
	/// `&`
	Amp = "&",
	/// `|`
	Pipe = "|",
	/// `!`
	Bang = "!",
	/// `~`
	Tilde = "~",
	/// `=`
	Eq = "=",
	/// `?`
	QuestionMark = "?",
	/// `<`
	Lt = "<",
	/// `>`
	Gt = ">",
	/// `,`
	Comma = ",",
	/// `;`
	Semicolon = ";",
	/// `:`
	Colon = ":",
	/// `(`
	LParen = "(",
	/// `)`
	RParen = ")",
	/// `{`
	LBrace = "{",
	/// `}`
	RBrace = "}",
	/// `[`
	LBracket = "[",
	/// `]`
	RBracket = "]",
	/// `#`
	Hash = "#",
	/// `.`
	Dot = ".",

	/// `-=`
	MinusEq = "-=",
	/// `-|`
	MinusPipe = "-|",
	/// `-%`
	MinusPercent = "-%",
	/// `+=`
	PlusEq = "+=",
	/// `+|`
	PlusPipe = "+|",
	/// `+%`
	PlusPercent = "+%",
	/// `**`
	StarStar = "**",
	/// `*=`
	StarEq = "*=",
	/// `*|`
	StarPipe = "*|",
	/// `*%`
	StarPercent = "*%",
	/// `/=`
	SlashEq = "/=",
	/// `%=`
	PercentEq = "%=",
	/// `<=`
	LtEq = "<=",
	/// `<<`
	LtLt = "<<",
	/// `>>`
	GtGt = ">>",
	/// `>=`
	GtEq = ">=",
	/// `^=`
	CaretEq = "^=",
	/// `&=`
	AmpEq = "&=",
	/// `|=`
	PipeEq = "|=",
	/// `&&`
	AmpAmp = "&&",
	/// `||`
	PipePipe = "||",
	/// `==`
	EqEq = "==",
	/// `!=`
	BangEq = "!=",
	/// `~=`
	TildeEq = "~=",
	/// `=>`
	FatArrow = "=>",
	/// `..`
	DotDot = "..",
	/// `.#`
	DotHash = ".#",
	/// `.*`
	DotStar = ".*",
	/// `.?`
	DotQuestionMark = ".?",

	/// `-|=`
	MinusPipeEq = "-|=",
	/// `-%=`
	MinusPercentEq = "-%=",
	/// `+|=`
	PlusPipeEq = "+|=",
	/// `+%=`
	PlusPercentEq = "+%=",
	/// `<<=`
	LtLtEq = "<<=",
	/// `>>=`
	GtGtEq = ">>=",
	/// `<<|`
	LtLtPipe = "<<|",
	/// `>>|`
	GtGtPipe = ">>|",
	/// `<<%`
	LtLtPercent = "<<%",
	/// `>>%`
	GtGtPercent = ">>%",
	/// `**|`
	StarStarPipe = "**|",
	/// `**%`
	StarStarPercent = "**%",
	/// `**=`
	StarStarEq = "**=",
	/// `*|=`
	StarPipeEq = "*|=",
	/// `*%=`
	StarPercentEq = "*%=",
	/// `&&=`
	AmpAmpEq = "&&=",
	/// `||=`
	PipePipeEq = "||=",
	/// `..=`
	DotDotEq = "..=",
	/// `...`
	Ellipsis = "...",

	/// `<<|=`
	LtLtPipeEq = "<<|=",
	/// `<<%=`
	LtLtPercentEq = "<<%=",
	/// `>>|=`
	GtGtPipeEq = ">>|=",
	/// `>>%=`
	GtGtPercentEq = ">>%=",
	/// `**|=`
	StarStarPipeEq = "**|=",
	/// `**%=`
	StarStarPercentEq = "**%=",

	/// Identifier
	Ident {
		symbol: Intern<str>,
		kind: IdentKind,
	} = "identifier",
	/// Unknown `#...` directive
	DirectiveIdent {
		symbol: Intern<str>,
	} = "directive",

	/// `123`
	/// `0x1A3F`
	/// `0o765`
	/// `0b1101`
	LitInt {
		symbol: Intern<str>,
		radix: Radix,
	} = "int literal",
	/// `3.14`
	/// `0.1e10`
	/// `.0`
	LitFloat {
		symbol: Intern<str>,
	} = "float literal",
	/// `true` | `false`
	LitBool(bool) = "bool literal",
	/// `"hello"`
	LitStr(Intern<[u8]>) = "string literal",
	/// ```
	/// \\hello
	/// \\world
	/// ```
	LitMultiLineStr(Intern<[u8]>) = "multi-line string literal",
	/// `'a'`
	LitChar(u8) = "char literal",
	/// `null`
	LitNull = "null literal",

	/// `/// documentation`
	DocComment = "doc comment",
	/// `//! mod documentation`
	ModComment = "mod comment",

	TyUsize = "usize",
	TyIsize = "isize",
	TyU(u16) = "uint",
	TyI(u16) = "int",
	TyF16 = "f16",
	TyF32 = "f32",
	TyF64 = "f64",
	TyF128 = "f128",
	TyBool = "bool",
	TyVoid = "void",
	TyNever = "never",
	TyAny = "any",
	TyAnyint = "anyint",
	TyAnyfloat = "anyfloat",
	TyAnyerror = "anyerror",
	TyType = "type",

	@keyword KwFn = "fn",
	@keyword KwIf = "if",
	@keyword KwIn = "in",
	@keyword KwOr = "or",
	@keyword KwAnd = "and",
	@keyword KwPub = "pub",
	@keyword KwVar = "var",
	@keyword KwFor = "for",
	@keyword KwTry = "try",
	@keyword KwElse = "else",
	@keyword KwLoop = "loop",
	@keyword KwEnum = "enum",
	@keyword KwTest = "test",
	@keyword KwConst = "const",
	@keyword KwComptime = "comptime",
	@keyword KwWhile = "while",
	@keyword KwBreak = "break",
	@keyword KwDefer = "defer",
	@keyword KwCatch = "catch",
	@keyword KwUnion = "union",
	@keyword KwError = "error",
	@keyword KwSwitch = "switch",
	@keyword KwReturn = "return",
	@keyword KwStruct = "struct",
	@keyword KwExtern = "extern",
	DirInline = "#inline",
	DirPacked = "#packed",
	DirLinear = "#linear",
	@keyword KwConcept = "concept",
	DirCallconv = "#callconv",
	@keyword KwRequires = "requires",
	@keyword KwContinue = "continue",
	@keyword KwErrdefer = "errdefer",
	DirNoinline = "#noinline",
	DirVolatile = "#volatile",
	DirAddrspace = "#addrspace",
	@keyword KwUndefined = "undefined",
	@keyword KwUnreachable = "unreachable",

	Eof = "EOF",
}

#[derive(Copy, Clone, Debug)]
pub struct Token {
	pub kind: TokenKind,
	/// Span of the token in the source code
	pub span: Span,
}

impl Token {
	#[inline(always)]
	pub const fn is_eof(&self) -> bool {
		unlikely(matches!(self.kind, TokenKind::Eof))
	}
}

impl PartialEq for Token {
	#[inline(always)]
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(&self.kind) == core::mem::discriminant(&other.kind) && self.span == other.span
	}

	#[inline(always)]
	fn ne(
		&self,
		other: &Self,
	) -> bool {
		core::mem::discriminant(&self.kind) != core::mem::discriminant(&other.kind) || self.span != other.span
	}
}

impl core::fmt::Display for Token {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		write!(f, "<{}@{}>", self.kind, self.span)
	}
}

const EOF: u8 = 0;

pub struct Lexer<'src> {
	bytes: &'src [u8],
	len: usize,
	offset: usize,
	span_start: usize,
	module_id: ModuleId,
	errors: Vec<Diagnostic>,
	scratch_pad: Vec<u8>,
}

impl<'src> Lexer<'src> {
	#[inline(always)]
	pub fn new(
		source: &'src str,
		module_id: ModuleId,
	) -> Self {
		let bytes = source.as_bytes();
		let len = bytes.len() - 1;

		Self {
			bytes,
			len,
			offset: 0,
			span_start: 0,
			module_id,
			errors: Vec::new(),
			scratch_pad: Vec::with_capacity(128),
		}
	}

	#[inline(always)]
	pub fn take_errors(&mut self) -> Vec<Diagnostic> {
		core::mem::take(&mut self.errors)
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
	pub fn next(&mut self) -> Token {
		#[derive(Copy, Clone)]
		enum State {
			S0,

			Minus,
			MinusPipe,
			MinusPercent,

			Plus,
			PlusPipe,
			PlusPercent,

			Star,
			StarPipe,
			StarPercent,

			StarStar,
			StarStarPipe,
			StarStarPercent,

			Slash,
			Percent,
			Caret,
			Amp,
			Pipe,
			Bang,
			Tilde,
			Eq,
			Hash,
			At,
			Dollar,

			Lt,
			LtLt,
			LtLtPipe,
			LtLtPercent,

			Gt,
			GtGt,
			GtGtPipe,
			GtGtPercent,

			AmpAmp,
			PipePipe,

			Dot,
			DotDot,

			Backslash,

			LineComment,

			Ident,
			Str,
			EscapeStr,
			IntZero,
			IntBin,
			IntOct,
			IntHex,
			IntDec,
			FloatDot1,
			FloatDot2,
			FloatExp1,
			FloatExp2,
		}

		use State::*;

		let mut state = S0;
		let mut kind = TokenKind::Eof;

		#[loop_match]
		'lexer: loop {
			state = 'state: {
				match state {
					S0 => {
						if likely(char_class::is_whitespace(self.peek()))
							&& let Some(offset) = memx::memnechr_qpl(&self.bytes[self.offset..], b' ', b'\t', b'\n', b'\r')
						{
							self.offset += offset;
						}

						self.span_start = self.offset;

						match self.bump() {
							EOF => {
								kind = TokenKind::Eof;
								break 'lexer;
							},
							b'?' => {
								kind = TokenKind::QuestionMark;
								break 'lexer;
							},
							b',' => {
								kind = TokenKind::Comma;
								break 'lexer;
							},
							b';' => {
								kind = TokenKind::Semicolon;
								break 'lexer;
							},
							b':' => {
								kind = TokenKind::Colon;
								break 'lexer;
							},
							b'(' => {
								kind = TokenKind::LParen;
								break 'lexer;
							},
							b')' => {
								kind = TokenKind::RParen;
								break 'lexer;
							},
							b'{' => {
								kind = TokenKind::LBrace;
								break 'lexer;
							},
							b'}' => {
								kind = TokenKind::RBrace;
								break 'lexer;
							},
							b'[' => {
								kind = TokenKind::LBracket;
								break 'lexer;
							},
							b']' => {
								kind = TokenKind::RBracket;
								break 'lexer;
							},
							b'#' => {
								kind = TokenKind::Hash;
								#[const_continue]
								break 'state Hash;
							},
							b'-' => {
								kind = TokenKind::Minus;
								#[const_continue]
								break 'state Minus;
							},
							b'+' => {
								kind = TokenKind::Plus;
								#[const_continue]
								break 'state Plus;
							},
							b'*' => {
								kind = TokenKind::Star;
								#[const_continue]
								break 'state Star;
							},
							b'/' => {
								kind = TokenKind::Slash;
								#[const_continue]
								break 'state Slash;
							},
							b'%' => {
								kind = TokenKind::Percent;
								#[const_continue]
								break 'state Percent;
							},
							b'^' => {
								kind = TokenKind::Caret;
								#[const_continue]
								break 'state Caret;
							},
							b'&' => {
								kind = TokenKind::Amp;
								#[const_continue]
								break 'state Amp;
							},
							b'|' => {
								kind = TokenKind::Pipe;
								#[const_continue]
								break 'state Pipe;
							},
							b'!' => {
								kind = TokenKind::Bang;
								#[const_continue]
								break 'state Bang;
							},
							b'~' => {
								kind = TokenKind::Tilde;
								#[const_continue]
								break 'state Tilde;
							},
							b'=' => {
								kind = TokenKind::Eq;
								#[const_continue]
								break 'state Eq;
							},
							b'@' => {
								#[const_continue]
								break 'state At;
							},
							b'$' => {
								#[const_continue]
								break 'state Dollar;
							},
							b'<' => {
								kind = TokenKind::Lt;
								#[const_continue]
								break 'state Lt;
							},
							b'>' => {
								kind = TokenKind::Gt;
								#[const_continue]
								break 'state Gt;
							},
							b'.' => {
								kind = TokenKind::Dot;
								#[const_continue]
								break 'state Dot;
							},
							b'\\' => {
								#[const_continue]
								break 'state Backslash;
							},
							b'"' => {
								kind = TokenKind::LitStr(COMMON_INTERNS.empty_bytes);
								#[const_continue]
								break 'state Str;
							},
							b'0' => {
								#[const_continue]
								break 'state IntZero;
							},
							chr => match char_class::CHAR_TABLE[chr as usize] {
								class if class & char_class::IDENT_START != 0 => {
									kind = TokenKind::Ident {
										symbol: COMMON_INTERNS.empty_str,
										kind: IdentKind::User,
									};

									#[const_continue]
									break 'state Ident;
								},
								class if class & char_class::DIGIT != 0 => {
									#[const_continue]
									break 'state IntDec;
								},
								chr => {
									self.diag_unexpected_character(chr);
									kind = TokenKind::Eof;
									#[const_continue]
									break 'state S0;
								},
							},
						}
					},
					Minus => match self.bump() {
						b'=' => {
							kind = TokenKind::MinusEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::MinusPipe;
							#[const_continue]
							break 'state MinusPipe;
						},
						b'%' => {
							kind = TokenKind::MinusPercent;
							#[const_continue]
							break 'state MinusPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					MinusPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::MinusPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					MinusPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::MinusPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Plus => match self.bump() {
						b'=' => {
							kind = TokenKind::PlusEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::PlusPipe;
							#[const_continue]
							break 'state PlusPipe;
						},
						b'%' => {
							kind = TokenKind::PlusPercent;
							#[const_continue]
							break 'state PlusPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					PlusPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::PlusPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					PlusPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::PlusPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Star => match self.bump() {
						b'*' => {
							kind = TokenKind::StarStar;
							#[const_continue]
							break 'state StarStar;
						},
						b'|' => {
							kind = TokenKind::StarPipe;
							#[const_continue]
							break 'state StarPipe;
						},
						b'%' => {
							kind = TokenKind::StarPercent;
							#[const_continue]
							break 'state StarPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					StarPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::StarPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					StarPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::StarPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					StarStar => match self.bump() {
						b'=' => {
							kind = TokenKind::StarStarEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::StarStarPipe;
							#[const_continue]
							break 'state StarStarPipe;
						},
						b'%' => {
							kind = TokenKind::StarStarPercent;
							#[const_continue]
							break 'state StarStarPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					StarStarPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::StarStarPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					StarStarPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::StarStarPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Slash => match self.bump() {
						b'=' => {
							kind = TokenKind::SlashEq;
							break 'lexer;
						},
						b'/' => {
							#[const_continue]
							break 'state LineComment;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Percent => match self.bump() {
						b'=' => {
							kind = TokenKind::PercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Caret => match self.bump() {
						b'=' => {
							kind = TokenKind::CaretEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Amp => match self.bump() {
						b'=' => {
							kind = TokenKind::AmpEq;
							break 'lexer;
						},
						b'&' => {
							kind = TokenKind::AmpAmp;
							#[const_continue]
							break 'state AmpAmp;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					AmpAmp => match self.bump() {
						b'=' => {
							kind = TokenKind::AmpAmpEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Pipe => match self.bump() {
						b'=' => {
							kind = TokenKind::PipeEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::PipePipe;
							#[const_continue]
							break 'state PipePipe;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					PipePipe => match self.bump() {
						b'=' => {
							kind = TokenKind::PipePipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Bang => match self.bump() {
						b'=' => {
							kind = TokenKind::BangEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Tilde => match self.bump() {
						b'=' => {
							kind = TokenKind::TildeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Eq => match self.bump() {
						b'=' => {
							kind = TokenKind::EqEq;
							break 'lexer;
						},
						b'>' => {
							kind = TokenKind::FatArrow;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Hash => match self.bump() {
						chr if char_class::is_ident_start(chr) => {
							while likely(char_class::is_ident_cont(self.peek())) {
								self.offset += 1;
							}

							let str = &self.bytes[self.span_start + 1..self.offset];
							kind = match str {
								b"inline" => TokenKind::DirInline,
								b"packed" => TokenKind::DirPacked,
								b"linear" => TokenKind::DirLinear,
								b"callconv" => TokenKind::DirCallconv,
								b"noinline" => TokenKind::DirNoinline,
								b"volatile" => TokenKind::DirVolatile,
								b"addrspace" => TokenKind::DirAddrspace,
								_ => TokenKind::DirectiveIdent { symbol: intern_str(str) },
							};
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					At => match self.bump() {
						chr if char_class::is_ident_start(chr) => {
							kind = TokenKind::Ident {
								symbol: COMMON_INTERNS.empty_str,
								kind: IdentKind::Builtin,
							};

							#[const_continue]
							break 'state Ident;
						},
						b'"' => {
							kind = TokenKind::Ident {
								symbol: COMMON_INTERNS.empty_str,
								kind: IdentKind::UserEscaped,
							};

							#[const_continue]
							break 'state Str;
						},
						chr => {
							self.diag_unexpected_character(chr);
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
					},
					Dollar => match self.bump() {
						chr if char_class::is_ident_start(chr) => {
							kind = TokenKind::Ident {
								symbol: COMMON_INTERNS.empty_str,
								kind: IdentKind::Generic,
							};

							#[const_continue]
							break 'state Ident;
						},
						chr => {
							self.diag_unexpected_character(chr);
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
					},
					Lt => match self.bump() {
						b'=' => {
							kind = TokenKind::LtEq;
							break 'lexer;
						},
						b'<' => {
							kind = TokenKind::LtLt;
							#[const_continue]
							break 'state LtLt;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					LtLt => match self.bump() {
						b'=' => {
							kind = TokenKind::LtLtEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::LtLtPipe;
							#[const_continue]
							break 'state LtLtPipe;
						},
						b'%' => {
							kind = TokenKind::LtLtPercent;
							#[const_continue]
							break 'state LtLtPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					LtLtPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::LtLtPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					LtLtPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::LtLtPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Gt => match self.bump() {
						b'=' => {
							kind = TokenKind::GtEq;
							break 'lexer;
						},
						b'>' => {
							kind = TokenKind::GtGt;
							#[const_continue]
							break 'state GtGt;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					GtGt => match self.bump() {
						b'=' => {
							kind = TokenKind::GtGtEq;
							break 'lexer;
						},
						b'|' => {
							kind = TokenKind::GtGtPipe;
							#[const_continue]
							break 'state GtGtPipe;
						},
						b'%' => {
							kind = TokenKind::GtGtPercent;
							#[const_continue]
							break 'state GtGtPercent;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					GtGtPipe => match self.bump() {
						b'=' => {
							kind = TokenKind::GtGtPipeEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					GtGtPercent => match self.bump() {
						b'=' => {
							kind = TokenKind::GtGtPercentEq;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Dot => match self.bump() {
						b'.' => {
							kind = TokenKind::DotDot;
							#[const_continue]
							break 'state DotDot;
						},
						b'#' => {
							kind = TokenKind::DotHash;
							break 'lexer;
						},
						b'*' => {
							kind = TokenKind::DotStar;
							break 'lexer;
						},
						b'?' => {
							kind = TokenKind::DotQuestionMark;
							break 'lexer;
						},
						chr if char_class::is_digit(chr) => {
							#[const_continue]
							break 'state FloatDot2;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					DotDot => match self.bump() {
						b'=' => {
							kind = TokenKind::DotDotEq;
							break 'lexer;
						},
						b'.' => {
							kind = TokenKind::Ellipsis;
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							break 'lexer;
						},
					},
					Backslash => match self.bump() {
						b'\\' => {
							self.eat_until_newline();
							let slice = &self.bytes[self.span_start + 2..self.offset];
							kind = TokenKind::LitMultiLineStr(intern_bytes(slice));
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							kind = TokenKind::Backslash;
							break 'lexer;
						},
					},
					LineComment => {
						// TODO(ldubos): for the moment we just skip comments entirely
						// we might want to store them later for doc generation/metaprogramming
						if true {
							self.eat_until_newline();
							#[const_continue]
							break 'state S0;
						} else {
							let discriminant = self.bump();

							if self.peek() == b' ' {
								self.offset += 1;
							}

							self.eat_until_newline();

							kind = match discriminant {
								b'/' => TokenKind::DocComment,
								b'!' => TokenKind::ModComment,
								_ => {
									#[const_continue]
									break 'state S0;
								},
							};

							break 'lexer;
						}
					},
					Ident => {
						while likely(char_class::is_ident_cont(self.peek())) {
							self.offset += 1;
						}

						let ident_kind = match &kind {
							TokenKind::Ident { kind, .. } => *kind,
							// SAFETY: we only enter Ident state when kind is Ident
							_ => unsafe { unreachable_unchecked() },
						};

						match ident_kind {
							IdentKind::UserEscaped => {
								// Remove the '@"' prefix and the '"' suffix
								kind = TokenKind::Ident {
									symbol: intern_str(&self.bytes[self.span_start + 2..self.offset - 1]),
									kind: IdentKind::UserEscaped,
								};
								break 'lexer;
							},
							ident_kind @ (IdentKind::Builtin | IdentKind::Generic) => {
								kind = TokenKind::Ident {
									symbol: intern_str(&self.bytes[self.span_start..self.offset]),
									kind: ident_kind,
								};
								break 'lexer;
							},
							_ => {},
						}

						let str = &self.bytes[self.span_start..self.offset];
						let len = str.len();

						let candidate = if unlikely(!(MIN_KEYWORD_LEN..=MAX_KEYWORD_LEN).contains(&len) || len == 10) {
							None
						} else {
							hashify::tiny_map! {
								str,
								"fn" => TokenKind::KwFn,
								"if" => TokenKind::KwIf,
								"in" => TokenKind::KwIn,
								"or" => TokenKind::KwOr,
								"i1" => TokenKind::TyI(1),
								"i2" => TokenKind::TyI(2),
								"i3" => TokenKind::TyI(3),
								"i4" => TokenKind::TyI(4),
								"i5" => TokenKind::TyI(5),
								"i6" => TokenKind::TyI(6),
								"i7" => TokenKind::TyI(7),
								"i8" => TokenKind::TyI(8),
								"i9" => TokenKind::TyI(9),
								"u1" => TokenKind::TyU(1),
								"u2" => TokenKind::TyU(2),
								"u3" => TokenKind::TyU(3),
								"u4" => TokenKind::TyU(4),
								"u5" => TokenKind::TyU(5),
								"u6" => TokenKind::TyU(6),
								"u7" => TokenKind::TyU(7),
								"u8" => TokenKind::TyU(8),
								"u9" => TokenKind::TyU(9),
								"and" => TokenKind::KwAnd,
								"pub" => TokenKind::KwPub,
								"var" => TokenKind::KwVar,
								"for" => TokenKind::KwFor,
								"try" => TokenKind::KwTry,
								"any" => TokenKind::TyAny,
								"f16" => TokenKind::TyF16,
								"f32" => TokenKind::TyF32,
								"f64" => TokenKind::TyF64,
								"i16" => TokenKind::TyI(16),
								"i32" => TokenKind::TyI(32),
								"i64" => TokenKind::TyI(64),
								"u16" => TokenKind::TyU(16),
								"u32" => TokenKind::TyU(32),
								"u64" => TokenKind::TyU(64),
								"self" => TokenKind::Ident {
									symbol: COMMON_INTERNS.self_symbol,
									kind: IdentKind::User,
								},
								"Self" => TokenKind::Ident {
									symbol: COMMON_INTERNS.self_ty_symbol,
									kind: IdentKind::User,
								},
								"else" => TokenKind::KwElse,
								"loop" => TokenKind::KwLoop,
								"enum" => TokenKind::KwEnum,
								"test" => TokenKind::KwTest,
								"null" => TokenKind::LitNull,
								"true" => TokenKind::LitBool(true),
								"void" => TokenKind::TyVoid,
								"never" => TokenKind::TyNever,
								"bool" => TokenKind::TyBool,
								"type" => TokenKind::TyType,
								"f128" => TokenKind::TyF128,
								"i128" => TokenKind::TyI(128),
								"u128" => TokenKind::TyU(128),
								"const" => TokenKind::KwConst,
								"while" => TokenKind::KwWhile,
								"break" => TokenKind::KwBreak,
								"defer" => TokenKind::KwDefer,
								"catch" => TokenKind::KwCatch,
								"union" => TokenKind::KwUnion,
								"error" => TokenKind::KwError,
								"false" => TokenKind::LitBool(false),
								"usize" => TokenKind::TyUsize,
								"isize" => TokenKind::TyIsize,
								"switch" => TokenKind::KwSwitch,
								"return" => TokenKind::KwReturn,
								"struct" => TokenKind::KwStruct,
								"extern" => TokenKind::KwExtern,
								"anyint" => TokenKind::TyAnyint,
								"comptime" => TokenKind::KwComptime,
								"concept" => TokenKind::KwConcept,
								"requires" => TokenKind::KwRequires,
								"continue" => TokenKind::KwContinue,
								"errdefer" => TokenKind::KwErrdefer,
								"anyfloat" => TokenKind::TyAnyfloat,
								"anyerror" => TokenKind::TyAnyerror,
								"undefined" => TokenKind::KwUndefined,
								"unreachable" => TokenKind::KwUnreachable,
							}
						};

						kind = match candidate {
							Some(kind) => kind,
							None => match str[0] {
								prefix @ (b'u' | b'i') if len >= 2 && str[1..].iter().copied().all(char_class::is_digit) => {
									let bits = u64::from_ascii_radix(&str[1..], 10).unwrap_or(0);

									if unlikely(bits == 0 || bits > u16::MAX as u64) {
										self.diag_invalid_integer_bit_width();
										kind = TokenKind::Eof;
										#[const_continue]
										break 'state S0;
									}

									match prefix {
										b'u' => TokenKind::TyU(bits as u16),
										b'i' => TokenKind::TyI(bits as u16),
										// SAFETY: we have the guarantee that kind is either 'u' or 'i'
										_ => unsafe { unreachable_unchecked() },
									}
								},
								_ => TokenKind::Ident {
									symbol: intern_str(str),
									kind: IdentKind::User,
								},
							},
						};

						break 'lexer;
					},
					Str => match self.bump() {
						EOF => {
							self.diag_unexpected_eof();
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
						b'\n' => {
							self.diag_unexpected_character(b'\n');
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
						b'"' => {
							match &mut kind {
								TokenKind::LitStr(symbol) => {
									*symbol = intern_bytes(&self.scratch_pad[..]);
								},
								TokenKind::Ident { symbol, .. } => {
									*symbol = intern_str(&self.scratch_pad[..]);
								},
								// SAFETY: we only enter Str state when kind is LitStr or Ident
								_ => unsafe { unreachable_unchecked() },
							};
							self.scratch_pad.clear();
							break 'lexer;
						},
						b'\\' => {
							#[const_continue]
							break 'state EscapeStr;
						},
						_ => {
							let start = self.offset - 1;
							self.eat_until3(b'"', b'\\', b'\n');
							self.scratch_pad.extend_from_slice(&self.bytes[start..self.offset]);

							#[const_continue]
							break 'state Str;
						},
					},
					EscapeStr => {
						match self.bump() {
							EOF => {
								self.diag_unexpected_eof();
								kind = TokenKind::Eof;
								#[const_continue]
								break 'state S0;
							},
							b'n' => self.scratch_pad.push(b'\n'),
							b'r' => self.scratch_pad.push(b'\r'),
							b't' => self.scratch_pad.push(b'\t'),
							b'"' => self.scratch_pad.push(b'"'),
							b'\\' => self.scratch_pad.push(b'\\'),
							b'0' => self.scratch_pad.push(b'\0'),
							b'x' => {
								let hi = self.bump();
								let lo = self.bump();

								if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
									let byte = hex_value(hi) << 4 | hex_value(lo);
									self.scratch_pad.push(byte);
								} else {
									self.diag_invalid_escape_sequence();
									kind = TokenKind::Eof;
									#[const_continue]
									break 'state S0;
								}
							},
							b'u' => {
								if self.bump() != b'{' {
									self.diag_invalid_unicode_escape();
									kind = TokenKind::Eof;
									#[const_continue]
									break 'state S0;
								}

								let mut codepoint: u32 = 0;

								while self.peek() != b'}' {
									if unlikely(self.is_eof()) {
										self.diag_unexpected_eof();
										kind = TokenKind::Eof;
										#[const_continue]
										break 'state S0;
									}

									let digit = self.bump();

									if digit.is_ascii_hexdigit() {
										codepoint = (codepoint << 4) | (hex_value(digit) as u32);
									} else {
										self.diag_invalid_unicode_escape();
										kind = TokenKind::Eof;
										#[const_continue]
										break 'state S0;
									}
								}

								self.offset += 1; // consume the closing '}'

								let chr = match core::char::from_u32(codepoint) {
									Some(c) => c,
									None => {
										self.diag_invalid_unicode_escape();
										kind = TokenKind::Eof;
										#[const_continue]
										break 'state S0;
									},
								};

								let mut buf: [u8; 4] = [0; 4];
								let encoded = chr.encode_utf8(&mut buf);
								self.scratch_pad.extend_from_slice(encoded.as_bytes());
							},
							_ => {
								self.diag_invalid_escape_sequence();
								kind = TokenKind::Eof;
								#[const_continue]
								break 'state S0;
							},
						}

						#[const_continue]
						break 'state Str;
					},
					IntZero => match self.bump() {
						b'b' => {
							#[const_continue]
							break 'state IntBin;
						},
						b'o' => {
							#[const_continue]
							break 'state IntOct;
						},
						b'x' => {
							#[const_continue]
							break 'state IntHex;
						},
						b'.' => {
							#[const_continue]
							break 'state FloatDot1;
						},
						b'e' | b'E' => {
							#[const_continue]
							break 'state FloatExp1;
						},
						chr if char_class::is_bin_digit_or_uscore(chr) => {
							#[const_continue]
							break 'state IntDec;
						},
						_ => {
							kind = TokenKind::LitInt {
								// SAFETY: '0' is guaranteed to be in 0..=9
								symbol: unsafe { *COMMON_INTERNS.int_digits.get_unchecked(0) },
								radix: Radix::Decimal,
							};
							self.offset -= 1;
							break 'lexer;
						},
					},
					IntBin => {
						while likely(char_class::is_bin_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						kind = TokenKind::LitInt {
							symbol: intern_str(&self.bytes[self.span_start..self.offset]),
							radix: Radix::Binary,
						};
						break 'lexer;
					},
					IntOct => {
						while likely(char_class::is_oct_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						kind = TokenKind::LitInt {
							symbol: intern_str(&self.bytes[self.span_start..self.offset]),
							radix: Radix::Octal,
						};
						break 'lexer;
					},
					IntHex => {
						while likely(char_class::is_hex_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						kind = TokenKind::LitInt {
							symbol: intern_str(&self.bytes[self.span_start..self.offset]),
							radix: Radix::Hexadecimal,
						};
						break 'lexer;
					},
					IntDec => {
						while likely(char_class::is_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						match self.peek() {
							b'.' => {
								self.offset += 1;
								#[const_continue]
								break 'state FloatDot1;
							},
							b'e' | b'E' => {
								self.offset += 1;
								#[const_continue]
								break 'state FloatExp1;
							},
							_ => {
								if self.offset - self.span_start == 1 {
									let digit = self.bytes[self.span_start] - b'0';
									assume!(digit <= 9);

									kind = TokenKind::LitInt {
										// SAFETY: digit is guaranteed to be in 0..=9
										symbol: unsafe { *COMMON_INTERNS.int_digits.get_unchecked(digit as usize) },
										radix: Radix::Decimal,
									};
								} else {
									kind = TokenKind::LitInt {
										symbol: intern_str(&self.bytes[self.span_start..self.offset]),
										radix: Radix::Decimal,
									};
								}
								break 'lexer;
							},
						}
					},
					FloatDot1 => match self.bump() {
						b'e' | b'E' => {
							self.diag_invalid_float_literal();
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
						chr if char_class::is_digit(chr) => {
							#[const_continue]
							break 'state FloatDot2;
						},
						b'.' => {
							// .. => backtrack
							self.offset -= 2;
							let symbol = intern_str(&self.bytes[self.span_start..self.offset]);
							kind = TokenKind::LitInt {
								symbol,
								radix: Radix::Decimal,
							};
							break 'lexer;
						},
						_ => {
							self.offset -= 1;
							let symbol = intern_str(&self.bytes[self.span_start..self.offset]);
							kind = TokenKind::LitFloat { symbol };
							break 'lexer;
						},
					},
					FloatDot2 => {
						while likely(char_class::is_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						match self.peek() {
							b'e' | b'E' => {
								self.offset += 1;
								kind = TokenKind::LitFloat {
									symbol: COMMON_INTERNS.empty_str,
								};
								#[const_continue]
								break 'state FloatExp1;
							},
							_ => {
								let symbol = intern_str(&self.bytes[self.span_start..self.offset]);
								kind = TokenKind::LitFloat { symbol };
								break 'lexer;
							},
						}
					},
					FloatExp1 => match self.peek() {
						chr if char_class::is_digit_or_sign(chr) => {
							self.offset += 1;

							#[const_continue]
							break 'state FloatExp2;
						},
						_ => {
							self.diag_invalid_float_literal();
							kind = TokenKind::Eof;
							#[const_continue]
							break 'state S0;
						},
					},
					FloatExp2 => {
						while likely(char_class::is_digit_or_uscore(self.peek())) {
							self.offset += 1;
						}

						let symbol = intern_str(&self.bytes[self.span_start..self.offset]);
						kind = TokenKind::LitFloat { symbol };
						break 'lexer;
					},
				}
			};
		}

		let span = Span::new(self.span_start..self.offset);
		Token { kind, span }
	}

	#[inline(always)]
	fn bump(&mut self) -> u8 {
		// SAFETY: we should always read from a null-terminated string for this to works
		let b = unsafe { *self.bytes.get_unchecked(self.offset) };
		self.offset += 1;
		b
	}

	#[inline(always)]
	fn peek(&self) -> u8 {
		// SAFETY: we should always read from a null-terminated string for this to works
		unsafe { *self.bytes.get_unchecked(self.offset) }
	}

	#[inline(always)]
	fn is_eof(&self) -> bool {
		self.offset >= self.len
	}

	fn eat_until_newline(&mut self) {
		while let Some(offset) = memx::memchr_dbl(&self.bytes[self.offset..], b'\n', b'\r') {
			if likely(self.bytes[self.offset + offset] == b'\n') {
				self.offset += offset;
				break;
			} else {
				let offset = self.offset + offset;

				// Check for \r\n sequence
				if likely(offset + 1 < self.len && self.bytes[offset + 1] == b'\n') {
					self.offset = offset + 1;
				} else {
					self.offset = offset;
				}
			}
		}
	}

	#[inline(always)]
	fn eat_until(
		&mut self,
		needle: u8,
	) {
		if let Some(offset) = memx::memchr(&self.bytes[self.offset..], needle) {
			self.offset += offset;
		} else {
			self.offset = self.len;
		}
	}

	fn eat_until3(
		&mut self,
		needle1: u8,
		needle2: u8,
		needle3: u8,
	) {
		if let Some(offset) = memx::memchr_tpl(&self.bytes[self.offset..], needle1, needle2, needle3) {
			self.offset += offset;
		} else {
			self.offset = self.len;
		}
	}

	#[inline(always)]
	fn span(&self) -> Span {
		Span::new(self.span_start..self.offset)
	}

	#[cold]
	fn diag_unexpected_eof(&mut self) {
		self.errors.push(
			Diagnostic::error()
				.with_message("unexpected end of file")
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}

	#[cold]
	fn diag_unexpected_character(
		&mut self,
		chr: u8,
	) {
		self.errors.push(
			Diagnostic::error()
				.with_message(format!("unexpected character '{}'", chr as char))
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}

	#[cold]
	fn diag_invalid_escape_sequence(&mut self) {
		self.errors.push(
			Diagnostic::error()
				.with_message("invalid escape sequence")
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}

	#[cold]
	fn diag_invalid_unicode_escape(&mut self) {
		self.errors.push(
			Diagnostic::error()
				.with_message("invalid unicode escape")
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}

	#[cold]
	fn diag_invalid_float_literal(&mut self) {
		self.errors.push(
			Diagnostic::error()
				.with_message("invalid float literal")
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}

	#[cold]
	fn diag_invalid_integer_bit_width(&mut self) {
		self.errors.push(
			Diagnostic::error()
				.with_message("invalid integer bit width")
				.with_label(Label::primary().with_span(self.diag_span(self.span()))),
		);
	}
}

#[inline(always)]
fn intern_str(slice: &[u8]) -> Intern<str> {
	// SAFETY: we assume the lexer only produces valid UTF-8 sequences
	let str = unsafe { core::str::from_utf8_unchecked(slice) };
	Intern::from(str)
}

#[inline(always)]
fn intern_bytes(slice: &[u8]) -> Intern<[u8]> {
	Intern::from(slice)
}

#[inline(always)]
fn hex_value(chr: u8) -> u8 {
	assume!(chr.is_ascii_hexdigit(), "character is not a valid hexadecimal digit");

	match chr {
		b'0'..=b'9' => chr - b'0',
		b'a'..=b'f' => chr - b'a' + 10,
		b'A'..=b'F' => chr - b'A' + 10,
		// SAFETY: we already checked the character is a hex digit
		_ => unsafe { unreachable_unchecked() },
	}
}
