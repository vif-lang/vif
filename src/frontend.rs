pub mod ast;
pub mod lexer;
pub mod parser;

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Radix {
	/// Binary 0 or 1
	Binary = 2,
	/// Octal 0-7
	Octal = 8,
	/// Decimal 0-9
	Decimal = 10,
	/// Hexadecimal with upper or lowercase letters up to F.
	Hexadecimal = 16,
}

impl Radix {
	#[inline(always)]
	pub fn base(&self) -> u8 {
		(*self) as u8
	}
}

#[cfg(feature = "llvm")]
impl From<Radix> for inkwell::types::StringRadix {
	#[inline(always)]
	fn from(base: Radix) -> Self {
		match base {
			Radix::Binary => Self::Binary,
			Radix::Octal => Self::Octal,
			Radix::Decimal => Self::Decimal,
			Radix::Hexadecimal => Self::Hexadecimal,
		}
	}
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum IdentKind {
	User,
	UserEscaped,
	Generic,
	Builtin,
}

impl IdentKind {
	#[inline(always)]
	pub const fn is_user(&self) -> bool {
		matches!(self, Self::User | Self::UserEscaped)
	}

	#[inline(always)]
	pub const fn is_builtin(&self) -> bool {
		matches!(self, Self::Builtin)
	}

	#[inline(always)]
	pub const fn is_generic(&self) -> bool {
		matches!(self, Self::Generic)
	}
}
