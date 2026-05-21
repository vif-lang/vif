use std::sync::LazyLock;

use internment::Intern;

pub struct CommonInterns {
	pub empty_str: Intern<str>,
	pub empty_bytes: Intern<[u8]>,
	pub main_symbol: Intern<str>,
	pub int_digits: [Intern<str>; 10],
	pub self_symbol: Intern<str>,
	pub self_ty_symbol: Intern<str>,
	pub builtin_symbol: Intern<str>,
	pub builtin_symbol_bytes: Intern<[u8]>,
	pub calling_convention_symbol: Intern<str>,
}

pub static COMMON_INTERNS: LazyLock<CommonInterns> = LazyLock::new(|| CommonInterns {
	empty_str: Intern::from(""),
	empty_bytes: Intern::from(&[]),
	main_symbol: Intern::from("main"),
	int_digits: [
		Intern::from("0"),
		Intern::from("1"),
		Intern::from("2"),
		Intern::from("3"),
		Intern::from("4"),
		Intern::from("5"),
		Intern::from("6"),
		Intern::from("7"),
		Intern::from("8"),
		Intern::from("9"),
	],
	self_symbol: Intern::from("self"),
	self_ty_symbol: Intern::from("Self"),
	builtin_symbol: Intern::from("builtin"),
	builtin_symbol_bytes: Intern::from(b"builtin" as &[u8]),
	calling_convention_symbol: Intern::from("CallingConvention"),
});
