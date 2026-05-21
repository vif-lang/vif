//! Common utilities and shared types.
//!
//! This module contains data structures and utilities used throughout the compiler.

mod alloc;
mod containers;
pub mod diagnostic;
mod interns;
mod non_max_u32;
pub mod os;
mod span;

pub use alloc::*;

pub use containers::*;
pub use interns::*;
pub use non_max_u32::*;
pub use span::*;

mod macros {
	#[macro_export]
	macro_rules! assume {
		($cond:expr $(, $($arg:tt)*)?) => {
			{
				#[cfg(debug_assertions)]
				{
					debug_assert!($cond $(, $($arg)*)?);
				}
				#[cfg(not(debug_assertions))]
				{
					unsafe { core::hint::assert_unchecked($cond) };
				}
			}
		};
	}
}
