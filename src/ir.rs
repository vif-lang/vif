//! Intermediate representations for the vifc compiler.
//!
//! This module contains all intermediate representations used during compilation:
//! - VUIR: unresolved IR (untyped, pre-semantic analysis)
//! - VTIR: typed IR (typed, post-semantic analysis)

pub mod vtir;
pub mod vuir;
pub mod id;
