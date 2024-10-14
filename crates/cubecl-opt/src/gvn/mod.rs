//! Implements value-based partial redundancy elimination based on the algorithm described in
//! Vandrunen, Thomas & Hosking, Antony, "Value-Based Partial Redundancy Elimination"
//! Lecture Notes in Computer Science vol. 2985, pp. 167-184, March 2004
//! http://dx.doi.org/10.1007/978-3-540-24723-4_12
//!
//! ALso see https://en.wikipedia.org/wiki/Partial-redundancy_elimination

/// Analysis functions for building annotated CFG
mod analysis;
/// Apply functions that hoist expressions and eliminate redundancies
mod apply;
/// Base types
mod base;
/// Conversion functions for converting between the expression-based hashable IR and cubecl IR
mod convert;
/// Value numbering algorithm, including canonicalization of commutative expressions
mod numbering;

pub use analysis::*;
pub use base::*;
