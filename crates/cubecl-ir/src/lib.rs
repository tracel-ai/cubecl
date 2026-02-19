#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod features;

mod address;
mod allocator;
mod arithmetic;
mod atomic;
mod barrier;
mod bitwise;
mod branch;
mod cmma;
mod comparison;
mod marker;
mod metadata;
mod non_semantic;
mod operation;
mod operator;
mod plane;
mod processing;
mod properties;
mod reflect;
mod runtime_properties;
mod scope;
mod synchronization;
mod tma;
mod r#type;
mod type_hash;
mod variable;

pub use address::*;
pub use allocator::*;
pub use arithmetic::*;
pub use atomic::*;
pub use barrier::*;
pub use bitwise::*;
pub use branch::*;
pub use cmma::*;
pub use comparison::*;
pub use marker::*;
pub use metadata::*;
pub use non_semantic::*;
pub use operation::*;
pub use operator::*;
pub use plane::*;
pub use processing::*;
pub use properties::*;
pub use reflect::*;
pub use runtime_properties::*;
pub use scope::*;
pub use synchronization::*;
pub use tma::*;
pub use r#type::*;
pub use variable::*;

pub(crate) use cubecl_macros_internal::{OperationArgs, OperationCode, OperationReflect, TypeHash};
pub use type_hash::TypeHash;
