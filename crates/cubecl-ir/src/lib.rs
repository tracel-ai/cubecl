#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod features;

mod address;
mod properties;
mod runtime_properties;
mod scope;
mod r#type;
mod type_hash;
mod value;

pub mod arena;
pub mod attributes;
pub mod dialect;
pub mod interfaces;
pub mod types;

pub use address::*;
pub use properties::*;
pub use runtime_properties::*;
pub use scope::*;
pub use r#type::*;
pub use value::*;

pub(crate) use cubecl_macros_internal::TypeHash;
pub use type_hash::TypeHash;

pub mod pliron {
    pub use pliron::derive::*;
    pub use pliron::*;
    pub mod prelude {
        pub use alloc::{vec, vec::Vec};
        pub use pliron::derive::derive_op_interface_impl as op_interfaces;
        pub use pliron::{
            builtin::op_interfaces::*,
            common_traits::Verify,
            context::{Context, Ptr},
            derive::pliron_op,
            op::Op,
            operation::Operation,
            result::Result,
            r#type::{Type, TypeObj, Typed},
            value::Value,
        };
    }
}
