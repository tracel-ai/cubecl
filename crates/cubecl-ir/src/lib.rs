#![no_std]

pub extern crate alloc;
extern crate std;

pub mod features;

mod address;
mod properties;
mod runtime_properties;
mod scope;
mod r#type;
mod type_hash;
mod value;

#[cfg(feature = "spirv")]
mod spirv;

pub mod arena;
pub mod attributes;
pub mod dialect;
pub mod interfaces;
pub mod metadata;
pub mod rewrite;
pub mod settings;
pub mod types;

pub use address::*;
pub use properties::*;
pub use runtime_properties::*;
pub use scope::*;
pub use r#type::*;
pub use value::*;

pub(crate) use cubecl_macros_internal::TypeHash;
pub use type_hash::TypeHash;

pub mod prelude {
    pub use crate::{
        ContextExt, dialect::OperationPtrExt, rewrite::*, scope::FuncOpExt, verify_op_succ,
        verify_ty_succ,
    };
    pub use alloc::{vec, vec::Vec};
    pub use cubecl_macros_internal::{cube_op, op_traits, pass_name};
    pub use pliron::derive::derive_op_interface_impl as op_interfaces;
    pub use pliron::{
        attribute::AttributeDict,
        builtin::{attributes::TypeAttr, op_interfaces::*, type_interfaces::*},
        common_traits::Verify,
        context::{Context, Ptr},
        derive::*,
        graph::walkers::{
            IRNode, WALKCONFIG_POSTORDER_FORWARD, WALKCONFIG_POSTORDER_REVERSE,
            WALKCONFIG_PREORDER_FORWARD, WALKCONFIG_PREORDER_REVERSE,
        },
        irbuild::{
            IRStatus,
            dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
            inserter::Inserter,
            listener::Recorder,
            match_rewrite::{MatchRewrite, MatchRewriter},
            rewriter::{IRRewriter, Rewriter},
        },
        location::Located,
        op::{Op, op_cast, op_impls},
        operation::Operation,
        pass::*,
        result::Result,
        r#type::{Type, TypeHandle, Typed, TypedHandle, type_cast, type_impls},
        value::Value,
    };
    pub type PassRewriter = pliron::irbuild::rewriter::IRRewriter<Recorder>;
}

pub use cubecl_macros_internal::cube_op;

pub mod pliron {
    pub use pliron::derive::*;
    pub use pliron::*;
}
