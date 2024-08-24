use quote::format_ident;
use syn::{Attribute, FnArg, ItemFn, Meta, PatType, Receiver};

pub mod expand_impl;
pub mod expression;
pub mod field_expand;
pub mod kernel;
pub mod statement;
