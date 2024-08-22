use quote::format_ident;
use syn::{Attribute, FnArg, ItemFn, Meta, PatType, Receiver};

pub mod expression;
pub mod kernel;
pub mod kernel_struct;
pub mod statement;

pub fn strip_comptime(func: &mut ItemFn) {
    let not_comptime = |attr: &Attribute| !matches!(&attr.meta, Meta::Path(path) if path.is_ident(&format_ident!("comptime")));

    for input in func.sig.inputs.iter_mut() {
        match input {
            FnArg::Typed(PatType { attrs, .. }) => attrs.retain(not_comptime),
            FnArg::Receiver(Receiver { attrs, .. }) => attrs.retain(not_comptime),
        };
    }
}
