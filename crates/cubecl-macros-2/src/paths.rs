use proc_macro2::Span;
use quote::format_ident;
use std::cell::LazyCell;
use syn::{Ident, Path, Token};

#[allow(clippy::declare_interior_mutable_const)]
const CORE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let span = Span::call_site();
    let mut path = Path::from(format_ident!("cubecl"));
    //path.leading_colon = Some(Token![::](span));
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const IR_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("new_ir").into());
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const PRELUDE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("prelude").into());
    path
});

pub fn ir_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    IR_PATH.clone()
}

pub fn prelude_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    PRELUDE_PATH.clone()
}

pub fn core_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    CORE_PATH.clone()
}

pub fn prefix_ir(ident: Ident) -> Path {
    let mut path = ir_path();
    path.segments.push(ident.into());
    path
}

pub fn core_type(ty: &str) -> Path {
    let mut path = core_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn ir_type(ty: &str) -> Path {
    let mut path = ir_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn prelude_type(ty: &str) -> Path {
    let mut path = prelude_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}
