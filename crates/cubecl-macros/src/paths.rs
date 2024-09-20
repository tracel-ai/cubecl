use quote::format_ident;
use std::cell::LazyCell;
use syn::Path;

#[allow(clippy::declare_interior_mutable_const)]
const CORE_PATH: LazyCell<Path> = LazyCell::new(|| Path::from(format_ident!("cubecl")));
#[allow(clippy::declare_interior_mutable_const)]
const FRONTEND_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("frontend").into());
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const PRELUDE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("prelude").into());
    path
});
#[allow(clippy::declare_interior_mutable_const)]
const TUNE_PATH: LazyCell<Path> = LazyCell::new(|| {
    let mut path = core_path();
    path.segments.push(format_ident!("tune").into());
    path
});

pub fn frontend_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    FRONTEND_PATH.clone()
}

pub fn prelude_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    PRELUDE_PATH.clone()
}

pub fn tune_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    TUNE_PATH.clone()
}

pub fn core_path() -> Path {
    #[allow(clippy::borrow_interior_mutable_const)]
    CORE_PATH.clone()
}

pub fn core_type(ty: &str) -> Path {
    let mut path = core_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}

pub fn frontend_type(ty: &str) -> Path {
    let mut path = frontend_path();
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

pub fn tune_type(ty: &str) -> Path {
    let mut path = tune_path();
    let ident = format_ident!("{ty}");
    path.segments.push(ident.into());
    path
}
