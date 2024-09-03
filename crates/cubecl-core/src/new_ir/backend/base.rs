use crate::{new_ir::Expr, prelude::ExpandElement};

macro_rules! e {
    ($ty:path) => {
        impl Expr<Output = $ty>
    };
}

pub trait Backend {
    fn expand_binop<T>(left: e!(T), right: e!(T)) -> ExpandElement {}
}
