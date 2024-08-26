use syn::{visit_mut::VisitMut, Attribute};

pub struct RemoveHelpers;

impl VisitMut for RemoveHelpers {
    fn visit_fn_arg_mut(&mut self, i: &mut syn::FnArg) {
        match i {
            syn::FnArg::Receiver(recv) => recv.attrs.retain(|it| !is_comptime_attr(it)),
            syn::FnArg::Typed(typed) => typed.attrs.retain(|it| !is_comptime_attr(it)),
        }
    }

    fn visit_expr_for_loop_mut(&mut self, i: &mut syn::ExprForLoop) {
        i.attrs.retain(|attr| !is_unroll_attr(attr))
    }
}

pub fn is_comptime_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("comptime")
}

pub fn is_unroll_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("unroll")
}

pub fn is_helper(attr: &Attribute) -> bool {
    is_comptime_attr(attr) || is_unroll_attr(attr)
}
