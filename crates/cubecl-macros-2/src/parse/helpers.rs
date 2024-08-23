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
    attr.path()
        .get_ident()
        .map(ToString::to_string)
        .map(|it| it == "comptime")
        .unwrap_or(false)
}

pub fn is_unroll_attr(attr: &Attribute) -> bool {
    attr.path()
        .get_ident()
        .map(ToString::to_string)
        .map(|it| it == "unroll")
        .unwrap_or(false)
}
