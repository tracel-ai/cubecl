use proc_macro2::TokenStream;
use syn::{
    visit_mut::{self, VisitMut},
    Attribute, Generics, ImplItem, ImplItemFn, ItemFn, ItemImpl, Token, Type,
};

#[derive(Default)]
pub struct ExpandImplVisitor(pub Option<ExpandImpl>);

pub struct ExpandImpl {
    pub attrs: Vec<Attribute>,
    pub defaultness: Option<Token![default]>,
    pub unsafety: Option<Token![unsafe]>,
    pub generics: Generics,
    pub self_ty: Type,
    pub expanded_fns: Vec<ImplItemFn>,
}

impl VisitMut for ExpandImplVisitor {
    fn visit_impl_item_mut(&mut self, i: &mut syn::ImplItem) {
        let expanded = self.0.as_mut().unwrap();
        match i {
            syn::ImplItem::Fn(method) if method.attrs.iter().any(is_expanded) => {
                method.attrs.retain(|attr| !is_expanded(attr));
                expanded.expanded_fns.push(method.clone());
                *i = ImplItem::Verbatim(TokenStream::new())
            }
            _ => visit_mut::visit_impl_item_mut(self, i),
        }
    }

    fn visit_item_impl_mut(&mut self, i: &mut ItemImpl) {
        let expand = ExpandImpl {
            attrs: i.attrs.clone(),
            defaultness: i.defaultness.clone(),
            unsafety: i.unsafety.clone(),
            generics: i.generics.clone(),
            self_ty: *i.self_ty.clone(),
            expanded_fns: Default::default(),
        };
        self.0 = Some(expand);
        visit_mut::visit_item_impl_mut(self, i);
    }
}

fn is_expanded(attr: &Attribute) -> bool {
    attr.path()
        .get_ident()
        .map(|it| it == "expanded")
        .unwrap_or(false)
}
