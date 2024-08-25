use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{spanned::Spanned, Generics, Path, PathArguments, Type, TypePath};

use crate::{ir_type, parse::expand_impl::ExpandImpl};

impl ToTokens for ExpandImpl {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let span = tokens.span();
        let path = type_path(&self.self_ty);
        let ty_path = &path.segments;
        let ty = path.segments.last().unwrap();
        let args = &ty.arguments;
        let mut expanded_path = ty_path.clone();
        let expanded_ty = expanded_path.last_mut().unwrap();
        expanded_ty.ident = format_ident!("{}Expand", ty.ident);
        apply_generic_names(&mut expanded_ty.arguments);
        let mut generics = self.generics.clone();
        apply_generic_params(&mut generics, &path);
        let methods = &self.expanded_fns;

        let out = quote_spanned! {span=>
            impl #generics #expanded_path {
                #(#methods)*
            }
        };
        tokens.extend(out);
    }
}

fn type_path(ty: &Type) -> Path {
    match ty {
        Type::Path(path) => path.path.clone(),
        ty => panic!("type_path: {ty:?}"),
    }
}

fn apply_generic_params(args: &mut Generics, base: &Path) {
    let expr = ir_type("Expr");
    args.params
        .push(syn::parse2(quote![__Inner: #expr<Output = #base>]).unwrap());
}

fn apply_generic_names(args: &mut PathArguments) {
    let expr = ir_type("Expr");
    match args {
        PathArguments::None => {
            *args = PathArguments::AngleBracketed(syn::parse2(quote![<__Inner>]).unwrap());
        }
        PathArguments::AngleBracketed(args) => {
            args.args.push(syn::parse2(quote![__Inner]).unwrap());
        }
        PathArguments::Parenthesized(_) => panic!(),
    }
}
