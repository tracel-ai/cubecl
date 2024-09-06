use quote::{format_ident, quote_spanned, ToTokens};
use syn::{parse_quote, spanned::Spanned, Generics, Path, PathArguments, Type};

use crate::{parse::expand_impl::ExpandImpl, paths::frontend_type};

impl ToTokens for ExpandImpl {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let span = tokens.span();
        let path = type_path(&self.self_ty);
        let ty_path = &path.segments;
        let ty = path.segments.last().unwrap();
        let mut expanded_path = ty_path.clone();
        let expanded_ty = expanded_path.last_mut().unwrap();
        expanded_ty.ident = format_ident!("{}Expand", ty.ident);
        apply_generic_names(&mut expanded_ty.arguments);
        let mut generics = self.generics.clone();
        apply_generic_params(&mut generics, &path);
        let methods = &self.expanded_fns;
        let attrs = &self.attrs;
        let defaultness = &self.defaultness;
        let unsafety = &self.unsafety;
        let where_clause = &self.generics.where_clause;

        let out = quote_spanned! {span=>
            #[allow(clippy::new_ret_no_self)]
            #(#attrs)*
            #defaultness #unsafety impl #generics #expanded_path #where_clause {
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
    let expr = frontend_type("Expr");
    args.params
        .push(parse_quote![__Inner: #expr<Output = #base>]);
}

fn apply_generic_names(args: &mut PathArguments) {
    match args {
        PathArguments::None => {
            *args = PathArguments::AngleBracketed(parse_quote![<__Inner>]);
        }
        PathArguments::AngleBracketed(args) => {
            args.args.push(parse_quote![__Inner]);
        }
        PathArguments::Parenthesized(_) => panic!(),
    }
}
