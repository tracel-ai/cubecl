use crate::{
    ir_type,
    parse::expand::{Expand, ExpandField},
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::parse_quote;

impl ToTokens for Expand {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let expand_ty = ir_type("Expand");
        let expr = ir_type("Expr");
        let expression = ir_type("Expression");
        let square_ty = ir_type("SquareType");
        let elem_ty = ir_type("Elem");
        let elem = self
            .ir_type
            .as_ref()
            .map(|ty| quote![#ty])
            .unwrap_or_else(|| quote![#elem_ty::Unit]);

        let fields = &self.fields;
        let span = self.ident.span();
        let name = &self.ident;
        let expand_name = self.name.as_ref().unwrap();
        let vis = &self.vis;
        let base_generics = &self.generics;
        let where_clause = &base_generics.where_clause;
        let base_generic_names = &self.generic_names;
        let mut expand_generics = base_generics.clone();
        let mut expand_generic_names = base_generic_names.clone();

        let inner_param = parse_quote![__Inner: #expr<Output = #name #base_generic_names>];
        expand_generics.params.push(inner_param);
        expand_generic_names.params.push(parse_quote![__Inner]);

        let expr_body = quote! {
            type Output = Self;

            fn expression_untyped(&self) -> #expression {
                panic!("Can't expand struct directly");
            }

            fn vectorization(&self) -> Option<::core::num::NonZero<u8>> {
                None
            }
        };

        let expand = quote_spanned! {span=>
            #vis struct #expand_name #expand_generics(__Inner) #where_clause;

            impl #base_generics #expand_ty for #name #base_generic_names #where_clause {
                type Expanded<__Inner: #expr<Output = Self>> = #expand_name #expand_generic_names;

                fn expand<__Inner: #expr<Output = Self>>(inner: __Inner) -> Self::Expanded<__Inner> {
                    #expand_name(inner)
                }
            }

            impl #expand_generics #expand_name #expand_generic_names #where_clause {
                #(#fields)*
            }
        };

        let out = quote_spanned! {span=>
            #expand
            impl #base_generics #expr for #name #base_generic_names #where_clause {
                #expr_body
            }
            impl #base_generics #expr for &#name #base_generic_names #where_clause {
                #expr_body
            }
            impl #base_generics #expr for &mut #name #base_generic_names #where_clause {
                #expr_body
            }
            impl #base_generics #square_ty for #name #base_generic_names #where_clause {
                fn ir_type() -> #elem_ty {
                    #elem
                }
            }
        };
        tokens.extend(out);
    }
}

impl ToTokens for ExpandField {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let func = format_ident!("__{name}");
        let ty = &self.ty;
        let vis = &self.vis;
        let access = ir_type("FieldAccess");
        tokens.extend(quote! {
            #vis fn #func(self) -> #access<#ty, __Inner> {
                #access::new(self.0, #name)
            }
        });
    }
}
