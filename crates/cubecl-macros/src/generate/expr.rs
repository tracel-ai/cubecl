use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

use crate::{
    parse::expr::{Expression, ExpressionArg},
    paths::{ir_type, prelude_type},
};

impl ToTokens for Expression {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let expr = ir_type("NewExpr");
        let expand_elem = prelude_type("ExpandElement");
        let vec = ir_type("Vectorization");

        let vis = &self.vis;
        let (generics, gen_names, where_clause) = self.generics.split_for_impl();
        let name = &self.name;
        let args = &self.args;
        let output = &self.output;

        let phantom_data = self
            .phantom_generics
            .as_ref()
            .map(|generics| quote![__type: #generics]);
        let vectorization = &self.vectorization;
        let item = &self.item;
        let inner_name = &item.sig.ident;
        let expand_params = self
            .args
            .iter()
            .map(|it| &it.name)
            .map(|it| quote![&self.#it]);

        tokens.extend(quote! {
            #[derive(new)]
            #vis struct #name #generics #where_clause {
                #(#args,)*
                #phantom_data
            }

            impl #generics #expr<B> for #name #gen_names #where_clause {
                type Output = #output;

                fn expand(&self, backend: &mut B) -> #expand_elem {
                    #item
                    #inner_name(#(#expand_params,)* backend)
                }

                fn vectorization(&self) -> #vec {
                    #vectorization
                }
            }
        });
    }
}

impl ToTokens for ExpressionArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let name = &self.name;
        let ty = &self.ty;
        tokens.extend(quote![pub #name: #ty])
    }
}
