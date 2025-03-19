use crate::parse::cube_trait::{CubeTrait, CubeTraitImpl, CubeTraitImplItem, CubeTraitItem};
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::quote;

impl ToTokens for CubeTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let original_body = &self.original_trait.items;
        let colon = &self.original_trait.colon_token;
        let base_traits = &self.original_trait.supertraits;
        let attrs = &self.attrs;
        let vis = &self.vis;
        let unsafety = &self.unsafety;
        let name = &self.name;
        let generics = &self.generics;
        let fns = self.items.iter().filter_map(CubeTraitItem::func);

        let out = quote! {
            #(#attrs)*
            #[allow(clippy::too_many_arguments)]
            #vis #unsafety trait #name #generics #colon #base_traits {
                #(#original_body)*

                #(
                    #[allow(clippy::too_many_arguments)]
                    #fns;
                )*
            }
        };
        tokens.extend(out);
    }
}

impl CubeTraitImpl {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let unsafety = &self.unsafety;
        let items = &self.original_items;
        let fns = &self
            .items
            .iter_mut()
            .filter_map(CubeTraitImplItem::func)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let struct_name = &self.struct_name;
        let trait_name = &self.trait_name;
        let (generics, _, impl_where) = self.generics.split_for_impl();

        quote! {
            #unsafety impl #generics #trait_name for #struct_name #impl_where {
                #(#items)*
                #(
                    #[allow(unused, clone_on_copy, clippy::all)]
                    #fns
                )*
            }
        }
    }
}
