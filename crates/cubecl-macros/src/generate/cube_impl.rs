use crate::parse::cube_impl::{CubeImpl, CubeImplItem};
use proc_macro2::TokenStream;
use quote::quote;

impl CubeImpl {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let unsafety = &self.unsafety;
        let items = &self.original_items;
        let fns = &self
            .items
            .iter_mut()
            .filter_map(CubeImplItem::func)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let struct_name = &self.struct_name;
        let (generics, _, impl_where) = self.generics.split_for_impl();

        quote! {
            #unsafety impl #generics #struct_name #impl_where {
                #(#items)*
                #(
                    #[allow(unused, clone_on_copy, clippy::all)]
                    #fns
                )*
            }
        }
    }
}
