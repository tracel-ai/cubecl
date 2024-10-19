use crate::parse::cube_impl::{CubeImpl, CubeImplItem};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, Ident};

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

        let fns_tokens = quote! {
            #unsafety impl #generics #struct_name #impl_where {
                #(#items)*
                #(
                    #[allow(unused, clone_on_copy, clippy::all)]
                    #fns
                )*
            }
        };

        let methods = &self
            .items
            .iter_mut()
            .filter_map(CubeImplItem::method)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let fns_expand = &self
            .items
            .iter_mut()
            .filter_map(CubeImplItem::fn_expand)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();

        let methods_tokens = if !methods.is_empty() {
            let struct_expand_name = match self.struct_name.clone() {
                syn::Type::Path(mut pat) => {
                    let seg = pat.path.segments.first_mut().unwrap();
                    let struct_expand_name = Ident::new(
                        format!("{}Expand", seg.ident.to_string()).as_str(),
                        self.struct_name.span(),
                    );
                    seg.ident = struct_expand_name;
                    pat
                }
                _ => todo!(),
            };

            quote! {
                impl #generics #struct_expand_name #impl_where {
                    #(
                        #[allow(unused, clone_on_copy, clippy::all)]
                        #methods
                    )*
                    #(
                        #[allow(unused, clone_on_copy, clippy::all)]
                        #fns_expand
                    )*

                }
            }
        } else {
            quote! {}
        };

        quote! {
            #fns_tokens
            #methods_tokens
        }
    }
}
