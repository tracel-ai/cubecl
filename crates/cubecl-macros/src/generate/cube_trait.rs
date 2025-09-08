use crate::parse::cube_trait::{CubeTrait, CubeTraitImpl, CubeTraitImplItem, CubeTraitItem};
use proc_macro2::TokenStream;
use quote::quote;
use quote::{ToTokens, format_ident};
use syn::{Type, parse_quote, spanned::Spanned};

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
        let assoc_fns = self.items.iter().filter_map(CubeTraitItem::func);

        let has_expand = self
            .items
            .iter()
            .any(|it| matches!(it, CubeTraitItem::Method(_)));

        let out = quote! {
            #(#attrs)*
            #[allow(clippy::too_many_arguments)]
            #vis #unsafety trait #name #generics #colon #base_traits {
                #(#original_body)*

                #(
                    #[allow(clippy::too_many_arguments)]
                    #assoc_fns;
                )*
            }
        };
        tokens.extend(out);

        if has_expand {
            tokens.extend(self.generate_expand());
        }
    }
}

impl CubeTrait {
    fn generate_expand(&self) -> TokenStream {
        let attrs = &self.attrs;
        let vis = &self.vis;
        let unsafety = &self.unsafety;
        let name = format_ident!("{}Expand", self.name);
        let generics = &self.generics;
        let others = self.items.iter().filter_map(CubeTraitItem::other);
        let methods = self.items.iter().filter_map(CubeTraitItem::method);
        let mut supertraits = self.expand_supertraits.clone();
        supertraits.push(parse_quote!(Clone));

        quote! {
            #(#attrs)*
            #[allow(clippy::too_many_arguments)]
            #vis #unsafety trait #name #generics: #supertraits {
                #(#others)*

                #(
                    #[allow(clippy::too_many_arguments)]
                    #methods;
                )*
            }
        }
    }
}

impl CubeTraitImpl {
    pub fn to_tokens_mut(&mut self) -> TokenStream {
        let has_expand = self
            .items
            .iter()
            .any(|it| matches!(it, CubeTraitImplItem::Method(_)));

        let expand = if has_expand {
            self.generate_expand()
        } else {
            quote![]
        };

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
            #expand
        }
    }
}

impl CubeTraitImpl {
    fn generate_expand(&mut self) -> TokenStream {
        let others = self
            .items
            .iter()
            .filter_map(CubeTraitImplItem::other)
            .cloned()
            .collect::<Vec<_>>();
        let methods = self
            .items
            .iter_mut()
            .filter_map(CubeTraitImplItem::method)
            .map(|it| it.to_tokens_mut())
            .collect::<Vec<_>>();
        let unsafety = &self.unsafety;

        let mut struct_name = match &self.struct_name {
            Type::Path(path) => path.clone(),
            other => {
                return syn::Error::new(other.span(), "Struct name must be a path")
                    .to_compile_error();
            }
        };
        let struct_ident = struct_name.path.segments.last_mut().unwrap();
        struct_ident.ident = format_ident!("{}Expand", struct_ident.ident);

        let mut trait_name = self.trait_name.clone();
        let trait_ident = trait_name.segments.last_mut().unwrap();
        trait_ident.ident = format_ident!("{}Expand", trait_ident.ident);

        let (generics, _, impl_where) = self.generics.split_for_impl();

        quote! {
            #unsafety impl #generics #trait_name for #struct_name #impl_where {
                #(#others)*
                #(
                    #[allow(unused, clone_on_copy, clippy::all)]
                    #methods
                )*
            }
        }
    }
}
