use crate::{
    parse::cube_trait::{CubeTrait, CubeTraitImpl, CubeTraitImplItem, CubeTraitItem},
    paths::ir_type,
};
use proc_macro2::TokenStream;
use quote::quote;
use quote::ToTokens;

impl ToTokens for CubeTrait {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let static_expanded = ir_type("StaticExpanded");

        let original = &self.original_trait;
        let attrs = &self.attrs;
        let vis = &self.vis;
        let unsafety = &self.unsafety;
        let expand_name = &self.expand_name;
        let generics = &self.generics;
        let fns = &self.items;

        let out = quote! {
            #original

            #(#attrs)*
            #vis #unsafety trait #expand_name #generics: #static_expanded {
                #(#fns)*
            }
        };
        tokens.extend(out);
    }
}

impl ToTokens for CubeTraitItem {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let out = match self {
            CubeTraitItem::Fn(func) => quote![#func;],
            CubeTraitItem::Other(tokens) => tokens.clone(),
        };
        tokens.extend(out);
    }
}

impl ToTokens for CubeTraitImplItem {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let out = match self {
            CubeTraitImplItem::Fn(func) => quote![#func],
            CubeTraitImplItem::Other(tokens) => tokens.clone(),
        };
        tokens.extend(out);
    }
}

impl ToTokens for CubeTraitImpl {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        //let static_expand = ir_type("StaticExpand");

        let unsafety = &self.unsafety;
        let fns = &self.items;
        //let struct_name = &self.struct_name;
        let struct_expand_name = &self.struct_expand_name;
        let trait_expand_name = &self.trait_expand_name;
        let (generics, _, impl_where) = self.generics.split_for_impl();
        let (_, struct_generic_names, _) = self.struct_generics.split_for_impl();

        let out = quote! {
            #unsafety impl #generics #trait_expand_name for #struct_expand_name #struct_generic_names #impl_where {
                #(
                    #[allow(unused, clone_on_copy, clippy::all)]
                    #fns
                )*
            }
        };
        tokens.extend(out);
    }
}
