use darling::{FromMeta, util::Flag};
use proc_macro2::TokenStream;
use syn::{DeriveInput, Meta, parse_quote, parse2};

use crate::{generate_cube_type, generate_cube_type_mut, generate_into_runtime};

#[derive(FromMeta)]
#[darling(rename_all = "PascalCase")]
pub struct DeriveExpand {
    cube_type: Flag,
    cube_type_mut: Flag,
    cube_launch: Flag,
    into_runtime: Flag,
}

pub fn generate_derive_expand(input: TokenStream, meta: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = parse2(input)?;
    let meta: Meta = parse_quote!(derive_expand(#meta));
    let derives = DeriveExpand::from_meta(&meta)?;

    let mut out = TokenStream::new();

    if derives.cube_type.is_present() {
        out.extend(generate_cube_type(&input, false)?);
    }
    if derives.cube_type_mut.is_present() {
        out.extend(generate_cube_type_mut(&input)?);
    }
    if derives.cube_launch.is_present() {
        out.extend(generate_cube_type(&input, true)?);
    }
    if derives.into_runtime.is_present() {
        out.extend(generate_into_runtime(&input)?)
    }

    Ok(out)
}
