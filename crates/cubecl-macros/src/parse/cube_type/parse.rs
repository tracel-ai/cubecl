use super::*;
use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use syn::DeriveInput;

#[derive(Debug)]
pub(crate) enum CubeType {
    Enum(CubeTypeEnum),
    Struct(CubeTypeStruct),
}

impl FromDeriveInput for CubeType {
    fn from_derive_input(input: &syn::DeriveInput) -> darling::Result<Self> {
        match &input.data {
            syn::Data::Struct(_) => Ok(Self::Struct(CubeTypeStruct::from_derive_input(input)?)),
            syn::Data::Enum(_) => Ok(Self::Enum(CubeTypeEnum::from_derive_input(input)?)),
            syn::Data::Union(_) => Err(darling::Error::custom("Union not supported")),
        }
    }
}

pub fn generate_cube_type(input: &DeriveInput, with_launch: bool) -> syn::Result<TokenStream> {
    let cube_type = CubeType::from_derive_input(input)?;

    Ok(cube_type.generate(with_launch))
}

pub fn generate_cube_harness(input: &DeriveInput) -> syn::Result<TokenStream> {
    match CubeType::from_derive_input(input)? {
        CubeType::Struct(data) => data.generate_harness(),
        CubeType::Enum(_) => Err(syn::Error::new_spanned(
            &input.ident,
            "`CubeHarness` only supports structs",
        )),
    }
}
