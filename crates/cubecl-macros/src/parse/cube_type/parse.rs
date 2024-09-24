use super::*;
use darling::FromDeriveInput;

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
