use syn::{parse::Parse, ItemStruct};

pub struct Expand {
    pub strct: ItemStruct,
}

impl Parse for Expand {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let strct: ItemStruct = input.parse()?;

        Ok(Self { strct })
    }
}
