use syn::{parse::Parse, ItemStruct};

pub struct FieldExpand {
    pub strct: ItemStruct,
}

impl Parse for FieldExpand {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let strct: ItemStruct = input.parse()?;

        Ok(Self { strct })
    }
}
