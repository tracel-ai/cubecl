use syn::{parse::Parse, ItemStruct};

pub struct KernelStruct {
    pub strct: ItemStruct,
}

impl Parse for KernelStruct {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let strct: ItemStruct = input.parse()?;

        Ok(Self { strct })
    }
}
