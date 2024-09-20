use std::str::FromStr;

use darling::{ast::Data, util::PathList, FromDeriveInput, FromField, FromMeta};
use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse_quote, punctuated::Punctuated, Block, Field, GenericParam, Generics, Ident, ItemFn, Path,
    ReturnType, Type, Visibility,
};

#[derive(FromDeriveInput)]
#[darling(supports(struct_any))]
pub struct AutotuneKey {
    pub ident: Ident,
    pub vis: Visibility,
    pub generics: Generics,
    pub data: Data<(), AutotuneKeyField>,
}

impl AutotuneKey {
    pub fn is_tuple(&self) -> bool {
        self.data.as_ref().take_struct().unwrap().is_tuple()
    }
}

#[derive(FromField)]
#[darling(attributes(autotune))]
pub struct AutotuneKeyField {
    pub ident: Option<Ident>,
    pub ty: Type,
    pub anchor: Option<Anchor>,
    pub name: Option<String>,
}

#[derive(FromMeta)]
pub enum Anchor {
    #[darling(word)]
    Unlimited,
    Max(usize),
}

impl Anchor {
    pub fn max(&self) -> TokenStream {
        match self {
            Anchor::Unlimited => quote![None],
            Anchor::Max(value) => quote![Some(#value)],
        }
    }
}

#[derive(FromMeta)]
pub struct AutotuneOperationsArgs {
    name: Option<Ident>,
    key: Option<Ident>,
    create_key: Option<Path>,
    should_run: Option<Path>,
    operations: PathList,
}

pub struct AutotuneOperations {
    pub name: Ident,
    pub generics: Generics,
    pub key: Ident,
    pub key_ty: Type,
    pub output: Type,
    pub input_fields: Vec<Field>,
    pub ty: Option<TokenStream>,
    pub tunables_fn: Block,
    pub create_key: Option<Path>,
    pub should_run: Option<Path>,
    pub operations: Vec<Path>,
}

pub fn operation_name(op: &Path) -> Ident {
    let name = op.segments.last().unwrap();
    let name = RenameRule::PascalCase.apply_to_field(name.ident.to_string());
    format_ident!("{name}")
}

impl AutotuneOperations {
    pub fn from_item_fn(item: ItemFn, args: AutotuneOperationsArgs) -> syn::Result<Self> {
        let name = args.name.unwrap_or_else(|| {
            let name = RenameRule::PascalCase.apply_to_field(item.sig.ident.to_string());
            format_ident!("{name}")
        });
        let key = args.key.unwrap_or_else(|| format_ident!("key"));
        let generics = item.sig.generics.clone();
        let fields = item.sig.inputs.iter().map(|input| parse_quote!(#input));
        let ty = (!generics.params.is_empty()).then(|| {
            let names = generics.params.iter().map(|it| match it {
                GenericParam::Lifetime(lifetime_param) => {
                    let mut param = lifetime_param.clone();
                    param.bounds = Punctuated::new();
                    param.colon_token = None;
                    param.to_token_stream()
                }
                GenericParam::Type(type_param) => type_param.ident.to_token_stream(),
                GenericParam::Const(const_param) => const_param.ident.to_token_stream(),
            });
            quote![__ty: ::core::marker::PhantomData<(#(#names),*)>,]
        });
        let output = match item.sig.output {
            ReturnType::Default => parse_quote![()],
            ReturnType::Type(_, ty) => *ty,
        };

        let operations = args
            .operations
            .to_strings()
            .into_iter()
            .map(|s| syn::parse2(TokenStream::from_str(&s).unwrap()).unwrap())
            .collect();

        let input_fields: Vec<Field> = fields.collect();
        let key_ty = input_fields
            .iter()
            .find(|field| field.ident.as_ref().unwrap() == &key)
            .ok_or_else(|| syn::Error::new_spanned(item.sig.inputs, "Missing key from inputs"))?
            .ty
            .clone();

        Ok(Self {
            name,
            generics,
            input_fields,
            ty,
            key_ty,
            tunables_fn: *item.block,
            key,
            create_key: args.create_key,
            should_run: args.should_run,
            output,
            operations,
        })
    }
}
