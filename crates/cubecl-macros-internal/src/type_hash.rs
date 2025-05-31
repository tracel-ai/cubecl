use darling::{
    FromDeriveInput, FromField, FromVariant,
    ast::{Data, Fields},
    util::Flag,
};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, Expr, Generics, Ident, Path, Type};

#[derive(FromDeriveInput)]
#[darling(attributes(type_hash))]
pub struct TypeHash {
    pub ident: Ident,
    pub generics: Generics,
    pub data: Data<TypeHashVariant, TypeHashField>,
}

#[derive(FromVariant)]
#[darling(attributes(type_hash))]
pub struct TypeHashVariant {
    pub discriminant: Option<Expr>,
    pub fields: Fields<TypeHashField>,
}

#[derive(FromField)]
#[darling(attributes(type_hash))]
pub struct TypeHashField {
    pub ident: Option<Ident>,
    pub ty: Type,

    pub r#as: Option<Path>,
    pub skip: Flag,
    pub foreign_type: Flag,
}

pub fn type_hash_impl(input: DeriveInput) -> TokenStream {
    let type_hash = TypeHash::from_derive_input(&input).unwrap();
    match type_hash.data {
        Data::Enum(data) => type_hash_enum(&type_hash.ident, &type_hash.generics, &data),
        Data::Struct(fields) => type_hash_struct(&type_hash.ident, &type_hash.generics, &fields),
    }
}

fn type_hash_struct(
    ident: &Ident,
    generics: &Generics,
    data: &Fields<TypeHashField>,
) -> TokenStream {
    let name = ident.to_string();
    let fields = data.iter().flat_map(write_field_hash);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics crate::TypeHash for #ident #ty_generics #where_clause {
            fn write_hash(hasher: &mut impl core::hash::Hasher) {
                hasher.write(#name.as_bytes());
                #(#fields)*
            }
        }
    }
}

fn type_hash_enum(ident: &Ident, generics: &Generics, data: &[TypeHashVariant]) -> TokenStream {
    let name = ident.to_string();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let variants = data.iter().flat_map(|v| {
        v.discriminant
            .iter()
            .map(|discriminant| quote! [core::hash::Hash::hash(&(#discriminant as isize), hasher);])
            .chain(v.fields.iter().map(write_field_hash))
    });
    quote! {
        impl #impl_generics crate::TypeHash for #ident #ty_generics #where_clause{
            fn write_hash(hasher: &mut impl core::hash::Hasher) {
                hasher.write(#name.as_bytes());
                #(#variants)*
            }
        }
    }
}

fn write_field_hash(field: &TypeHashField) -> TokenStream {
    let field_name = field
        .ident
        .as_ref()
        .map(|name| {
            let name = name.to_token_stream().to_string();
            quote! { hasher.write(#name.as_bytes()); }
        })
        .unwrap_or_default();
    if let Some(ty) = &field.r#as {
        quote! {
            #field_name
            <#ty as crate::TypeHash>::write_hash(hasher);
        }
    } else if field.skip.is_present() {
        TokenStream::new()
    } else if field.foreign_type.is_present() {
        let type_str = field.ty.to_token_stream().to_string();
        quote! {
            #field_name
            hasher.write(#type_str.as_bytes());
        }
    } else {
        let field_type = &field.ty;
        quote! {
            #field_name
            <#field_type as crate::TypeHash>::write_hash(hasher);
        }
    }
}
