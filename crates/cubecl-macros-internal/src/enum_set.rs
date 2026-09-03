use darling::{
    FromDeriveInput, FromVariant,
    ast::{Data, Fields, Style},
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Error, Expr, Generics, Ident};

#[derive(FromDeriveInput)]
pub struct EnumSetTypeInput {
    pub ident: Ident,
    pub generics: Generics,
    pub data: Data<EnumSetVariant, darling::util::Ignored>,
}

#[derive(FromVariant)]
pub struct EnumSetVariant {
    pub ident: Ident,
    pub discriminant: Option<Expr>,
    pub fields: Fields<darling::util::Ignored>,
}

/// Backing store for an `EnumSet`, which fixes the variant ceiling.
const MAX_VARIANTS: usize = 64;

pub fn enum_set_type_impl(input: DeriveInput) -> Result<TokenStream, Error> {
    let parsed = EnumSetTypeInput::from_derive_input(&input).map_err(Error::from)?;
    let ident = &parsed.ident;

    let Data::Enum(variants) = &parsed.data else {
        return Err(Error::new_spanned(
            &input.ident,
            "EnumSetType can only be derived for enums",
        ));
    };

    if !parsed.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &parsed.generics,
            "EnumSetType cannot be derived for generic enums",
        ));
    }

    if variants.is_empty() {
        return Err(Error::new_spanned(
            &input.ident,
            "EnumSetType cannot be derived for an enum with no variants",
        ));
    }

    if variants.len() > MAX_VARIANTS {
        return Err(Error::new_spanned(
            &input.ident,
            format!("EnumSetType supports at most {MAX_VARIANTS} variants, an `EnumSet` is a u64"),
        ));
    }

    for variant in variants {
        if !matches!(variant.fields.style, Style::Unit) {
            return Err(Error::new_spanned(
                &variant.ident,
                "EnumSetType requires every variant to be a unit variant",
            ));
        }
        // A discriminant would make the variant's value and its bit index disagree, and nothing in
        // the tree needs both.
        if variant.discriminant.is_some() {
            return Err(Error::new_spanned(
                &variant.ident,
                "EnumSetType does not support explicit discriminants",
            ));
        }
    }

    let count = variants.len() as u32;
    let to_bit = variants.iter().enumerate().map(|(bit, variant)| {
        let name = &variant.ident;
        let bit = bit as u32;
        quote! { #ident::#name => #bit }
    });
    let from_bit = variants.iter().enumerate().map(|(bit, variant)| {
        let name = &variant.ident;
        let bit = bit as u32;
        quote! { #bit => #ident::#name }
    });

    Ok(quote! {
        impl ::cubecl_ir::EnumSetType for #ident {
            const VARIANTS: u32 = #count;

            fn to_bit(self) -> u32 {
                match self {
                    #(#to_bit,)*
                }
            }

            fn from_bit(bit: u32) -> Self {
                match bit {
                    #(#from_bit,)*
                    _ => panic!(
                        concat!("bit index out of range for `", stringify!(#ident), "`"),
                    ),
                }
            }
        }

        impl ::core::ops::BitOr<#ident> for #ident {
            type Output = ::cubecl_ir::EnumSet<#ident>;

            fn bitor(self, other: #ident) -> ::cubecl_ir::EnumSet<#ident> {
                ::cubecl_ir::EnumSet::only(self) | ::cubecl_ir::EnumSet::only(other)
            }
        }

        impl ::core::ops::BitOr<::cubecl_ir::EnumSet<#ident>> for #ident {
            type Output = ::cubecl_ir::EnumSet<#ident>;

            fn bitor(self, other: ::cubecl_ir::EnumSet<#ident>) -> ::cubecl_ir::EnumSet<#ident> {
                ::cubecl_ir::EnumSet::only(self) | other
            }
        }

        impl ::core::ops::Not for #ident {
            type Output = ::cubecl_ir::EnumSet<#ident>;

            fn not(self) -> ::cubecl_ir::EnumSet<#ident> {
                !::cubecl_ir::EnumSet::only(self)
            }
        }
    })
}
