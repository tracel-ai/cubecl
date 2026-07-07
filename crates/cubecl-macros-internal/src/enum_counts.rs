use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident};

/// Generates a flat, named-field companion for a fieldless enum, plus variant reflection on the
/// enum itself. For an enum `Kind { A, B }` it emits:
///
/// - `impl Kind { const COUNT; const VARIANTS; const fn index(self) }`
/// - `struct KindCounts<T> { pub a: T, pub b: T }` (`#[repr(C)]`, `Pod`/`Zeroable` when `T` is),
///   with `get`/`get_mut`, `iter`, `as_slice`, and `Index`/`IndexMut<Kind>`.
///
/// Field order matches declaration order, so `KindCounts::as_slice()[k.index()]` is the entry for
/// `k`. Adding a variant automatically extends both the reflection and the companion struct.
pub fn enum_counts_impl(input: DeriveInput) -> syn::Result<TokenStream> {
    let enum_ident = &input.ident;
    let vis = &input.vis;

    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            enum_ident,
            "EnumCounts can only be derived on enums",
        ));
    };

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "EnumCounts does not support generic enums",
        ));
    }

    let mut variants = Vec::new();
    let mut fields = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                &variant.ident,
                "EnumCounts requires fieldless (unit) variants",
            ));
        }
        let ident = &variant.ident;
        fields.push(Ident::new(&ident.to_string().to_lowercase(), ident.span()));
        variants.push(ident.clone());
    }

    let count = variants.len();
    let counts_ident = format_ident!("{}Counts", enum_ident);

    let index_arms = variants
        .iter()
        .enumerate()
        .map(|(i, v)| quote! { #enum_ident::#v => #i });
    let get_arms = variants
        .iter()
        .zip(&fields)
        .map(|(v, f)| quote! { #enum_ident::#v => &self.#f });
    let get_mut_arms = variants
        .iter()
        .zip(&fields)
        .map(|(v, f)| quote! { #enum_ident::#v => &mut self.#f });
    let entries = variants
        .iter()
        .zip(&fields)
        .map(|(v, f)| quote! { (#enum_ident::#v, &self.#f) });
    let variants_list = variants.iter().map(|v| quote! { #enum_ident::#v });
    let struct_fields = fields.iter().map(|f| quote! { pub #f: T });

    Ok(quote! {
        impl #enum_ident {
            /// Number of variants.
            pub const COUNT: usize = #count;
            /// All variants, in declaration order.
            pub const VARIANTS: [#enum_ident; #count] = [ #(#variants_list),* ];

            /// Stable index of this variant (declaration order, `0..COUNT`).
            pub const fn index(self) -> usize {
                match self { #(#index_arms),* }
            }
        }

        #[doc = concat!("Per-variant values keyed by [`", stringify!(#enum_ident), "`].")]
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        #vis struct #counts_ident<T> {
            #(#struct_fields),*
        }

        // Safe: `#[repr(C)]` with `COUNT` fields all of type `T` (no padding), so the layout is
        // exactly `[T; COUNT]`.
        unsafe impl<T: ::bytemuck::Zeroable> ::bytemuck::Zeroable for #counts_ident<T> {}
        unsafe impl<T: ::bytemuck::Pod> ::bytemuck::Pod for #counts_ident<T> {}

        impl<T> #counts_ident<T> {
            /// Number of entries (equals `COUNT` of the key enum).
            pub const LEN: usize = #count;

            /// The entry for `key`.
            pub fn get(&self, key: #enum_ident) -> &T {
                match key { #(#get_arms),* }
            }

            /// The mutable entry for `key`.
            pub fn get_mut(&mut self, key: #enum_ident) -> &mut T {
                match key { #(#get_mut_arms),* }
            }

            /// Iterate `(variant, &value)` in declaration order.
            pub fn iter(&self) -> impl Iterator<Item = (#enum_ident, &T)> {
                [ #(#entries),* ].into_iter()
            }

            /// Contiguous view in declaration order.
            pub fn as_slice(&self) -> &[T] {
                unsafe { ::core::slice::from_raw_parts((self as *const Self).cast::<T>(), Self::LEN) }
            }

            /// Mutable contiguous view in declaration order.
            pub fn as_mut_slice(&mut self) -> &mut [T] {
                unsafe {
                    ::core::slice::from_raw_parts_mut((self as *mut Self).cast::<T>(), Self::LEN)
                }
            }
        }

        impl<T> ::core::ops::Index<#enum_ident> for #counts_ident<T> {
            type Output = T;
            fn index(&self, key: #enum_ident) -> &T {
                self.get(key)
            }
        }

        impl<T> ::core::ops::IndexMut<#enum_ident> for #counts_ident<T> {
            fn index_mut(&mut self, key: #enum_ident) -> &mut T {
                self.get_mut(key)
            }
        }
    })
}
