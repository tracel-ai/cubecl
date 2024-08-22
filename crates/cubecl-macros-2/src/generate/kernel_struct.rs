use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{
    spanned::Spanned, Field, Fields, FieldsNamed, FieldsUnnamed, GenericParam, Ident, ItemStruct,
    Type, TypeParam,
};

use crate::{ir_type, parse::kernel_struct::KernelStruct};

impl ToTokens for KernelStruct {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let span = self.strct.span();
        let mut item = self.strct.clone();
        let original = quote![#item];
        let name = item.ident.clone();

        item.fields = parse_fields(item.fields, &item.ident);
        item.ident = format_ident!("{}Expand", item.ident);
        item.generics.params.push(generic_param(&name));
        let expand = quote![#item];
        let expr = ir_type("Expr");
        let expression = ir_type("Expression");
        let kernel_struct = ir_type("KernelStruct");
        let square_ty = ir_type("SquareType");
        let elem = ir_type("Elem");
        let expand_name = &item.ident;
        let expand_init = expand_init(&item.fields, &expand_name);

        let out = quote_spanned! {span=>
            #expand
            impl #expr for #name {
                type Output = #name;

                fn expression_untyped(&self) -> #expression {
                    panic!("Can't expand struct directly");
                }
            }
            impl #square_ty for #name {
                fn ir_type() -> #elem {
                    #elem::Pointer
                }
            }
            impl<Base: #expr<Output = #name> + Clone> #expand_name<Base> {
                pub fn new(base: Base) -> Self {
                    #expand_init
                }
            }
            impl #kernel_struct for #name {
                type Expanded<Base: #expr<Output = #name> + Clone> = #expand_name<Base>;

                fn expand<Base: #expr<Output = #name> + Clone>(base: Base) -> #expand_name<Base> {
                    #expand_name::new(base)
                }
            }
        };
        tokens.extend(out);
    }
}

fn parse_fields(fields: Fields, struct_name: &Ident) -> Fields {
    match fields {
        Fields::Named(fields) => Fields::Named(parse_named_fields(fields, struct_name)),
        Fields::Unnamed(fields) => Fields::Unnamed(parse_unnamed_fields(fields, struct_name)),
        Fields::Unit => Fields::Unit,
    }
}

fn parse_named_fields(mut fields: FieldsNamed, struct_name: &Ident) -> FieldsNamed {
    for field in fields.named.iter_mut() {
        field.ty = parse_field_ty(&field.ty, struct_name);
    }
    fields
}
fn parse_unnamed_fields(mut fields: FieldsUnnamed, struct_name: &Ident) -> FieldsUnnamed {
    for field in fields.unnamed.iter_mut() {
        field.ty = parse_field_ty(&field.ty, struct_name);
    }
    fields
}

fn parse_field_ty(field: &Type, struct_name: &Ident) -> Type {
    let access = ir_type("FieldAccess");
    syn::parse2(quote![#access<#field, Base>]).unwrap()
}

fn expand_init(fields: &Fields, name: &Ident) -> TokenStream {
    match fields {
        Fields::Named(named) => expand_init_named(named, name),
        Fields::Unnamed(unnamed) => expand_init_unnamed(unnamed, name),
        Fields::Unit => quote![#name],
    }
}

fn expand_init_named(fields: &FieldsNamed, name: &Ident) -> TokenStream {
    let access = ir_type("FieldAccess");
    let fields = fields.named.iter().map(|field| {
        let name = field.ident.as_ref().unwrap();
        let var_name = name.to_string();
        quote![#name: #access::new(Box::new(base.clone()), #var_name)]
    });
    quote![#name { #(#fields),* }]
}

fn expand_init_unnamed(fields: &FieldsUnnamed, name: &Ident) -> TokenStream {
    let access = ir_type("FieldAccess");
    let fields = fields.unnamed.iter().enumerate().map(|(i, field)| {
        let var_name = i.to_string();
        quote![#access::new(Box::new(base.clone()), #var_name)]
    });
    quote![#name(#(#fields),*)]
}

fn generic_param(name: &Ident) -> GenericParam {
    let expr = ir_type("Expr");
    syn::parse2(quote![Base: #expr<Output = #name> + Clone]).unwrap()
}

// fn display_impl(item: &ItemStruct) -> TokenStream {
//     let name = &item.ident;
//     let (format_args, accessors) = display_args(&item.fields);
//     let format_string = format!("{name}{format_args}");
//     quote! {
//         impl ::core::fmt::Display for #name {
//             fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
//                 write!(f, #format_string, #accessors)
//             }
//         }
//     }
// }

// fn display_args(fields: &Fields) -> (String, TokenStream) {
//     match fields {
//         Fields::Named(named) => {
//             let args = named.named.iter().map(|field| {
//                 let field = field.ident.as_ref().unwrap();
//                 quote![#field: {}]
//             });
//             let accessors = named.named.iter().map(|field| {
//                 let field = field.ident.as_ref().unwrap();
//                 quote![self.#field]
//             });
//             let args = quote![{{ #(#args),* }}].to_string();
//             let accessors = quote![#(#accessors),*];
//             (args, accessors)
//         }
//         Fields::Unnamed(unnamed) => {
//             let args = (0..unnamed.unnamed.len()).map(|_| quote![{}]);
//             let accessors = (0..unnamed.unnamed.len()).map(|i| quote![self.#i]);
//             let args = quote![(#(#args),*)].to_string();
//             let accessors = quote![#(#accessors),*];
//             (args, accessors)
//         }
//         Fields::Unit => (String::new(), quote![]),
//     }
// }
