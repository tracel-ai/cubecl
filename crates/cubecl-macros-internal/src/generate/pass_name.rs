use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, ImplItem, ImplItemFn, ItemImpl, PathArguments, Type};

fn snake_case(input: &str) -> String {
    let mut snake_case_str = String::new();
    let mut chars = input.chars();
    let mut buffer = (chars.next(), chars.next());
    let mut uppercase_word = false;
    loop {
        match buffer {
            (Some(c1), Some(c2)) if c1.is_uppercase() && c2.is_lowercase() && uppercase_word => {
                snake_case_str.push('_');
                snake_case_str.push(c1.to_ascii_lowercase());
                uppercase_word = false;
            }
            (Some(c1), Some(c2)) if c1.is_lowercase() && c2.is_uppercase() => {
                snake_case_str.push(c1.to_ascii_lowercase());
                snake_case_str.push('_');
            }
            (Some(c1), Some(c2)) if c1.is_uppercase() && c2.is_uppercase() => {
                snake_case_str.push(c1.to_ascii_lowercase());
                uppercase_word = true;
            }
            (Some(c1), _) => {
                snake_case_str.push(c1.to_ascii_lowercase());
            }
            (None, None) => {
                break;
            }
            _ => {}
        }
        buffer = (buffer.1, chars.next());
    }
    snake_case_str
}

fn drop_pass_keyword(to_drop: &mut String) {
    if let Some(index) = to_drop.rfind("Pass") {
        let end = index + "Pass".len();
        to_drop.drain(index..end);
    }
}

/// Generates a `Pass::name` implementation returning the short name of the pass struct.
///
/// Applied to an `impl Pass for SomePass` block, it injects
/// `fn name(&self) -> &str { "SomePass" }`. The name is the last segment of the `Self` type, so it
/// is the bare struct name rather than a full module path. Generic type parameters are preserved
/// (with their own module paths shortened) unless they are the placeholder `T`; lifetimes are
/// dropped. For example `CombinedPass<P1, P2>` yields `"CombinedPass<P1, P2>"`, while
/// `MatchRewritePass<T>` yields `"MatchRewritePass"`.
pub fn generate_pass_name(mut item: ItemImpl) -> syn::Result<TokenStream> {
    // Reject a manual `name` method to avoid a silent duplicate definition.
    for impl_item in &item.items {
        if let ImplItem::Fn(func) = impl_item {
            if func.sig.ident == "name" {
                return Err(syn::Error::new_spanned(
                    &func.sig.ident,
                    "`pass_name` generates `name`; remove the manual `name` method",
                ));
            }
        }
    }

    let Type::Path(_) = item.self_ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "`pass_name` requires a named self type",
        ));
    };
    let mut name: String = short_type(&item.self_ty).into();
    drop_pass_keyword(&mut name);
    let name = snake_case(&name);
    let name_fn: ImplItemFn = syn::parse_quote! {
        fn name(&self) -> &str {
            #name
        }
    };
    item.items.push(ImplItem::Fn(name_fn));

    Ok(quote! { #item })
}

/// Renders a type to its short name: the last path segment, keeping its generic arguments (each
/// shortened the same way, dropping lifetimes and the placeholder `T`). Falls back to the raw
/// tokens for non-path types.
fn short_type(ty: &Type) -> String {
    let Type::Path(type_path) = ty else {
        return quote!(#ty).to_string();
    };
    let Some(segment) = type_path.path.segments.last() else {
        return quote!(#ty).to_string();
    };
    let base = segment.ident.to_string();
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return base;
    };
    let rendered = args
        .args
        .iter()
        .filter_map(short_generic_arg)
        .collect::<Vec<_>>();
    if rendered.is_empty() {
        base
    } else {
        format!("{base}<{}>", rendered.join(", "))
    }
}

/// Renders a single generic argument to its short name, or `None` if it should be omitted
/// (lifetimes, and the placeholder type `T`).
fn short_generic_arg(arg: &GenericArgument) -> Option<String> {
    match arg {
        GenericArgument::Lifetime(_) => None,
        GenericArgument::Type(ty) => {
            let name = short_type(ty);
            (name != "T").then_some(name)
        }
        other => Some(quote!(#other).to_string()),
    }
}
