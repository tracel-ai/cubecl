use std::mem::take;

use quote::{format_ident, quote, quote_spanned};
use syn::{
    Expr, ExprLoop, ExprWhile, Index, Local, LocalInit, Pat, PatIdent, PatSlice, PatStruct,
    PatTuple, PatTupleStruct, Stmt, parse_quote,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
};

pub struct Desugar;
impl VisitMut for Desugar {
    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        if let Expr::While(inner) = i {
            *i = Expr::Loop(desugar_while(inner))
        }
        visit_mut::visit_expr_mut(self, i);
    }

    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        let mut next_id = 0;
        let stmts = desugar_pats(take(&mut i.stmts), &mut next_id);

        i.stmts = stmts;
        visit_mut::visit_block_mut(self, i)
    }
}

fn desugar_pats(stmts: Vec<Stmt>, next_id: &mut usize) -> Vec<Stmt> {
    let mut output = Vec::new();
    for stmt in stmts {
        match stmt {
            Stmt::Local(Local {
                pat: Pat::Struct(pat),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_struct_destructure(pat, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat:
                    Pat::Tuple(PatTuple { elems, .. }) | Pat::TupleStruct(PatTupleStruct { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let stmts = desugar_tuple_destructure(elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            Stmt::Local(Local {
                pat: Pat::Slice(PatSlice { elems, .. }),
                init: Some(init),
                ..
            }) => {
                let elems = elems.into_iter().collect::<Vec<_>>();
                let stmts = desugar_slice_destructure(&elems, init, *next_id);
                *next_id += 1;
                output.extend(desugar_pats(stmts, next_id));
            }
            stmt => output.push(stmt),
        }
    }
    output
}

fn desugar_while(inner: &ExprWhile) -> ExprLoop {
    let cond = &inner.cond;
    let attrs = &inner.attrs;
    let label = &inner.label;
    let body = &inner.body;
    parse_quote! {
        #(#attrs)*
        #label loop {
            if !(#cond) {
                break;
            }
            #body
        }
    }
}

fn desugar_struct_destructure(pat: PatStruct, init: LocalInit, id: usize) -> Vec<Stmt> {
    let init_ident = format_ident!("__struct_destructure_init_{id}");
    let fields = pat.fields.into_iter().map(|field| {
        let attrs = field.attrs;
        let pat = field.pat;
        let member = field.member;
        quote_spanned! {pat.span()=>
            #(#attrs)* let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_tuple_destructure(
    fields: impl IntoIterator<Item = Pat>,
    init: LocalInit,
    id: usize,
) -> Vec<Stmt> {
    let init_ident = format_ident!("__tuple_destructure_init_{id}");
    let fields = fields.into_iter().enumerate().map(|(i, pat)| {
        let member = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_slice_destructure(fields: &[Pat], init: LocalInit, id: usize) -> Vec<Stmt> {
    if let Some(field) = fields.iter().find(|field| {
        matches!(
            field,
            Pat::Ident(PatIdent {
                subpat: Some(_),
                ..
            })
        )
    }) {
        let err = syn::Error::new(field.span(), "@ patterns are not currently supported")
            .to_compile_error();
        return vec![parse_quote!(#err;)];
    }

    // Slice patterns can't have more than one rest pattern, so it can always be cleanly separated
    // into before rest (which start at 0) and after rest (which start from len - n_after_rest).
    let rest_pos = fields
        .iter()
        .position(|field| matches!(field, Pat::Rest(_)))
        .unwrap_or(fields.len());
    let from_start = &fields[..rest_pos];
    let from_end = &fields[(rest_pos + 1).min(fields.len())..];
    let init_ident = format_ident!("__slice_destructure_init_{id}");
    let len_ident = format_ident!("__slice_destructure_len_{id}");

    let from_start_fields = from_start.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#offset];
        }
    });

    let init = init.expr;

    let len_expr = if from_end.is_empty() {
        quote![]
    } else {
        quote_spanned![init.span()=> let #len_ident = #init_ident.len();]
    };
    let from_end_fields = from_end.iter().enumerate().map(|(i, pat)| {
        let offset = Index::from(from_end.len() - i);
        // This requires a bit of a hack on `sub::expand` to make it work for `Sequence`
        quote_spanned! {pat.span()=>
            let #pat = #init_ident[#len_ident - #offset];
        }
    });

    let init = quote_spanned![init.span()=> let #init_ident = #init;];
    parse_quote! {
        #init
        #len_expr
        #(#from_start_fields)*
        #(#from_end_fields)*
    }
}

#[cfg(test)]
mod tests {
    use super::Desugar;
    use crate::{expression::Block, scope::Context};
    use syn::{Ident, parse_quote, visit_mut::VisitMut};

    #[test]
    fn nested_tuple_patterns_are_fully_desugared() {
        let mut block: syn::Block = parse_quote!({
            let (a, b, (c, d, (e, f))) = tuple;
        });

        Desugar.visit_block_mut(&mut block);

        let mut context = Context::new(parse_quote!(()), false, false);
        let result = Block::from_block(block, &mut context);

        assert!(result.is_ok(), "{result:?}");

        let bindings: [Ident; 6] = [
            parse_quote!(a),
            parse_quote!(b),
            parse_quote!(c),
            parse_quote!(d),
            parse_quote!(e),
            parse_quote!(f),
        ];
        for binding in bindings {
            assert!(
                context.variable(&binding).is_some(),
                "missing binding `{binding}`"
            );
        }
    }
}
