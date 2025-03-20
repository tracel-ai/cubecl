use std::mem::take;

use quote::quote_spanned;
use syn::{
    Expr, ExprLoop, ExprWhile, Index, Local, LocalInit, Pat, PatStruct, PatTuple, PatTupleStruct,
    Stmt, parse_quote,
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
        let stmts = desugar_pats(take(&mut i.stmts));

        i.stmts = stmts;
        visit_mut::visit_block_mut(self, i)
    }
}

fn desugar_pats(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.into_iter().flat_map(|stmt| {
        match stmt {
            Stmt::Local(Local {
                pat: Pat::Struct(pat),
                init: Some(init),
                ..
            }) => desugar_struct_destructure(pat, init),
            Stmt::Local(Local {
                pat:
                    Pat::Tuple(PatTuple { elems, .. }) | Pat::TupleStruct(PatTupleStruct { elems, .. }),
                init: Some(init),
                ..
            }) => desugar_tuple_destructure(elems, init),
            stmt => vec![stmt],
        }
    }).collect()
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

fn desugar_struct_destructure(pat: PatStruct, init: LocalInit) -> Vec<Stmt> {
    let fields = pat.fields.into_iter().map(|field| {
        let attrs = field.attrs;
        let pat = field.pat;
        let member = field.member;
        quote_spanned! {pat.span()=>
            #(#attrs)* let #pat = __struct_destructure_init.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let __struct_destructure_init = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}

fn desugar_tuple_destructure(fields: impl IntoIterator<Item = Pat>, init: LocalInit) -> Vec<Stmt> {
    let fields = fields.into_iter().enumerate().map(|(i, pat)| {
        let member = Index::from(i);
        quote_spanned! {pat.span()=>
            let #pat = __tuple_destructure_init.#member;
        }
    });
    let init = init.expr;
    let init = quote_spanned![init.span()=> let __tuple_destructure_init = #init;];
    parse_quote! {
        #init
        #(#fields)*
    }
}
