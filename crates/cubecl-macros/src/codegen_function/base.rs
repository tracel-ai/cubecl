use proc_macro2::TokenStream;
use quote::ToTokens;

use super::{expr::codegen_expr, variable::codegen_local};
use crate::tracker::VariableTracker;

/// Codegen for a statement (generally one line)
/// Entry point of code generation
pub fn codegen_statement(
    statement: &syn::Stmt,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    match statement {
        syn::Stmt::Local(local) => codegen_local(local, loop_level, variable_tracker),
        syn::Stmt::Expr(expr, semi) => {
            let expr = codegen_expr(expr, loop_level, variable_tracker).tokens;

            match semi {
                Some(_semi) => quote::quote!(
                    #expr;
                ),
                None => expr,
            }
        }
        _ => todo!("Codegen: statement {statement:?} not supported"),
    }
}

/// Codegen for a code block (a list of statements)
pub(crate) fn codegen_block(
    block: &syn::Block,
    loop_level: usize,
    variable_tracker: &mut VariableTracker,
) -> TokenStream {
    let mut statements = quote::quote!();

    for statement in block.stmts.iter() {
        statements.extend(codegen_statement(statement, loop_level, variable_tracker));
    }

    quote::quote! {
        {
            #statements
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CodegenKind {
    Comptime,
    Literal,
    Expand,
}

#[derive(Clone)]
pub(crate) struct Codegen {
    tokens: proc_macro2::TokenStream,
    array_indexing: Option<ArrayIndexing>,
    kind: CodegenKind,
}

#[derive(Clone)]
pub(crate) struct ArrayIndexing {
    pub array: proc_macro2::TokenStream,
    pub index: proc_macro2::TokenStream,
}

impl From<proc_macro2::TokenStream> for Codegen {
    fn from(tokens: proc_macro2::TokenStream) -> Self {
        Self {
            tokens,
            kind: CodegenKind::Expand,
            array_indexing: None,
        }
    }
}

impl Codegen {
    pub fn new<S: Into<proc_macro2::TokenStream>>(tokens: S, kind: CodegenKind) -> Self {
        Self {
            tokens: tokens.into(),
            kind,
            array_indexing: None,
        }
    }

    pub fn process(mut self) -> (proc_macro2::TokenStream, CodegenKind, Option<ArrayIndexing>) {
        let kind = self.kind;
        let array_indexing = self.pop_array_indexing();
        let tokens = self.tokens();

        (tokens, kind, array_indexing)
    }

    pub fn tokens(self) -> TokenStream {
        self.into_token_stream()
    }

    pub fn pop_array_indexing(&mut self) -> Option<ArrayIndexing> {
        let mut result = None;
        core::mem::swap(&mut result, &mut self.array_indexing);
        result
    }

    pub fn set_array_indexing(&mut self, array_indexing: Option<ArrayIndexing>) {
        self.array_indexing = array_indexing;
    }
}

impl ToTokens for Codegen {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let cloned = self.clone();
        let toks = cloned.into_token_stream();
        tokens.extend(toks);
    }
    fn into_token_stream(self) -> TokenStream
    where
        Self: Sized,
    {
        match self.kind {
            CodegenKind::Comptime => self.tokens,
            CodegenKind::Expand => self.tokens,
            CodegenKind::Literal => {
                let lit = self.tokens;
                quote::quote! {
                    cubecl::frontend::ExpandElementTyped::from_lit(#lit)
                }
            }
        }
    }
}
