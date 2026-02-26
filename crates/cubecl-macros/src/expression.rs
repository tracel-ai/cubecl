use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use syn::{
    AngleBracketedGenericArguments, Ident, Lit, LitStr, Member, Pat, Path, PathArguments,
    PathSegment, QSelf, Type,
};

use crate::{
    operator::Operator,
    scope::{Context, ManagedVar, Scope},
    statement::Statement,
};

#[derive(Clone, Debug)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
        span: Span,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        span: Span,
    },
    Variable(ManagedVar),
    FieldAccess {
        base: Box<Expression>,
        field: Member,
    },
    Path {
        path: Path,
        qself: Option<QSelf>,
    },
    Literal {
        value: Lit,
    },
    Assignment {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Block(Block),
    FunctionCall {
        func: Box<Expression>,
        args: Vec<Expression>,
        associated_type: Option<(Path, Option<QSelf>, PathSegment)>,
        span: Span,
    },
    CompilerIntrinsic {
        func: Path,
        args: Vec<Expression>,
    },
    MethodCall {
        receiver: Box<Expression>,
        method: Ident,
        generics: Option<AngleBracketedGenericArguments>,
        args: Vec<Expression>,
        span: Span,
    },
    Closure {
        params: Vec<Pat>,
        body: Box<Expression>,
        scope: Scope,
    },
    Cast {
        from: Box<Expression>,
        to: Type,
    },
    Break,
    /// Tokens not relevant to parsing
    Verbatim {
        tokens: TokenStream,
    },
    VerbatimTerminated {
        tokens: TokenStream,
    },
    #[allow(clippy::enum_variant_names)]
    ExpressionMacro {
        ident: Ident,
        args: Vec<Expression>,
    },
    Continue(Span),
    Return(Span),
    ForLoop {
        range: Box<Expression>,
        unroll: Option<Box<Expression>>,
        var_name: syn::Ident,
        var_ty: Option<syn::Type>,
        block: Block,
        scope: Scope,
    },
    Loop {
        block: Block,
        scope: Scope,
    },
    If {
        condition: Box<Expression>,
        then_block: Block,
        else_branch: Option<Box<Expression>>,
    },
    Switch {
        value: Box<Expression>,
        cases: Vec<(Expression, Block)>,
        default: Block,
    },
    Range {
        start: Box<Expression>,
        end: Option<Box<Expression>>,
        span: Span,
        inclusive: bool,
    },
    Array {
        elements: Vec<Expression>,
        span: Span,
    },
    Tuple {
        elements: Vec<Expression>,
    },
    Index {
        expr: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    Slice {
        span: Span,
        _ranges: Vec<Expression>,
    },
    ArrayInit {
        init: Box<Expression>,
        len: Box<Expression>,
    },
    Reference {
        inner: Box<Expression>,
    },
    StructInit {
        path: Path,
        fields: Vec<(Member, Expression)>,
    },
    Keyword {
        name: Ident,
    },
    Match {
        // True implies that discriminants are matched at comptime,
        // but the values of the variants are only known at runtime.
        // False implies that both the discriminants and the variant's values are known at
        // comptime.
        runtime_variants: bool,

        expr: Box<Expression>,
        arms: Vec<MatchArm>,
    },
    IfLet {
        runtime_variants: bool,
        expr: Box<Expression>,
        arm: MatchArm,
        else_branch: Option<Box<Expression>>,
    },
    Comment {
        content: LitStr,
    },
    RustMacro {
        ident: Ident,
        tokens: TokenStream,
    },
    Terminate,
    AssertConstant {
        inner: Box<Expression>,
    },
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pat: Pat,
    pub expr: Box<Expression>,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub inner: Vec<Statement>,
    pub ret: Option<Box<Expression>>,
}

impl Expression {
    pub fn is_const(&self) -> bool {
        match self {
            Expression::Literal { .. } => true,
            Expression::Path { .. } => true,
            Expression::Verbatim { .. } => true,
            Expression::VerbatimTerminated { .. } => true,
            Expression::Variable(var) => var.is_const,
            Expression::FieldAccess { base, .. } => base.is_const(),
            Expression::Reference { inner } => inner.is_const(),
            Expression::Array { elements, .. } => elements.iter().all(|it| it.is_const()),
            Expression::Tuple { elements, .. } => elements.iter().all(|it| it.is_const()),
            Expression::CompilerIntrinsic { .. } => true,
            Expression::Match { arms, .. } => arms.iter().all(|it| it.expr.is_const()),
            Expression::AssertConstant { .. } => true,
            _ => false,
        }
    }

    pub fn as_const_primitive(&self, _context: &mut Context) -> Option<TokenStream> {
        match self {
            Expression::Literal { value, .. } => match value {
                Lit::Int(_) | Lit::Float(_) | Lit::Bool(_) => Some(quote![#value]),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_const(&self, context: &mut Context) -> Option<TokenStream> {
        match self {
            Expression::Literal { value, .. } => Some(quote![#value]),
            Expression::Verbatim { tokens, .. } => Some(tokens.clone()),
            Expression::VerbatimTerminated { tokens, .. } => Some(tokens.clone()),
            Expression::Variable(ManagedVar {
                name,
                is_const: true,
                ..
            }) => Some(quote![#name.clone()]),
            Expression::Path { path, .. } => Some(quote![#path]),
            Expression::Array { elements, .. } => {
                let elements = elements
                    .iter()
                    .map(|it| it.as_const(context))
                    .collect::<Option<Vec<_>>>()?;
                Some(quote![[#(#elements),*]])
            }
            Expression::Tuple { elements, .. } => {
                let elements = elements
                    .iter()
                    .map(|it| it.as_const(context))
                    .collect::<Option<Vec<_>>>()?;
                Some(quote![(#(#elements),*)])
            }
            Expression::FieldAccess { base, field, .. } => {
                base.as_const(context).map(|base| quote![#base.#field])
            }
            Expression::Reference { inner } => inner.as_const(context).map(|base| quote![&#base]),
            Expression::MethodCall { .. } if self.is_const() => Some(self.to_tokens(context)),
            Expression::Match { .. } if self.is_const() => Some(self.to_tokens(context)),
            Expression::AssertConstant { inner } => Some(inner.to_tokens(context)),
            _ => None,
        }
    }

    pub fn as_index(&self) -> Option<(&Expression, &Expression)> {
        match self {
            Expression::Index { expr, index, .. } => Some((&**expr, &**index)),
            _ => None,
        }
    }

    pub fn needs_terminator(&self) -> bool {
        match self {
            Expression::If { then_block, .. } => then_block.ret.is_some(),
            Expression::Block(block) => block.ret.is_some(),
            Expression::ForLoop { .. } => false,
            Expression::Loop { .. } => false,
            Expression::VerbatimTerminated { .. } => false,
            _ => true,
        }
    }
}

pub fn is_intrinsic(path: &Path) -> bool {
    // Add both possible import paths
    let intrinsic_paths = [
        "::cubecl::prelude::vectorization_of",
        "::cubecl::frontend::vectorization_of",
    ];

    let mut path = path.clone();
    // Strip function generics
    path.segments.last_mut().unwrap().arguments = PathArguments::None;
    let func_path = path.to_token_stream().to_string();
    intrinsic_paths
        .iter()
        .any(|path| path.ends_with(&func_path))
}
