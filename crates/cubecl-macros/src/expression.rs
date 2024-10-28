use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    AngleBracketedGenericArguments, Ident, Lit, Member, Pat, Path, PathArguments, PathSegment, Type,
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
        ty: Option<Type>,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        ty: Option<Type>,
    },
    Variable(ManagedVar),
    FieldAccess {
        base: Box<Expression>,
        field: Member,
    },
    Path {
        path: Path,
    },
    Literal {
        value: Lit,
        ty: Type,
    },
    Assignment {
        left: Box<Expression>,
        right: Box<Expression>,
        ty: Option<Type>,
    },
    Block(Block),
    FunctionCall {
        func: Box<Expression>,
        args: Vec<Expression>,
        associated_type: Option<(Path, PathSegment)>,
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
    Continue(Span),
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
        cases: Vec<(Lit, Block)>,
        default: Block,
    },
    Return {
        expr: Option<Box<Expression>>,
        span: Span,
        _ty: Type,
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
    },
    Slice {
        expr: Box<Expression>,
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
        name: syn::Ident,
    },
    ConstMatch {
        const_expr: syn::Expr,
        arms: Vec<ConstMatchArm>,
    },
}

#[derive(Clone, Debug)]
pub struct ConstMatchArm {
    pub pat: syn::Pat,
    pub expr: Box<Expression>,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub inner: Vec<Statement>,
    pub ret: Option<Box<Expression>>,
    pub ty: Option<Type>,
}

impl Expression {
    pub fn ty(&self) -> Option<Type> {
        match self {
            Self::Binary { ty, .. } | Self::Unary { ty, .. } | Self::Assignment { ty, .. } => {
                ty.clone()
            }
            Self::Variable(var) => var.ty.clone(),
            Self::Literal { ty, .. } => Some(ty.clone()),

            Self::Block(block) => block.ty.clone(),

            Self::Cast { to, .. } => Some(to.clone()),

            Self::Range { start, .. } => start.ty(),

            Self::If { then_block, .. } => then_block.ty.clone(),
            Self::Switch { default, .. } => default.ty.clone(),
            Self::Return { expr, .. } => expr.as_ref().and_then(|expr| expr.ty()),

            Self::Slice { expr, .. } => expr.ty(),
            Self::ArrayInit { init, .. } => init.ty(),

            Self::Reference { inner } => inner.ty(),
            Self::Verbatim { .. }
            | Self::FunctionCall { .. }
            | Self::Break { .. }
            | Self::Array { .. }
            | Self::Index { .. }
            | Self::Tuple { .. }
            | Self::Continue { .. }
            | Self::ForLoop { .. }
            | Self::FieldAccess { .. }
            | Self::MethodCall { .. }
            | Self::Path { .. }
            | Self::Loop { .. }
            | Self::VerbatimTerminated { .. }
            | Self::StructInit { .. }
            | Self::Closure { .. }
            | Self::Keyword { .. }
            | Self::CompilerIntrinsic { .. }
            | Self::ConstMatch { .. } => None,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            Self::Literal { .. }
            | Self::Path { .. }
            | Self::Verbatim { .. }
            | Self::VerbatimTerminated { .. }
            | Self::CompilerIntrinsic { .. } => true,
            Self::Variable(var) => var.is_const,
            Self::FieldAccess { base, .. } => base.is_const(),
            Self::Reference { inner } => inner.is_const(),
            Self::Array { elements, .. } | Self::Tuple { elements, .. } => {
                elements.iter().all(|it| it.is_const())
            }

            _ => false,
        }
    }

    pub fn as_const(&self, context: &mut Context) -> Option<TokenStream> {
        match self {
            Self::Literal { value, .. } => Some(quote![#value]),
            Self::Verbatim { tokens, .. } => Some(tokens.clone()),
            Self::VerbatimTerminated { tokens, .. } => Some(tokens.clone()),
            Self::Variable(ManagedVar {
                name,
                is_const: true,
                ..
            }) => Some(quote![#name.clone()]),
            Self::Path { path, .. } => Some(quote![#path]),
            Self::Array { elements, .. } => {
                let elements = elements
                    .iter()
                    .map(|it| it.as_const(context))
                    .collect::<Option<Vec<_>>>()?;
                Some(quote![[#(#elements),*]])
            }
            Self::Tuple { elements, .. } => {
                let elements = elements
                    .iter()
                    .map(|it| it.as_const(context))
                    .collect::<Option<Vec<_>>>()?;
                Some(quote![(#(#elements),*)])
            }
            Self::FieldAccess { base, field, .. } => {
                base.as_const(context).map(|base| quote![#base.#field])
            }
            Self::Reference { inner } => inner.as_const(context).map(|base| quote![&#base]),
            Self::MethodCall { .. } if self.is_const() => Some(self.to_tokens(context)),
            _ => None,
        }
    }

    pub fn as_index(&self) -> Option<(&Expression, &Expression)> {
        match self {
            Self::Index { expr, index, .. } => Some((&**expr, &**index)),
            _ => None,
        }
    }

    pub fn needs_terminator(&self) -> bool {
        match self {
            Self::If { then_block, .. } => then_block.ret.is_some(),
            Self::Block(block) => block.ret.is_some(),
            Self::ForLoop { .. } | Self::Loop { .. } | Self::VerbatimTerminated { .. } => false,
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
