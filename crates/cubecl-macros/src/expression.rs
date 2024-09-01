use cubecl_common::operator::Operator;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Lit, Member, Path, PathSegment, Type};

use crate::statement::Statement;

#[derive(Clone, Debug)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
        ty: Option<Type>,
        span: Span,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        ty: Option<Type>,
        span: Span,
    },
    Variable {
        name: Ident,
        ty: Option<Type>,
        span: Span,
    },
    ConstVariable {
        name: Ident,
        ty: Option<Type>,
    },
    FieldAccess {
        base: Box<Expression>,
        field: Member,
        span: Span,
    },
    Path {
        path: Path,
    },
    Literal {
        value: Lit,
        ty: Type,
        span: Span,
    },
    Assigment {
        left: Box<Expression>,
        right: Box<Expression>,
        ty: Option<Type>,
        span: Span,
    },
    Block {
        inner: Vec<Statement>,
        ret: Option<Box<Expression>>,
        ty: Option<Type>,
        span: Span,
    },
    FunctionCall {
        func: Box<Expression>,
        args: Vec<Expression>,
        associated_type: Option<(Path, PathSegment)>,
        span: Span,
    },
    MethodCall {
        receiver: Box<Expression>,
        method: Ident,
        args: Vec<Expression>,
        span: Span,
    },
    Cast {
        from: Box<Expression>,
        to: Type,
        span: Span,
    },
    Break {
        span: Span,
    },
    /// Tokens not relevant to parsing
    Verbatim {
        tokens: TokenStream,
    },
    VerbatimTerminated {
        tokens: TokenStream,
    },
    Continue {
        span: Span,
    },
    ForLoop {
        range: Box<Expression>,
        unroll: Option<Box<Expression>>,
        var_name: syn::Ident,
        var_ty: Option<syn::Type>,
        block: Box<Expression>,
        span: Span,
    },
    WhileLoop {
        condition: Box<Expression>,
        block: Box<Expression>,
        span: Span,
    },
    Loop {
        block: Box<Expression>,
        span: Span,
    },
    If {
        condition: Box<Expression>,
        then_block: Box<Expression>,
        else_branch: Option<Box<Expression>>,
        span: Span,
    },
    Return {
        expr: Option<Box<Expression>>,
        ty: Type,
        span: Span,
    },
    Range {
        start: Box<Expression>,
        end: Option<Box<Expression>>,
        inclusive: bool,
        span: Span,
    },
    Array {
        elements: Vec<Expression>,
        span: Span,
    },
    Tuple {
        elements: Vec<Expression>,
        span: Span,
    },
    Index {
        expr: Box<Expression>,
        index: Box<Expression>,
        span: Span,
    },
    Slice {
        expr: Box<Expression>,
        ranges: Vec<Expression>,
        span: Span,
    },
    ArrayInit {
        init: Box<Expression>,
        len: Box<Expression>,
        span: Span,
    },
    Reference {
        inner: Box<Expression>,
    },
    StructInit {
        path: Path,
        fields: Vec<(Member, Expression)>,
    },
    Closure {
        tokens: proc_macro2::TokenStream,
    },
}

impl Expression {
    pub fn ty(&self) -> Option<Type> {
        match self {
            Expression::Binary { ty, .. } => ty.clone(),
            Expression::Unary { ty, .. } => ty.clone(),
            Expression::Variable { ty, .. } => ty.clone(),
            Expression::ConstVariable { ty, .. } => ty.clone(),
            Expression::Literal { ty, .. } => Some(ty.clone()),
            Expression::Assigment { ty, .. } => ty.clone(),
            Expression::Verbatim { .. } => None,
            Expression::Block { ty, .. } => ty.clone(),
            Expression::FunctionCall { .. } => None,
            Expression::Break { .. } => None,
            Expression::Cast { to, .. } => Some(to.clone()),
            Expression::Continue { .. } => None,
            Expression::ForLoop { .. } => None,
            Expression::FieldAccess { .. } => None,
            Expression::MethodCall { .. } => None,
            Expression::Path { .. } => None,
            Expression::Range { start, .. } => start.ty(),
            Expression::WhileLoop { .. } => None,
            Expression::Loop { .. } => None,
            Expression::If { then_block, .. } => then_block.ty(),
            Expression::Return { expr, .. } => expr.as_ref().and_then(|expr| expr.ty()),
            Expression::Array { .. } => None,
            Expression::Index { .. } => None,
            Expression::Tuple { .. } => None,
            Expression::Slice { expr, .. } => expr.ty(),
            Expression::ArrayInit { init, .. } => init.ty(),
            Expression::VerbatimTerminated { .. } => None,
            Expression::Reference { inner } => inner.ty(),
            Expression::StructInit { .. } => None,
            Expression::Closure { .. } => None,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            Expression::Literal { .. } => true,
            Expression::Path { .. } => true,
            Expression::Verbatim { .. } => true,
            Expression::VerbatimTerminated { .. } => true,
            Expression::ConstVariable { .. } => true,
            Expression::FieldAccess { base, .. } => base.is_const(),
            Expression::Reference { inner } => inner.is_const(),
            Expression::Array { elements, .. } => elements.iter().all(|it| it.is_const()),
            Expression::FunctionCall {
                args,
                associated_type,
                ..
            } if associated_type.is_some() => args.iter().all(|it| it.is_const()),
            _ => false,
        }
    }

    pub fn as_const(&self) -> Option<TokenStream> {
        match self {
            Expression::Literal { value, .. } => Some(quote![#value]),
            Expression::Verbatim { tokens, .. } => Some(tokens.clone()),
            Expression::VerbatimTerminated { tokens, .. } => Some(tokens.clone()),
            Expression::ConstVariable { name, .. } => Some(quote![#name]),
            Expression::Path { path, .. } => Some(quote![#path]),
            Expression::Array { elements, .. } => {
                let elements = elements
                    .iter()
                    .map(|it| it.as_const())
                    .collect::<Option<Vec<_>>>()?;
                Some(quote![[#(#elements),*]])
            }
            Expression::FieldAccess { base, field, .. } => {
                base.as_const().map(|base| quote![#base.#field])
            }
            Expression::Reference { inner } => inner.as_const().map(|base| quote![&#base]),
            Expression::FunctionCall { .. } if self.is_const() => Some(quote![#self]),
            _ => None,
        }
    }

    pub fn needs_terminator(&self) -> bool {
        match self {
            Expression::If { then_block, .. } => then_block.needs_terminator(),
            Expression::Block { ret, .. } => ret.is_some(),
            Expression::ForLoop { .. } => false,
            Expression::WhileLoop { .. } => false,
            Expression::Loop { .. } => false,
            Expression::VerbatimTerminated { .. } => false,
            _ => true,
        }
    }
}
