use std::{rc::Rc, sync::atomic::AtomicUsize};

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{AngleBracketedGenericArguments, Ident, Lit, Member, Pat, Path, PathSegment, Type};

use crate::{operator::Operator, scope::Context, statement::Statement};

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
    Variable {
        name: Ident,
        is_ref: bool,
        is_mut: bool,
        use_count: Rc<AtomicUsize>,
        ty: Option<Type>,
    },
    ConstVariable {
        name: Ident,
        use_count: Rc<AtomicUsize>,
        ty: Option<Type>,
    },
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
    Assigment {
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
    MethodCall {
        receiver: Box<Expression>,
        method: Ident,
        generics: Option<AngleBracketedGenericArguments>,
        args: Vec<Expression>,
    },
    Closure {
        params: Vec<Pat>,
        body: Box<Expression>,
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
    },
    WhileLoop {
        condition: Box<Expression>,
        block: Block,
    },
    Loop(Block),
    If {
        condition: Box<Expression>,
        then_block: Block,
        else_branch: Option<Box<Expression>>,
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
}

#[derive(Clone, Debug)]
pub struct Block {
    pub inner: Vec<Statement>,
    pub ret: Option<Box<Expression>>,
    pub ty: Option<Type>,
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
            Expression::Block(block) => block.ty.clone(),
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
            Expression::If { then_block, .. } => then_block.ty.clone(),
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
            Expression::Keyword { .. } => None,
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
            Expression::Tuple { elements, .. } => elements.iter().all(|it| it.is_const()),
            Expression::MethodCall {
                method,
                args,
                receiver,
                ..
            } => method == "vectorization_factor" && args.is_empty() || receiver.is_const(),
            _ => false,
        }
    }

    pub fn as_const(&self, context: &mut Context) -> Option<TokenStream> {
        match self {
            Expression::Literal { value, .. } => Some(quote![#value]),
            Expression::Verbatim { tokens, .. } => Some(tokens.clone()),
            Expression::VerbatimTerminated { tokens, .. } => Some(tokens.clone()),
            Expression::ConstVariable { name, .. } => Some(quote![#name.clone()]),
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
            Expression::WhileLoop { .. } => false,
            Expression::Loop { .. } => false,
            Expression::VerbatimTerminated { .. } => false,
            _ => true,
        }
    }
}
