use crate::{
    cmma::CmmaExpression,
    compute::GlobalType,
    ir::{self, ConstantScalarValue, Elem},
    prelude::{AtomicExpr, SharedMemoryExpr},
};
use derive_more::derive::From;
use std::{marker::PhantomData, num::NonZero, rc::Rc};

use super::{
    largest_common_vectorization, Operator, SquareType, Statement, SubcubeExpression,
    TensorExpression,
};

pub type Vectorization = Option<NonZero<u8>>;

#[derive(Clone, Debug, PartialEq, From)]
pub enum Expression {
    Binary {
        left: Box<Expression>,
        operator: Operator,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    Unary {
        input: Box<Expression>,
        operator: Operator,
        vectorization: Vectorization,
        ty: Elem,
    },
    Clamp {
        input: Box<Expression>,
        min: Box<Expression>,
        max: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    #[from]
    Variable(Var),
    Global {
        index: u16,
        global_ty: GlobalType,
        vectorization: Vectorization,
        ty: Elem,
    },
    FieldAccess {
        base: Box<Expression>,
        name: String,
        vectorization: Vectorization,
        ty: Elem,
    },
    Literal {
        value: ConstantScalarValue,
        vectorization: Vectorization,
        ty: Elem,
    },
    Assigment {
        left: Box<Expression>,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    /// Local variable initializer
    Init {
        left: Var,
        right: Box<Expression>,
        vectorization: Vectorization,
        ty: Elem,
    },
    Block(Block),
    Break,
    Cast {
        from: Box<Expression>,
        vectorization: Vectorization,
        to: Elem,
    },
    BitCast {
        from: Box<Expression>,
        vectorization: Vectorization,
        to: Elem,
    },
    Continue,
    ForLoop {
        range: Range,
        unroll: bool,
        variable: Var,
        block: Block,
    },
    WhileLoop {
        condition: Box<Expression>,
        block: Block,
    },
    Loop {
        block: Block,
    },
    If {
        condition: Box<Expression>,
        then_block: Block,
        else_branch: Option<Box<Expression>>,
    },
    Return {
        expr: Option<Box<Expression>>,
    },
    /// Subtype for tensor specific operations
    #[from]
    Tensor(TensorExpression),
    #[from]
    Subcube(SubcubeExpression),
    #[from]
    Cmma(CmmaExpression),
    #[from]
    Atomic(AtomicExpr),
    #[from]
    SharedMemory(SharedMemoryExpr),
    ArrayInit {
        size: u32,
        ty: Elem,
        vectorization: Vectorization,
    },
    KernelVar {
        kind: ir::Variable,
        ty: Elem,
    },
    /// A range used in for loops. Currently doesn't exist at runtime, so can be ignored in codegen.
    /// This only exists to pass the range down to the for loop it applies to
    __Range(Range),
    Fma {
        a: Box<Expression>,
        b: Box<Expression>,
        c: Box<Expression>,
        ty: crate::ir::Elem,
        vectorization: Option<std::num::NonZero<u8>>,
    },
}

#[derive(Clone, Debug, PartialEq, new)]
pub struct Var {
    pub name: String,
    pub vectorization: Vectorization,
    pub ty: Elem,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub start: Box<Expression>,
    pub end: Box<Expression>,
    pub step: Option<Box<Expression>>,
    pub inclusive: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub inner: Vec<Statement>,
    pub ret: Box<Expression>,
    pub vectorization: Vectorization,
    pub ty: Elem,
}

impl Expression {
    pub fn ir_type(&self) -> Elem {
        match self {
            Expression::Binary { ty, .. } => *ty,
            Expression::Unary { ty, .. } => *ty,
            Expression::Variable(var) => var.ty,
            Expression::Literal { ty, .. } => *ty,
            Expression::Assigment { ty, .. } => *ty,
            Expression::Init { ty, .. } => *ty,
            Expression::Block(block) => block.ret.ir_type(),
            Expression::Cast { to, .. } => *to,
            Expression::BitCast { to, .. } => *to,
            Expression::Break | Expression::Continue | Expression::ForLoop { .. } => Elem::Unit,
            Expression::FieldAccess { ty, .. } => *ty,
            Expression::__Range(_) => Elem::Unit,
            Expression::WhileLoop { .. } => Elem::Unit,
            Expression::Loop { .. } => Elem::Unit,
            Expression::If { then_block, .. } => then_block.ret.ir_type(),
            Expression::Return { expr } => {
                expr.as_ref().map(|it| it.ir_type()).unwrap_or(Elem::Unit)
            }
            Expression::Tensor(tensor) => tensor.ir_type(),
            Expression::ArrayInit { ty, .. } => *ty,
            Expression::Global { ty, .. } => *ty,
            Expression::KernelVar { ty, .. } => *ty,
            Expression::Subcube(expr) => expr.ir_type(),
            Expression::Cmma(expr) => expr.ir_type(),
            Expression::Atomic(expr) => expr.ir_type(),
            Expression::SharedMemory(expr) => expr.ir_type(),
            Expression::Fma { ty, .. } => *ty,
            Expression::Clamp { ty, .. } => *ty,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        match self {
            Expression::Binary { vectorization, .. } => *vectorization,
            Expression::Unary { vectorization, .. } => *vectorization,
            Expression::Variable(var) => var.vectorization,
            Expression::Global { vectorization, .. } => *vectorization,
            Expression::FieldAccess { vectorization, .. } => *vectorization,
            Expression::Literal { vectorization, .. } => *vectorization,
            Expression::Assigment { vectorization, .. } => *vectorization,
            Expression::Init { vectorization, .. } => *vectorization,
            Expression::Block(block) => block.vectorization,
            Expression::Break => None,
            Expression::Cast { vectorization, .. } => *vectorization,
            Expression::BitCast { vectorization, .. } => *vectorization,
            Expression::Continue => None,
            Expression::ForLoop { .. } => None,
            Expression::WhileLoop { block, .. } => block.vectorization,
            Expression::Loop { .. } => None,
            Expression::If { then_block, .. } => then_block.vectorization,
            Expression::Return { .. } => None,
            Expression::Tensor(tensor) => tensor.vectorization(),
            Expression::ArrayInit { vectorization, .. } => *vectorization,
            Expression::__Range(_) => None,
            Expression::KernelVar { .. } => None,
            Expression::Subcube(expr) => expr.vectorization(),
            Expression::Cmma(expr) => expr.vectorization(),
            Expression::SharedMemory(expr) => expr.vectorization(),
            Expression::Atomic(expr) => expr.vectorization(),
            Expression::Clamp { vectorization, .. } => *vectorization,
            Expression::Fma {
                vectorization: vectorisation,
                ..
            } => *vectorisation,
        }
    }

    pub fn as_range(&self) -> Option<&Range> {
        match self {
            Expression::__Range(range) => Some(range),
            _ => None,
        }
    }

    pub fn as_block(self) -> Option<Block> {
        match self {
            Expression::Block(block) => Some(block),
            _ => None,
        }
    }

    pub fn as_lit(self) -> Option<ConstantScalarValue> {
        match self {
            Expression::Literal { value, .. } => Some(value),
            _ => None,
        }
    }

    pub fn as_variable(self) -> Option<Var> {
        match self {
            Expression::Variable(var) => Some(var),
            _ => None,
        }
    }
}

pub trait Expr {
    type Output;

    fn expression_untyped(&self) -> Expression;
    fn vectorization(&self) -> Option<NonZero<u8>>;
}

#[derive(Debug, Hash, PartialEq)]
pub struct Variable<T: SquareType> {
    pub name: &'static str,
    pub vectorization: Vectorization,
    pub _type: PhantomData<T>,
}

#[derive(Debug, PartialEq)]
pub struct KernelVariable<T: SquareType> {
    pub kind: ir::Variable,
    pub _type: PhantomData<T>,
}

impl<T: SquareType> Copy for KernelVariable<T> {}
impl<T: SquareType> Clone for KernelVariable<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: SquareType> Expr for KernelVariable<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::KernelVar {
            kind: self.kind,
            ty: T::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<T: SquareType> Variable<T> {
    pub const fn new(name: &'static str, vectorization: Vectorization) -> Self {
        Self {
            name,
            vectorization,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType> Copy for Variable<T> {}
#[allow(clippy::non_canonical_clone_impl)]
impl<T: SquareType> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            vectorization: self.vectorization,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType> Expr for Variable<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Var {
            name: self.name.to_string(),
            ty: <T as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.vectorization
    }
}

#[derive(Debug, new, Hash, PartialEq)]
pub struct GlobalVariable<T: SquareType> {
    pub index: u16,
    pub ty: GlobalType,
    pub vectorization: Vectorization,
    pub _type: PhantomData<T>,
}

impl<T: SquareType> Copy for GlobalVariable<T> {}
#[allow(clippy::non_canonical_clone_impl)]
impl<T: SquareType> Clone for GlobalVariable<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            ty: self.ty,
            vectorization: self.vectorization,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType> Expr for GlobalVariable<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Global {
            index: self.index,
            global_ty: self.ty,
            ty: <T as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.vectorization
    }
}

#[derive(new, Hash)]
pub struct FieldAccess<T: SquareType, TBase: Expr> {
    pub base: TBase,
    pub name: &'static str,
    pub _type: PhantomData<T>,
}

impl<T: SquareType, TBase: Expr + Clone> Clone for FieldAccess<T, TBase> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            name: self.name,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType, TBase: Expr> Expr for FieldAccess<T, TBase> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::FieldAccess {
            base: Box::new(self.base.expression_untyped()),
            name: self.name.to_string(),
            ty: <T as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.base.vectorization()
    }
}

pub struct Assignment<Left: Expr, Right: Expr<Output = Left::Output>>
where
    Left::Output: SquareType,
{
    pub left: Left,
    pub right: Right,
}

impl<Left: Expr, Right: Expr<Output = Left::Output>> Expr for Assignment<Left, Right>
where
    Left::Output: SquareType,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::Assigment {
            left: Box::new(self.left.expression_untyped()),
            right: Box::new(self.right.expression_untyped()),
            ty: <Left::Output as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        largest_common_vectorization(self.left.vectorization(), self.right.vectorization())
    }
}

pub struct Initializer<Init: Expr>
where
    Init::Output: SquareType,
{
    pub left: Variable<Init::Output>,
    pub right: Init,
}

impl<Init: Expr> Expr for Initializer<Init>
where
    Init::Output: SquareType,
{
    type Output = Init::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Init {
            left: self.left.expression_untyped().as_variable().unwrap(),
            right: Box::new(self.right.expression_untyped()),
            ty: <Init::Output as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.right.vectorization()
    }
}

#[derive(new)]
pub struct Cast<From: Expr, TTo: SquareType>
where
    From::Output: SquareType,
{
    pub from: From,
    pub _to: PhantomData<TTo>,
}

impl<From: Expr, TTo: SquareType> Expr for Cast<From, TTo>
where
    From::Output: SquareType,
{
    type Output = TTo;

    fn expression_untyped(&self) -> Expression {
        Expression::Cast {
            from: Box::new(self.from.expression_untyped()),
            to: <TTo as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.from.vectorization()
    }
}

#[derive(new)]
pub struct BitCast<From: Expr, TTo: SquareType>
where
    From::Output: SquareType,
{
    pub from: From,
    pub _to: PhantomData<TTo>,
}

impl<From: Expr, TTo: SquareType> Expr for BitCast<From, TTo>
where
    From::Output: SquareType,
{
    type Output = TTo;

    fn expression_untyped(&self) -> Expression {
        Expression::BitCast {
            from: Box::new(self.from.expression_untyped()),
            to: <TTo as SquareType>::ir_type(),
            vectorization: self.vectorization(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.from.vectorization()
    }
}

pub struct DynamicExpr<T>(pub Box<dyn Expr<Output = T>>);

impl<T> DynamicExpr<T> {
    pub fn new(value: impl Expr<Output = T> + 'static) -> Self {
        Self(Box::new(value))
    }
}

impl<T> Expr for DynamicExpr<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        self.0.expression_untyped()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.0.vectorization()
    }
}

pub struct RcExpr<T>(pub Rc<dyn Expr<Output = T>>);

impl<T> RcExpr<T> {
    pub fn new(value: impl Expr<Output = T> + 'static) -> Self {
        Self(Rc::new(value))
    }
}

impl<T> Expr for RcExpr<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        self.0.expression_untyped()
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.0.vectorization()
    }
}

impl<T> Clone for RcExpr<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
