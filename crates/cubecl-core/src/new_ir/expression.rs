use crate::{
    cmma::CmmaExpression,
    compute::GlobalType,
    ir::{self, ConstantScalarValue, Elem, Synchronization},
    prelude::{AtomicExpr, ExpandElement, SharedMemoryExpr},
};
use derive_more::derive::From;
use std::{
    cell::RefCell, collections::HashMap, fmt::Debug, marker::PhantomData, num::NonZero, rc::Rc,
};

use super::{
    largest_common_vectorization, Operator, SquareType, Statement, SubcubeExpression,
    TensorExpression,
};

pub type Vectorization = Option<NonZero<u8>>;

#[derive(Clone)]
pub struct BlockConstructor(pub Rc<dyn Fn() -> Block>);

impl Debug for BlockConstructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BlockConstructor").finish()
    }
}

impl PartialEq for BlockConstructor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

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
    RuntimeStruct {
        fields: HashMap<&'static str, Expression>,
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
        variable: Var,
        unroll: bool,
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
    Once(Rc<OnceExpression>),
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
    Sync(Synchronization),
}

#[derive(Clone, Debug, PartialEq, new)]
pub struct Var {
    pub name: Rc<String>,
    pub mutable: bool,
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

impl Range {
    pub fn deep_clone(&self) -> Self {
        Self {
            start: Box::new(self.start.deep_clone()),
            end: Box::new(self.end.deep_clone()),
            step: self.step.as_ref().map(|it| Box::new(it.deep_clone())),
            inclusive: self.inclusive,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub inner: Vec<Statement>,
    pub ret: Box<Expression>,
    pub vectorization: Vectorization,
    pub ty: Elem,
}

impl Block {
    pub fn deep_clone(&self) -> Self {
        Block {
            inner: self.inner.iter().map(|it| it.deep_clone()).collect(),
            ret: Box::new(self.ret.deep_clone()),
            vectorization: self.vectorization,
            ty: self.ty,
        }
    }
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
            Expression::RuntimeStruct { .. } => Elem::Unit,
            Expression::Sync(_) => Elem::Unit,
            Expression::Once(once) => once.ty,
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
            Expression::RuntimeStruct { .. } => NonZero::new(1),
            Expression::Sync(_) => None,
            Expression::Once(once) => once.vectorization,
        }
    }

    /// Do a deep clone including of `Once` values
    pub fn deep_clone(&self) -> Self {
        match self {
            Expression::Init {
                left,
                right,
                vectorization,
                ty,
            } => Expression::Init {
                left: left.clone(),
                right: Box::new(right.deep_clone()),
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Once(once) => Expression::Once(Rc::new(once.deep_clone())),
            Expression::Binary {
                left,
                operator,
                right,
                vectorization,
                ty,
            } => Expression::Binary {
                left: Box::new(left.deep_clone()),
                operator: *operator,
                right: Box::new(right.deep_clone()),
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Unary {
                input,
                operator,
                vectorization,
                ty,
            } => Expression::Unary {
                input: Box::new(input.deep_clone()),
                operator: *operator,
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Clamp {
                input,
                min,
                max,
                vectorization,
                ty,
            } => Expression::Clamp {
                input: Box::new(input.deep_clone()),
                min: Box::new(min.deep_clone()),
                max: Box::new(max.deep_clone()),
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Variable(var) => Expression::Variable(var.clone()),
            Expression::Global {
                index,
                global_ty,
                vectorization,
                ty,
            } => Expression::Global {
                index: *index,
                global_ty: *global_ty,
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::FieldAccess {
                base,
                name,
                vectorization,
                ty,
            } => Expression::FieldAccess {
                base: Box::new(base.deep_clone()),
                name: name.clone(),
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::RuntimeStruct { fields } => Expression::RuntimeStruct {
                fields: fields
                    .iter()
                    .map(|(name, value)| (*name, value.deep_clone()))
                    .collect(),
            },
            Expression::Literal {
                value,
                vectorization,
                ty,
            } => Expression::Literal {
                value: *value,
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Assigment {
                left,
                right,
                vectorization,
                ty,
            } => Expression::Assigment {
                left: Box::new(left.deep_clone()),
                right: Box::new(right.deep_clone()),
                vectorization: *vectorization,
                ty: *ty,
            },
            Expression::Block(block) => Expression::Block(block.deep_clone()),
            Expression::Break => todo!(),
            Expression::Cast {
                from,
                vectorization,
                to,
            } => Expression::Cast {
                from: Box::new(from.deep_clone()),
                vectorization: *vectorization,
                to: *to,
            },
            Expression::BitCast {
                from,
                vectorization,
                to,
            } => Expression::BitCast {
                from: Box::new(from.deep_clone()),
                vectorization: *vectorization,
                to: *to,
            },
            Expression::Continue => Expression::Continue,
            Expression::ForLoop {
                range,
                variable,
                unroll,
                block,
            } => Expression::ForLoop {
                range: range.deep_clone(),
                variable: variable.clone(),
                unroll: *unroll,
                block: block.deep_clone(),
            },
            Expression::WhileLoop { condition, block } => Expression::WhileLoop {
                condition: Box::new(condition.deep_clone()),
                block: block.deep_clone(),
            },
            Expression::Loop { block } => Expression::Loop {
                block: block.deep_clone(),
            },
            Expression::If {
                condition,
                then_block,
                else_branch,
            } => Expression::If {
                condition: Box::new(condition.deep_clone()),
                then_block: then_block.deep_clone(),
                else_branch: else_branch.as_ref().map(|it| Box::new(it.deep_clone())),
            },
            Expression::Return { expr } => Expression::Return {
                expr: expr.as_ref().map(|it| Box::new(it.deep_clone())),
            },
            Expression::Tensor(tensor) => Expression::Tensor(tensor.deep_clone()),
            Expression::Subcube(subcube) => Expression::Subcube(subcube.deep_clone()),
            Expression::Cmma(cmma) => Expression::Cmma(cmma.deep_clone()),
            Expression::Atomic(atomic) => Expression::Atomic(atomic.deep_clone()),
            Expression::SharedMemory(shared) => Expression::SharedMemory(shared.deep_clone()),
            Expression::ArrayInit { .. } => self.clone(),
            Expression::KernelVar { .. } => self.clone(),
            Expression::__Range(range) => Expression::__Range(range.deep_clone()),
            Expression::Fma {
                a,
                b,
                c,
                ty,
                vectorization,
            } => Expression::Fma {
                a: Box::new(a.deep_clone()),
                b: Box::new(b.deep_clone()),
                c: Box::new(c.deep_clone()),
                ty: *ty,
                vectorization: *vectorization,
            },
            Expression::Sync(_) => self.clone(),
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

#[derive(Debug, Clone, PartialEq)]
pub struct OnceExpression {
    expr: Expression,
    expanded: RefCell<Option<ExpandElement>>,
    ty: Elem,
    vectorization: Vectorization,
}

impl OnceExpression {
    pub fn new(expr: Expression) -> Self {
        OnceExpression {
            ty: expr.ir_type(),
            vectorization: expr.vectorization(),
            expr,
            expanded: RefCell::new(None),
        }
    }

    pub fn get_or_expand_with(
        &self,
        init: impl FnOnce(Expression) -> ExpandElement,
    ) -> ExpandElement {
        let value = { self.expanded.borrow().clone() };
        if let Some(value) = value {
            value
        } else {
            let expanded = init(self.expr.clone());
            *self.expanded.borrow_mut() = Some(expanded.clone());
            expanded
        }
    }

    fn deep_clone(&self) -> Self {
        // Reset value
        Self {
            expr: self.expr.deep_clone(),
            expanded: RefCell::new(None),
            vectorization: self.vectorization,
            ty: self.ty,
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
    pub name: Rc<String>,
    pub mutable: bool,
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
    pub fn new(name: &'static str, mutable: bool, vectorization: Vectorization) -> Self {
        Self {
            name: Rc::new(name.to_string()),
            mutable,
            vectorization,
            _type: PhantomData,
        }
    }
}

//impl<T: SquareType> Copy for Variable<T> {}
#[allow(clippy::non_canonical_clone_impl)]
impl<T: SquareType> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            mutable: self.mutable,
            vectorization: self.vectorization,
            _type: PhantomData,
        }
    }
}

impl<T: SquareType> Expr for Variable<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Var {
            name: self.name.clone(),
            mutable: self.mutable,
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
        let inner = self.base.expression_untyped();
        match inner {
            Expression::RuntimeStruct { fields } => fields[self.name].clone(),
            inner => Expression::FieldAccess {
                base: Box::new(inner),
                name: self.name.to_string(),
                ty: <T as SquareType>::ir_type(),
                vectorization: self.vectorization(),
            },
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        // Reset vectorization for indexing
        None
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
pub struct BitCastExpr<From: Expr, TTo: SquareType>
where
    From::Output: SquareType,
{
    pub from: From,
    pub _to: PhantomData<TTo>,
}

impl<From: Expr, TTo: SquareType> Expr for BitCastExpr<From, TTo>
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

pub struct OnceExpr<T> {
    inner: Rc<OnceExpression>,
    _type: PhantomData<T>,
}

impl<T> OnceExpr<T> {
    pub fn new(value: impl Expr<Output = T> + 'static) -> Self {
        let value = OnceExpression::new(value.expression_untyped());
        Self {
            inner: Rc::new(value),
            _type: PhantomData,
        }
    }
}

impl<T> Expr for OnceExpr<T> {
    type Output = T;

    fn expression_untyped(&self) -> Expression {
        Expression::Once(self.inner.clone())
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.inner.vectorization
    }
}

impl<T> Clone for OnceExpr<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _type: PhantomData,
        }
    }
}
