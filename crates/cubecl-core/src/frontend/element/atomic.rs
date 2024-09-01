use crate::{
    ir::{BinaryOperator, CompareAndSwapOperator, Elem, Item, Operator, UnaryOperator},
    new_ir::{BinaryOp, Expr, Expression, SquareType, Vectorization},
    prelude::*,
    unexpanded,
};

use super::{ExpandElement, Numeric};

/// An atomic type. Represents an shared value that can be operated on atomically.
pub trait Atomic: Sized + SquareType {
    /// The numeric primitive represented by the atomic wrapper.
    type Primitive: Numeric;

    /// Load the value of the atomic.
    #[allow(unused_variables)]
    fn load(&self) -> Self::Primitive {
        unexpanded!()
    }

    /// Store the value of the atomic.
    #[allow(unused_variables)]
    fn store(&self, value: Self::Primitive) {
        unexpanded!()
    }

    /// Atomically stores the value into the atomic and returns the old value.
    #[allow(unused_variables)]
    fn swap(&self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Compare the value at `pointer` to `cmp` and set it to `value` only if they are the same.
    /// Returns the old value of the pointer before the store.
    ///
    /// ### Tip
    /// Compare the returned value to `cmp` to determine whether the store was successful.
    #[allow(unused_variables)]
    fn compare_and_swap(
        pointer: &Self,
        cmp: Self::Primitive,
        value: Self::Primitive,
    ) -> Self::Primitive {
        unexpanded!()
    }

    /// Atomically add a number to the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    fn add(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Atomically subtracts a number from the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    fn sub(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Atomically sets the value of the atomic variable to `max(current_value, value)`. Returns
    /// the old value.
    #[allow(unused_variables)]
    fn max(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Atomically sets the value of the atomic variable to `min(current_value, value)`. Returns the
    /// old value.
    #[allow(unused_variables)]
    fn min(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Executes an atomic bitwise and operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    fn and(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Executes an atomic bitwise or operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    fn or(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    /// Executes an atomic bitwise xor operation on the atomic variable. Returns the old value.
    #[allow(unused_variables)]
    fn xor(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AtomicExpr {
    Load {
        atomic: Box<Expression>,
        ty: Elem,
    },
    Store {
        atomic: Box<Expression>,
        value: Box<Expression>,
    },
    Swap {
        atomic: Box<Expression>,
        value: Box<Expression>,
        ty: Elem,
    },
    CompareAndSwap {
        atomic: Box<Expression>,
        cmp: Box<Expression>,
        value: Box<Expression>,
        ty: Elem,
    },
    Binary {
        atomic: Box<Expression>,
        value: Box<Expression>,
        op: AtomicOp,
        ty: Elem,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum AtomicOp {
    Add,
    Sub,
    Max,
    Min,
    And,
    Or,
    Xor,
}

impl AtomicExpr {
    pub fn ir_type(&self) -> Elem {
        match self {
            AtomicExpr::Load { ty, .. } => *ty,
            AtomicExpr::Store { .. } => Elem::Unit,
            AtomicExpr::Swap { ty, .. } => *ty,
            AtomicExpr::CompareAndSwap { ty, .. } => *ty,
            AtomicExpr::Binary { ty, .. } => *ty,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        None
    }

    pub fn flatten(self, context: &mut CubeContext) -> Option<ExpandElement> {
        match self {
            AtomicExpr::Load { atomic, ty } => {
                let atomic = atomic.flatten(context).unwrap().into_variable();
                let out = context.create_local(Item::new(ty));
                context.register(Operator::AtomicLoad(UnaryOperator {
                    input: atomic,
                    out: out.as_variable(),
                }));
                out.into()
            }
            AtomicExpr::Store { atomic, value } => {
                let atomic = atomic.flatten(context).unwrap().into_variable();
                let value = value.flatten(context).unwrap().into_variable();
                context.register(Operator::AtomicStore(UnaryOperator {
                    input: value,
                    out: atomic,
                }));
                None
            }
            AtomicExpr::Swap { atomic, value, ty } => {
                let atomic = atomic.flatten(context).unwrap().into_variable();
                let value = value.flatten(context).unwrap().into_variable();
                let out = context.create_local(Item::new(ty));
                context.register(Operator::AtomicSwap(BinaryOperator {
                    lhs: atomic,
                    rhs: value,
                    out: out.as_variable(),
                }));
                out.into()
            }
            AtomicExpr::CompareAndSwap {
                atomic,
                cmp,
                value,
                ty,
            } => {
                let atomic = atomic.flatten(context).unwrap().into_variable();
                let cmp = cmp.flatten(context).unwrap().into_variable();
                let value = value.flatten(context).unwrap().into_variable();
                let out = context.create_local(Item::new(ty));
                context.register(Operator::AtomicCompareAndSwap(CompareAndSwapOperator {
                    out: out.as_variable(),
                    input: atomic,
                    cmp,
                    val: value,
                }));
                out.into()
            }
            AtomicExpr::Binary {
                atomic,
                value,
                op,
                ty,
            } => {
                let atomic = atomic.flatten(context).unwrap().into_variable();
                let value = value.flatten(context).unwrap().into_variable();
                let out = context.create_local(Item::new(ty));
                let bin_op = BinaryOperator {
                    lhs: atomic,
                    rhs: value,
                    out: out.as_variable(),
                };
                context.register(map_op(op, bin_op));
                out.into()
            }
        }
    }
}

fn map_op(op: AtomicOp, bin_op: BinaryOperator) -> Operator {
    match op {
        AtomicOp::Add => Operator::AtomicAdd(bin_op),
        AtomicOp::Sub => Operator::AtomicSub(bin_op),
        AtomicOp::Max => Operator::AtomicMax(bin_op),
        AtomicOp::Min => Operator::AtomicMin(bin_op),
        AtomicOp::And => Operator::AtomicAnd(bin_op),
        AtomicOp::Or => Operator::AtomicOr(bin_op),
        AtomicOp::Xor => Operator::AtomicXor(bin_op),
    }
}

#[derive(new)]
pub struct AtomicLoad<T: Expr>(pub T)
where
    T::Output: Atomic;

impl<T: Expr> Expr for AtomicLoad<T>
where
    T::Output: Atomic,
{
    type Output = <T::Output as Atomic>::Primitive;

    fn expression_untyped(&self) -> Expression {
        AtomicExpr::Load {
            atomic: Box::new(self.0.expression_untyped()),
            ty: <T::Output as Atomic>::Primitive::ir_type(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct AtomicStore<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>>
where
    T::Output: Atomic,
{
    pub atomic: T,
    pub value: Value,
}

impl<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>> Expr for AtomicStore<T, Value>
where
    T::Output: Atomic,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        AtomicExpr::Store {
            atomic: Box::new(self.atomic.expression_untyped()),
            value: Box::new(self.value.expression_untyped()),
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct AtomicSwap<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>>
where
    T::Output: Atomic,
{
    pub atomic: T,
    pub value: Value,
}

impl<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>> Expr for AtomicSwap<T, Value>
where
    T::Output: Atomic,
{
    type Output = <T::Output as Atomic>::Primitive;

    fn expression_untyped(&self) -> Expression {
        AtomicExpr::Swap {
            atomic: Box::new(self.atomic.expression_untyped()),
            value: Box::new(self.value.expression_untyped()),
            ty: <T::Output as Atomic>::Primitive::ir_type(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct AtomicCompareAndSwap<
    T: Expr,
    Cmp: Expr<Output = <T::Output as Atomic>::Primitive>,
    Value: Expr<Output = <T::Output as Atomic>::Primitive>,
> where
    T::Output: Atomic,
{
    pub atomic: T,
    pub cmp: Cmp,
    pub value: Value,
}

impl<
        T: Expr,
        Cmp: Expr<Output = <T::Output as Atomic>::Primitive>,
        Value: Expr<Output = <T::Output as Atomic>::Primitive>,
    > Expr for AtomicCompareAndSwap<T, Cmp, Value>
where
    T::Output: Atomic,
{
    type Output = <T::Output as Atomic>::Primitive;

    fn expression_untyped(&self) -> Expression {
        AtomicExpr::CompareAndSwap {
            atomic: Box::new(self.atomic.expression_untyped()),
            cmp: Box::new(self.cmp.expression_untyped()),
            value: Box::new(self.value.expression_untyped()),
            ty: <T::Output as Atomic>::Primitive::ir_type(),
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

macro_rules! atomic_bin_op {
    ($name:ident, $op:ident) => {
        pub struct $name<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>>(
            pub BinaryOp<T, Value, <T::Output as Atomic>::Primitive>,
        )
        where
            T::Output: Atomic;

        impl<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>> $name<T, Value>
        where
            T::Output: Atomic,
        {
            pub fn new(left: T, right: Value) -> Self {
                Self(BinaryOp::new(left, right))
            }
        }

        impl<T: Expr, Value: Expr<Output = <T::Output as Atomic>::Primitive>> Expr
            for $name<T, Value>
        where
            T::Output: Atomic,
        {
            type Output = <T::Output as Atomic>::Primitive;

            fn expression_untyped(&self) -> Expression {
                AtomicExpr::Binary {
                    atomic: Box::new(self.0.left.expression_untyped()),
                    value: Box::new(self.0.right.expression_untyped()),
                    op: AtomicOp::$op,
                    ty: <T::Output as Atomic>::Primitive::ir_type(),
                }
                .into()
            }

            fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
                None
            }
        }
    };
}

atomic_bin_op!(AtomicAdd, Add);
atomic_bin_op!(AtomicSub, Sub);
atomic_bin_op!(AtomicMin, Min);
atomic_bin_op!(AtomicMax, Max);
atomic_bin_op!(AtomicOr, Or);
atomic_bin_op!(AtomicAnd, And);
atomic_bin_op!(AtomicXor, Xor);

macro_rules! impl_atomic_expand {
    ($name:ident, $unexpanded:ident) => {
        impl<Inner: Expr<Output = $unexpanded>> $name<Inner> {
            pub fn load(self) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicLoad::new(self.0)
            }

            pub fn store(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = ()> {
                AtomicStore::new(self.0, value)
            }

            pub fn swap(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicSwap::new(self.0, value)
            }

            pub fn compare_and_swap(
                self,
                cmp: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicCompareAndSwap::new(self.0, cmp, value)
            }

            #[allow(clippy::should_implement_trait)]
            pub fn add(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicAdd::new(self.0, value)
            }

            #[allow(clippy::should_implement_trait)]
            pub fn sub(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicSub::new(self.0, value)
            }

            pub fn max(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicMax::new(self.0, value)
            }

            pub fn min(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicMin::new(self.0, value)
            }

            pub fn and(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicAnd::new(self.0, value)
            }

            pub fn or(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicOr::new(self.0, value)
            }

            pub fn xor(
                self,
                value: impl Expr<Output = <Inner::Output as Atomic>::Primitive>,
            ) -> impl Expr<Output = <Inner::Output as Atomic>::Primitive> {
                AtomicXor::new(self.0, value)
            }
        }
    };
}

#[derive(Expand, Clone, Copy)]
#[expand(ir_type = u32::ir_type())]
pub struct AtomicU32(#[expand(skip)] pub u32);
impl Atomic for AtomicU32 {
    type Primitive = u32;
}

#[derive(Expand, Clone, Copy)]
#[expand(ir_type = i32::ir_type())]
pub struct AtomicI32(#[expand(skip)] pub i32);
impl Atomic for AtomicI32 {
    type Primitive = i32;
}

impl_atomic_expand!(AtomicU32Expand, AtomicU32);
impl_atomic_expand!(AtomicI32Expand, AtomicI32);
