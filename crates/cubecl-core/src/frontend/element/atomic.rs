use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime, LaunchArgExpand,
    Numeric,
};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement},
    ir::{
        BinaryOperator, CompareAndSwapOperator, Elem, Instruction, Item, Operation, UnaryOperator,
    },
    prelude::KernelBuilder,
    unexpanded,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AtomicOp {
    Load(UnaryOperator),
    Store(UnaryOperator),
    Swap(BinaryOperator),
    Add(BinaryOperator),
    Sub(BinaryOperator),
    Max(BinaryOperator),
    Min(BinaryOperator),
    And(BinaryOperator),
    Or(BinaryOperator),
    Xor(BinaryOperator),
    CompareAndSwap(CompareAndSwapOperator),
}

impl Display for AtomicOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomicOp::Load(op) => write!(f, "atomic_load({})", op.input),
            AtomicOp::Store(op) => write!(f, "atomic_store({})", op.input),
            AtomicOp::Swap(op) => {
                write!(f, "atomic_swap({}, {})", op.lhs, op.rhs)
            }
            AtomicOp::Add(op) => write!(f, "atomic_add({}, {})", op.lhs, op.rhs),
            AtomicOp::Sub(op) => write!(f, "atomic_sub({}, {})", op.lhs, op.rhs),
            AtomicOp::Max(op) => write!(f, "atomic_max({}, {})", op.lhs, op.rhs),
            AtomicOp::Min(op) => write!(f, "atomic_min({}, {})", op.lhs, op.rhs),
            AtomicOp::And(op) => write!(f, "atomic_and({}, {})", op.lhs, op.rhs),
            AtomicOp::Or(op) => write!(f, "atomic_or({}, {})", op.lhs, op.rhs),
            AtomicOp::Xor(op) => write!(f, "atomic_xor({}, {})", op.lhs, op.rhs),
            AtomicOp::CompareAndSwap(op) => {
                write!(f, "compare_and_swap({}, {}, {})", op.input, op.cmp, op.val)
            }
        }
    }
}

/// An atomic type. Represents an shared value that can be operated on atomically.
pub trait AtomicTrait: Sized + CubeType
where
    ExpandElement: From<<Self::Primitive as CubeType>::ExpandType>,
    ExpandElement: From<<Self as CubeType>::ExpandType>,
{
    /// The numeric primitive represented by the atomic wrapper.
    type Primitive: Numeric;

    /// Load the value of the atomic.
    #[allow(unused_variables)]
    fn load(pointer: &Self) -> Self::Primitive {
        unexpanded!()
    }

    /// Store the value of the atomic.
    #[allow(unused_variables)]
    fn store(pointer: &Self, value: Self::Primitive) {
        unexpanded!()
    }

    /// Atomically stores the value into the atomic and returns the old value.
    #[allow(unused_variables)]
    fn swap(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
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

    fn __expand_load(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let pointer: ExpandElement = pointer.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Load(UnaryOperator { input: *pointer }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_store(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        context.register(Instruction::new(
            AtomicOp::Store(UnaryOperator { input: *value }),
            *ptr,
        ));
    }

    fn __expand_swap(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Swap(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_compare_and_swap(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        cmp: <Self::Primitive as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let pointer: ExpandElement = pointer.into();
        let cmp: ExpandElement = cmp.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::CompareAndSwap(CompareAndSwapOperator {
                input: *pointer,
                cmp: *cmp,
                val: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_add(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Add(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_sub(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Sub(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_max(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Max(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_min(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Min(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_and(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::And(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_or(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Or(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }

    fn __expand_xor(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem(context)));
        context.register(Instruction::new(
            AtomicOp::Xor(BinaryOperator {
                lhs: *ptr,
                rhs: *value,
            }),
            *new_var,
        ));
        new_var.into()
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Atomic<Inner: CubePrimitive> {
    pub val: Inner,
}

impl<Inner: CubePrimitive> CubeType for Atomic<Inner> {
    type ExpandType = ExpandElementTyped<Self>;
}

impl<Inner: CubePrimitive> IntoRuntime for Atomic<Inner> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        unimplemented!("Atomics don't exist at compile time")
    }
}

impl<Inner: CubePrimitive> CubePrimitive for Atomic<Inner> {
    fn as_elem_native() -> Option<Elem> {
        match Inner::as_elem_native() {
            Some(Elem::Float(kind)) => Some(Elem::AtomicFloat(kind)),
            Some(Elem::Int(kind)) => Some(Elem::AtomicInt(kind)),
            Some(Elem::UInt(kind)) => Some(Elem::AtomicUInt(kind)),
            None => None,
            _ => unreachable!("Atomics can only be float/int/uint"),
        }
    }

    fn as_elem(context: &CubeContext) -> Elem {
        match Inner::as_elem(context) {
            Elem::Float(kind) => Elem::AtomicFloat(kind),
            Elem::Int(kind) => Elem::AtomicInt(kind),
            Elem::UInt(kind) => Elem::AtomicUInt(kind),
            _ => unreachable!("Atomics can only be float/int/uint"),
        }
    }

    fn as_elem_native_unchecked() -> Elem {
        match Inner::as_elem_native_unchecked() {
            Elem::Float(kind) => Elem::AtomicFloat(kind),
            Elem::Int(kind) => Elem::AtomicInt(kind),
            Elem::UInt(kind) => Elem::AtomicUInt(kind),
            _ => unreachable!("Atomics can only be float/int/uint"),
        }
    }

    fn size() -> Option<usize> {
        Inner::size()
    }

    fn from_expand_elem(elem: ExpandElement) -> Self::ExpandType {
        ExpandElementTyped::new(elem)
    }
}

impl<Inner: CubePrimitive> ExpandElementBaseInit for Atomic<Inner> {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl<Inner: CubePrimitive> LaunchArgExpand for Atomic<Inner> {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(Self::as_elem_native_unchecked()).into()
    }
}

impl<Inner: Numeric> AtomicTrait for Atomic<Inner> {
    type Primitive = Inner;
}

impl From<AtomicOp> for Operation {
    fn from(value: AtomicOp) -> Self {
        Operation::Atomic(value)
    }
}
