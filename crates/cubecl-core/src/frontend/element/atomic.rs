use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime, LaunchArgExpand,
    Numeric,
};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement},
    ir::{
        BinaryOperator, CompareAndSwapOperator, Elem, Instruction, IntKind, Item, Operation,
        UIntKind, UnaryOperator,
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
pub trait Atomic: Sized + CubeType
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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
        let new_var = context.create_local_binding(Item::new(Self::Primitive::as_elem()));
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

macro_rules! impl_atomic_int {
    ($type:ident, $inner_type:ident, $primitive:ty) => {
        /// An unsigned atomic integer. Can only be acted on atomically.
        #[allow(clippy::derived_hash_with_manual_eq)]
        #[derive(Clone, Copy, Hash, PartialEq, Eq)]
        pub struct $type {
            pub val: $primitive,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl IntoRuntime for $type {
            fn __expand_runtime_method(
                self,
                _context: &mut CubeContext,
            ) -> ExpandElementTyped<Self> {
                unimplemented!("Atomics don't exist at compile time")
            }
        }

        impl CubePrimitive for $type {
            fn as_elem() -> Elem {
                Elem::AtomicInt(IntKind::$inner_type)
            }
        }

        impl ExpandElementBaseInit for $type {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl LaunchArgExpand for $type {
            type CompilationArg = ();

            fn expand(
                _: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> ExpandElementTyped<Self> {
                builder.scalar(Elem::AtomicInt(IntKind::$inner_type)).into()
            }
        }
    };
}

impl_atomic_int!(AtomicI32, I32, i32);
impl_atomic_int!(AtomicI64, I64, i64);

/// An atomic version of `u32`. Can only be acted on atomically.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
/// An atomic unsigned int.
pub struct AtomicU32 {
    pub val: u32,
}

impl core::fmt::Debug for AtomicU32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.val))
    }
}

impl CubeType for AtomicU32 {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for AtomicU32 {
    fn as_elem() -> Elem {
        Elem::AtomicUInt(UIntKind::U32)
    }
}

impl IntoRuntime for AtomicU32 {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> ExpandElementTyped<Self> {
        unimplemented!("Atomics don't exist at compile time")
    }
}

impl ExpandElementBaseInit for AtomicU32 {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl LaunchArgExpand for AtomicU32 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(Elem::AtomicUInt(UIntKind::U32)).into()
    }
}

impl Atomic for AtomicI32 {
    type Primitive = i32;
}
impl Atomic for AtomicI64 {
    type Primitive = i64;
}
impl Atomic for AtomicU32 {
    type Primitive = u32;
}

impl From<AtomicOp> for Operation {
    fn from(value: AtomicOp) -> Self {
        Operation::Atomic(value)
    }
}
