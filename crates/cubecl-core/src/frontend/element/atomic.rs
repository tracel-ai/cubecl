use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand, Numeric,
    Vectorized, I32, I64,
};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, UInt},
    ir::{
        BinaryOperator, CompareAndSwapOperator, Elem, IntKind, Item, Operator, UnaryOperator,
        Vectorization,
    },
    prelude::KernelBuilder,
    unexpanded,
};

/// An atomic type. Represents an shared value that can be operated on atomically.
pub trait Atomic: Sized + CubeType
where
    ExpandElement: From<<Self::Primitive as CubeType>::ExpandType>,
    ExpandElement: From<<Self as CubeType>::ExpandType>,
{
    /// The numeric primitive represented by the atomic wrapper.
    type Primitive: Numeric;

    /// Load the value of the atomic. This operation is only atomic in `wgpu`, CUDA does not support
    /// atomic loads.
    #[allow(unused_variables)]
    fn load(pointer: &Self) -> Self::Primitive {
        unexpanded!()
    }

    /// Store the value of the atomic. This is only actually atomic in `wgpu`, CUDA does not support
    /// atomic stores.
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
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicLoad(UnaryOperator {
            input: *pointer,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_store(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        context.register(Operator::AtomicStore(UnaryOperator {
            input: *value,
            out: *ptr,
        }));
    }

    fn __expand_swap(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicSwap(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
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
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicCompareAndSwap(CompareAndSwapOperator {
            out: *new_var,
            input: *pointer,
            cmp: *cmp,
            val: *value,
        }));
        new_var.into()
    }

    fn __expand_add(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicAdd(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_sub(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicSub(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_max(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicMax(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_min(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicMin(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_and(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicAnd(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_or(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicOr(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
        new_var.into()
    }

    fn __expand_xor(
        context: &mut CubeContext,
        pointer: <Self as CubeType>::ExpandType,
        value: <Self::Primitive as CubeType>::ExpandType,
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicXor(BinaryOperator {
            lhs: *ptr,
            rhs: *value,
            out: *new_var,
        }));
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
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
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
            fn expand(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElementTyped<Self> {
                assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
                builder.scalar(Elem::AtomicInt(IntKind::$inner_type)).into()
            }
        }

        impl Vectorized for $type {
            fn vectorization_factor(&self) -> UInt {
                UInt {
                    val: self.vectorization as u32,
                    vectorization: 1,
                }
            }

            fn vectorize(mut self, factor: UInt) -> Self {
                self.vectorization = factor.vectorization;
                self
            }
        }
    };
}

impl_atomic_int!(AtomicI32, I32, i32);
impl_atomic_int!(AtomicI64, I64, i64);

/// An atomic version of `UInt`. Can only be acted on atomically.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
/// An atomic unsigned int.
pub struct AtomicUInt {
    pub val: u32,
    pub vectorization: u8,
}

impl core::fmt::Debug for AtomicUInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.vectorization == 1 {
            f.write_fmt(format_args!("{}", self.val))
        } else {
            f.write_fmt(format_args!("{}-{}", self.val, self.vectorization))
        }
    }
}

impl CubeType for AtomicUInt {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for AtomicUInt {
    fn as_elem() -> Elem {
        Elem::AtomicUInt
    }
}

impl ExpandElementBaseInit for AtomicUInt {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl LaunchArgExpand for AtomicUInt {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Self> {
        assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
        builder.scalar(Elem::AtomicUInt).into()
    }
}

impl Atomic for AtomicI32 {
    type Primitive = I32;
}
impl Atomic for AtomicI64 {
    type Primitive = I64;
}
impl Atomic for AtomicUInt {
    type Primitive = UInt;
}

impl Vectorized for AtomicUInt {
    fn vectorization_factor(&self) -> UInt {
        UInt {
            val: self.vectorization as u32,
            vectorization: 1,
        }
    }

    fn vectorize(mut self, factor: UInt) -> Self {
        self.vectorization = factor.vectorization;
        self
    }
}
