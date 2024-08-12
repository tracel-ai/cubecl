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

pub trait Atomic: Sized + CubeType
where
    ExpandElement: From<<Self::Primitive as CubeType>::ExpandType>,
    ExpandElement: From<<Self as CubeType>::ExpandType>,
{
    type Primitive: Numeric;

    #[allow(unused_variables)]
    fn load(pointer: &Self) -> Self::Primitive {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn store(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn compare_and_swap(
        pointer: &Self,
        cmp: Self::Primitive,
        value: Self::Primitive,
    ) -> Self::Primitive {
        unexpanded!()
    }

    #[allow(unused_variables)]
    fn add(pointer: &Self, value: Self::Primitive) -> Self::Primitive {
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
    ) -> <Self::Primitive as CubeType>::ExpandType {
        let ptr: ExpandElement = pointer.into();
        let value: ExpandElement = value.into();
        let new_var = context.create_local(Item::new(Self::Primitive::as_elem()));
        context.register(Operator::AtomicStore(BinaryOperator {
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
}

macro_rules! impl_atomic_int {
    ($type:ident, $inner_type:ident, $primitive:ty) => {
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
