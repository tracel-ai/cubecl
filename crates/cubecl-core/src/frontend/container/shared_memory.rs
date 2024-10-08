use std::{marker::PhantomData, num::NonZero};

use crate::{
    frontend::{
        indexation::Index, CubeContext, CubePrimitive, CubeType, ExpandElementTyped, Init,
        IntoRuntime,
    },
    ir::Item,
    prelude::Line,
};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubePrimitive> Init for ExpandElementTyped<SharedMemory<T>> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<T: CubePrimitive> IntoRuntime for SharedMemory<T> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> ExpandElementTyped<Self> {
        unimplemented!("Shared memory can't exist at comptime");
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElementTyped<SharedMemory<T>>;
}

impl<T: CubePrimitive + Clone> SharedMemory<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_lined<S: Index>(_size: S, _vectorization_factor: u32) -> SharedMemory<Line<T>> {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_new_lined(
        context: &mut CubeContext,
        size: ExpandElementTyped<u32>,
        vectorization_factor: u32,
    ) -> <SharedMemory<Line<T>> as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = context.create_shared(
            Item::vectorized(T::as_elem(), NonZero::new(vectorization_factor as u8)),
            size,
        );
        ExpandElementTyped::new(var)
    }
    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: u32) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_vectorized(
        context: &mut CubeContext,
        size: ExpandElementTyped<u32>,
        vectorization_factor: u32,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = context.create_shared(
            Item::vectorized(T::as_elem(), NonZero::new(vectorization_factor as u8)),
            size,
        );
        ExpandElementTyped::new(var)
    }

    pub fn __expand_new(
        context: &mut CubeContext,
        size: ExpandElementTyped<u32>,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = context.create_shared(Item::new(T::as_elem()), size);
        ExpandElementTyped::new(var)
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use crate::{
        ir::{BinaryOperator, Operator},
        prelude::{CubeIndex, CubeIndexMut},
        unexpanded,
    };

    use super::*;

    impl<E: CubePrimitive> SharedMemory<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_assign_unchecked<I: Index>(&mut self, _i: I, _value: E)
        where
            Self: CubeIndexMut<I>,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<SharedMemory<E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item());
            context.register(Operator::UncheckedIndex(BinaryOperator {
                out: *out,
                lhs: *self.expand,
                rhs: i.expand.consume(),
            }));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            context.register(Operator::UncheckedIndexAssign(BinaryOperator {
                out: *self.expand,
                lhs: i.expand.consume(),
                rhs: value.expand.consume(),
            }));
        }
    }
}
