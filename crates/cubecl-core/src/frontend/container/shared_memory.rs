use std::{marker::PhantomData, num::NonZero};

use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementTyped, Init, indexation::Index},
    ir::{Item, Scope},
    prelude::{Line, List, ListExpand, ListMut, ListMutExpand, index, index_assign},
};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubePrimitive> Init for ExpandElementTyped<SharedMemory<T>> {
    fn init(self, _scope: &mut Scope) -> Self {
        self
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

    pub fn new_aligned<S: Index>(
        _size: S,
        _vectorization_factor: u32,
        _alignment: u32,
    ) -> SharedMemory<Line<T>> {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_new_lined(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
        vectorization_factor: u32,
    ) -> <SharedMemory<Line<T>> as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = scope.create_shared(
            Item::vectorized(T::as_elem(scope), NonZero::new(vectorization_factor as u8)),
            size,
            None,
        );
        ExpandElementTyped::new(var)
    }

    pub fn __expand_new_aligned(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
        vectorization_factor: u32,
        alignment: u32,
    ) -> <SharedMemory<Line<T>> as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = scope.create_shared(
            Item::vectorized(T::as_elem(scope), NonZero::new(vectorization_factor as u8)),
            size,
            Some(alignment),
        );
        ExpandElementTyped::new(var)
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: u32) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_vectorized(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
        vectorization_factor: u32,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = scope.create_shared(
            Item::vectorized(T::as_elem(scope), NonZero::new(vectorization_factor as u8)),
            size,
            None,
        );
        ExpandElementTyped::new(var)
    }

    pub fn __expand_new(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = scope.create_shared(Item::new(T::as_elem(scope)), size, None);
        ExpandElementTyped::new(var)
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::Operator;

    use crate::{
        ir::{BinaryOperator, Instruction},
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
            scope: &mut Scope,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = scope.create_local(self.expand.item);
            scope.register(Instruction::new(
                Operator::UncheckedIndex(BinaryOperator {
                    lhs: *self.expand,
                    rhs: i.expand.consume(),
                }),
                *out,
            ));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            scope: &mut Scope,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            scope.register(Instruction::new(
                Operator::UncheckedIndexAssign(BinaryOperator {
                    lhs: i.expand.consume(),
                    rhs: value.expand.consume(),
                }),
                *self.expand,
            ));
        }
    }
}

impl<T: CubePrimitive> List<T> for SharedMemory<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: ExpandElementTyped<SharedMemory<T>>,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_read_method(
        self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self, idx)
    }
}

impl<T: CubePrimitive> ListMut<T> for SharedMemory<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<SharedMemory<T>>,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_write_method(
        self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self, idx, value);
    }
}
