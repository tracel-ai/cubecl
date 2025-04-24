use std::{marker::PhantomData, num::NonZero};

use crate as cubecl;
use cubecl_macros::{cube, intrinsic};

use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementTyped, Init, indexation::Index},
    ir::{Item, Scope},
    prelude::{
        Line, List, ListExpand, ListMut, ListMutExpand, index, index_assign, index_unchecked,
    },
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

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new_aligned(
        #[comptime] size: u32,
        #[comptime] vectorization_factor: u32,
        #[comptime] alignment: u32,
    ) -> SharedMemory<Line<T>> {
        intrinsic!(|scope| {
            let var = scope.create_shared(
                Item::vectorized(T::as_elem(scope), NonZero::new(vectorization_factor as u8)),
                size,
                Some(alignment),
            );
            ExpandElementTyped::new(var)
        })
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::Operator;

    use crate::ir::{BinaryOperator, Instruction};

    use super::*;

    type SharedMemoryExpand<E> = ExpandElementTyped<SharedMemory<E>>;

    #[cube]
    impl<E: CubePrimitive> SharedMemory<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: u32) -> &E {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.item);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(BinaryOperator {
                        lhs: *self.expand,
                        rhs: i.expand.consume(),
                    }),
                    *out,
                ));
                out.into()
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_assign_unchecked(&mut self, i: u32, value: E) {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(BinaryOperator {
                        lhs: i.expand.consume(),
                        rhs: value.expand.consume(),
                    }),
                    *self.expand,
                ));
            })
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
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
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
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
