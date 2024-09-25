use std::{marker::PhantomData, num::NonZero};

use crate::{
    frontend::{indexation::Index, CubeContext, CubePrimitive, CubeType},
    ir::Item,
};

use super::{ExpandElementTyped, Init, IntoRuntime, Line};

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
