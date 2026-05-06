use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::{
    self as cubecl,
    prelude::{Vectorized, VectorizedExpand},
    unexpanded,
};
use cubecl_ir::{Marker, VectorSize};
use cubecl_macros::{cube, intrinsic};

use crate::{
    frontend::{CubePrimitive, CubeType, IntoMut, NativeExpand},
    ir::Scope,
    prelude::*,
};

pub type SharedMemoryExpand<T> = NativeExpand<SharedMemory<T>>;
pub type SharedExpand<T> = NativeExpand<Shared<T>>;

#[derive(Clone, Copy)]
pub struct Shared<E: CubePrimitive> {
    _val: PhantomData<E>,
}

#[derive(Clone, Copy)]
pub struct SharedMemory<E: CubePrimitive> {
    _val: PhantomData<E>,
}

impl<T: CubePrimitive> IntoMut for NativeExpand<SharedMemory<T>> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = NativeExpand<SharedMemory<T>>;
}

impl<T: CubePrimitive> AsMutExpand for SharedMemoryExpand<T> {
    fn __expand_as_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<T: CubePrimitive> IntoMut for NativeExpand<Shared<T>> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for Shared<T> {
    type ExpandType = NativeExpand<Shared<T>>;
}

impl<T: CubePrimitive> AsMutExpand for SharedExpand<T> {
    fn __expand_as_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new(#[comptime] size: usize) -> Self {
        intrinsic!(|scope| {
            let ty = Type::array(T::__expand_as_type(scope), size);
            scope.create_shared(ty, None).into()
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        intrinsic!(|_| len_static(&self))
    }

    pub fn buffer_len(&self) -> usize {
        self.len()
    }
}

#[cube]
impl<T: CubePrimitive> Shared<T> {
    pub fn new() -> Self {
        intrinsic!(|scope| {
            let var = scope.create_shared(T::__expand_as_type(scope), None);
            NativeExpand::new(var)
        })
    }

    pub fn inner_ref(&self) -> &T {
        intrinsic!(|scope| { unsafe { self.as_type_ref_unchecked() } })
    }

    pub fn inner_mut(&mut self) -> &mut T {
        intrinsic!(|scope| { unsafe { self.as_type_mut_unchecked() } })
    }
}

impl<T: CubePrimitive> Default for Shared<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: CubePrimitive> Shared<T> {
    pub fn __expand_default(scope: &Scope) -> <Self as CubeType>::ExpandType {
        Self::__expand_new(scope)
    }
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new_aligned(#[comptime] size: usize, #[comptime] alignment: usize) -> SharedMemory<T> {
        intrinsic!(|scope| {
            let ty = Type::array(T::__expand_as_type(scope), size);
            let var = scope.create_shared(ty, Some(alignment));
            NativeExpand::new(var)
        })
    }

    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(self) {
        intrinsic!(|scope| { scope.register(Marker::Free(self.expand)) })
    }
}

fn len_static<T: CubePrimitive>(shared: &NativeExpand<SharedMemory<T>>) -> NativeExpand<usize> {
    let Type::Array(_, length) = shared.expand.ty else {
        unreachable!("Kind of shared memory is always shared memory")
    };
    length.into()
}

impl<T: CubePrimitive> Deref for SharedMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for SharedMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> List<T> for SharedMemory<T> {}
impl<T: CubePrimitive> ListExpand<T> for NativeExpand<SharedMemory<T>> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        Self::__expand_len_method(self, scope)
    }
}

impl<T: CubePrimitive> Vectorized for SharedMemory<T> {}
impl<T: CubePrimitive> VectorizedExpand for NativeExpand<SharedMemory<T>> {
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}

impl<T: CubePrimitive> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}
impl<T: CubePrimitive> DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefExpand for SharedExpand<T> {
    type Target = T::ExpandType;

    fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
        unsafe { *self.as_type_ref_unchecked::<T>() }
    }
}

impl<T: CubePrimitive> AsDerefExpand for SharedExpand<T> {
    type Target = T::ExpandType;

    fn __expand_as_deref_method(&self, _: &Scope) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked::<T>() }
    }
}
impl<T: CubePrimitive> AsDerefMutExpand for SharedExpand<T> {
    fn __expand_as_deref_mut_method(&mut self, _: &Scope) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked::<T>() }
    }
}

impl<T: CubePrimitive> Assign<NativeExpand<T>> for SharedExpand<T> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: NativeExpand<T>) {
        self.__expand_deref_method(scope)
            .__expand_assign_method(scope, value);
    }

    fn init_mut(&self, _: &Scope) -> Self {
        *self
    }
}
