use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::{
    self as cubecl,
    prelude::{Vectorized, VectorizedExpand},
    unexpanded,
};
use cubecl_ir::{Marker, VariableKind, VectorSize};
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
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = NativeExpand<SharedMemory<T>>;
}

impl<T: CubePrimitive> IntoMut for NativeExpand<Shared<T>> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for Shared<T> {
    type ExpandType = NativeExpand<Shared<T>>;
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new(#[comptime] size: usize) -> Self {
        intrinsic!(|scope| {
            scope
                .create_shared_array(T::as_type(scope), size, None)
                .into()
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
            let var = scope.create_shared(T::as_type(scope));
            NativeExpand::new(var)
        })
    }
}

impl<T: CubePrimitive> Default for Shared<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: CubePrimitive> Shared<T> {
    pub fn __expand_default(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        Self::__expand_new(scope)
    }
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new_aligned(#[comptime] size: usize, #[comptime] alignment: usize) -> SharedMemory<T> {
        intrinsic!(|scope| {
            let var = scope.create_shared_array(T::as_type(scope), size, Some(alignment));
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
    let VariableKind::SharedArray { length, .. } = shared.expand.kind else {
        unreachable!("Kind of shared memory is always shared memory")
    };
    length.into()
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::{IndexMutOperator, IndexOperator, Operator};

    use crate::ir::Instruction;

    use super::*;

    type SharedMemoryExpand<E> = NativeExpand<SharedMemory<E>>;

    #[cube]
    impl<'a, E: CubePrimitive> SharedMemory<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&'a self, i: usize) -> &'a E {
            intrinsic!(|scope| {
                let ty = self.expand.ty;
                let class = self.expand.pointer_class();
                let out = scope.create_local(Type::pointer(ty, class));
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: self.expand,
                        index: i.expand,
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    out,
                ));
                scope.create_kernel_ref(out.into())
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_assign_unchecked(&'a mut self, i: usize) -> &'a mut E {
            intrinsic!(|scope| {
                let ty = self.expand.ty;
                let class = self.expand.pointer_class();
                let out = scope.create_local(Type::pointer(ty, class));
                scope.register(Instruction::new(
                    Operator::UncheckedIndexMut(IndexMutOperator {
                        list: self.expand,
                        index: i.expand,
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    out,
                ));
                scope.create_kernel_ref(out.into())
            })
        }
    }
}

impl<'a, T: CubePrimitive> List<'a, T> for SharedMemory<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: &'a NativeExpand<SharedMemory<T>>,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        this.__expand_index_method(scope, idx)
    }
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

impl<'a, T: CubePrimitive> ListExpand<'a, T> for NativeExpand<SharedMemory<T>> {
    fn __expand_read_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        self.__expand_index_method(scope, idx)
    }
    fn __expand_read_unchecked_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        self.__expand_index_unchecked_method(scope, idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> NativeExpand<usize> {
        Self::__expand_len_method(self, scope)
    }
}

impl<T: CubePrimitive> Vectorized for SharedMemory<T> {}
impl<T: CubePrimitive> VectorizedExpand for NativeExpand<SharedMemory<T>> {
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}

impl<'a, T: CubePrimitive> ListMut<'a, T> for SharedMemory<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: &'a NativeExpand<SharedMemory<T>>,
        idx: NativeExpand<usize>,
    ) -> &'a mut NativeExpand<T> {
        let mut this = this.clone();
        let reference = this.__expand_index_mut_method(scope, idx);
        // SAFETY: This is safe because cloning smem only clones the reference to the global var
        unsafe { core::mem::transmute(reference) }
    }
}

impl<'a, T: CubePrimitive> ListMutExpand<'a, T> for NativeExpand<SharedMemory<T>> {
    fn __expand_write_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a mut NativeExpand<T> {
        let mut this = self.clone();
        let reference = this.__expand_index_mut_method(scope, idx);
        // SAFETY: This is safe because cloning smem only clones the reference to the global var
        unsafe { core::mem::transmute(reference) }
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

impl<T: CubePrimitive> CubeDeref for SharedExpand<T> {
    type Target = T::ExpandType;

    fn __expand_deref_method(&self, _: &mut Scope) -> Self::Target {
        unsafe { self.as_type_ref_unchecked::<T>().clone() }
    }
}

impl<T: CubePrimitive> Assign<NativeExpand<T>> for SharedExpand<T> {
    fn __expand_assign_method(&mut self, scope: &mut Scope, value: NativeExpand<T>) {
        self.__expand_deref_method(scope)
            .__expand_assign_method(scope, value);
    }

    fn init_mut(&self, _: &mut Scope) -> Self {
        self.clone()
    }
}
