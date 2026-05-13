use core::ops::{Deref, DerefMut};

use crate::{
    self as cubecl,
    frontend::container::slice,
    prelude::{Vectorized, VectorizedExpand},
    unexpanded,
};
use cubecl_ir::{
    AddressSpace, AggregateKind, BoundsCheckMetadata, Marker, MetadataKind, SliceMetadata,
    VariableKind, VectorSize,
};
use cubecl_macros::{cube, intrinsic};

use crate::{
    frontend::{CubePrimitive, CubeType, IntoMut, NativeExpand},
    ir::Scope,
    prelude::*,
};

pub type SharedMemory<T> = Shared<[T]>;
pub type SharedMemoryExpand<T> = NativeExpand<Shared<[T]>>;
pub type SharedExpand<T> = NativeExpand<Shared<T>>;

pub struct Shared<E: NativeCubeType + ?Sized> {
    _val: *mut E,
}

// Treat it as a shared smart pointer
impl<E: NativeCubeType + ?Sized> Copy for Shared<E> {}
impl<E: NativeCubeType + ?Sized> Clone for Shared<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: NativeCubeType + ?Sized> IntoMut for NativeExpand<Shared<T>> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<T: NativeCubeType + ?Sized> CubeType for Shared<T> {
    type ExpandType = NativeExpand<Shared<T>>;
}

impl<T: NativeCubeType + ?Sized> AsMutExpand for SharedExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

#[cube]
impl<T: CubePrimitive> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new(#[comptime] size: usize) -> SharedMemory<T> {
        intrinsic!(|scope| {
            let ty = Type::array(T::__expand_as_type(scope), size, AddressSpace::Shared);
            let buffer = scope.create_shared(ty, None);
            let slice = slice::from_raw_parts::<T>(
                scope,
                buffer,
                0usize.into_expand(scope),
                size.into_expand(scope),
            );
            slice.expand.into()
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        intrinsic!(|_| len_static(&self))
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

    #[allow(unused_variables)]
    pub fn new_array(#[comptime] size: usize) -> SharedMemory<T> {
        intrinsic!(|scope| {
            let ty = Type::array(T::__expand_as_type(scope), size, AddressSpace::Shared);
            let buffer = scope.create_shared(ty, None);
            let slice = slice::from_raw_parts::<T>(
                scope,
                buffer,
                0usize.into_expand(scope),
                size.into_expand(scope),
            );
            slice.expand.into()
        })
    }

    #[allow(unused_variables)]
    pub fn new_aligned_array(
        #[comptime] size: usize,
        #[comptime] alignment: usize,
    ) -> SharedMemory<T> {
        intrinsic!(|scope| {
            let ty = Type::array(T::__expand_as_type(scope), size, AddressSpace::Shared);
            let buffer = scope.create_shared(ty, Some(alignment));
            let slice = slice::from_raw_parts::<T>(
                scope,
                buffer,
                0usize.into_expand(scope),
                size.into_expand(scope),
            );
            slice.expand.into()
        })
    }
}

#[cube]
impl<T: NativeCubeType + ?Sized> Shared<T> {
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
impl<T: NativeCubeType + ?Sized> Shared<T> {
    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(&self) {
        intrinsic!(|scope| {
            let var = match self.expand.kind {
                VariableKind::Aggregate { aggregate_kind, .. } => match aggregate_kind {
                    AggregateKind::Ptr(MetadataKind::Slice) => {
                        scope.extract_field(self.expand, self.expand.ty, SliceMetadata::LIST)
                    }
                    AggregateKind::Ptr(MetadataKind::BoundsCheck) => scope.extract_field(
                        self.expand,
                        self.expand.ty,
                        BoundsCheckMetadata::POINTER,
                    ),
                },
                _ => self.expand,
            };
            scope.register(Marker::Free(var))
        })
    }
}

fn len_static<T: CubePrimitive>(shared: &NativeExpand<SharedMemory<T>>) -> NativeExpand<usize> {
    let Type::Array(_, length, _) = shared.expand.ty else {
        unreachable!("Kind of shared memory is always shared memory")
    };
    length.into()
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

impl<T: NativeCubeType + ?Sized> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}
impl<T: NativeCubeType + ?Sized> DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: NativeCubeType + ?Sized> Deref for SharedExpand<T> {
    type Target = T::ExpandType;

    fn deref(&self) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked() }
    }
}
impl<T: NativeCubeType + ?Sized> DerefMut for SharedExpand<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked() }
    }
}

impl<T: NativeCubeType + ?Sized> AsDerefExpand for SharedExpand<T> {
    type Target = T::ExpandType;

    fn __expand_as_deref_method(&self, _: &Scope) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked::<T>() }
    }
}
impl<T: NativeCubeType + ?Sized> AsDerefMutExpand for SharedExpand<T> {
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
