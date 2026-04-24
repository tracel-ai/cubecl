use alloc::boxed::Box;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{Branch, ElemType, FloatKind, RangeLoop, Variable, VectorSize};
use cubecl_macros::intrinsic;

#[derive(Clone, Copy)]
pub struct ReadOnly;
#[derive(Clone, Copy)]
pub struct ReadWrite;

/// A read-only contiguous list of elements
///
/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
#[derive(Clone, Copy)]
pub struct Slice<E: CubePrimitive, IO: SliceVisibility = ReadOnly> {
    _e: PhantomData<E>,
    _io: PhantomData<IO>,
    _offset: PhantomData<usize>,
    length: usize,
}

#[derive(CubeType)]
pub enum SliceOrigin<E: CubePrimitive> {
    Tensor(Tensor<E>),
    Array(Array<E>),
    SharedMemory(SharedMemory<E>),
}

impl<E: CubePrimitive> Copy for SliceOriginExpand<E> {}
impl<E: CubePrimitive> Clone for SliceOriginExpand<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: CubePrimitive> SliceOriginExpand<E> {
    pub fn vector_size(&self) -> VectorSize {
        match self {
            SliceOriginExpand::Tensor(t) => t.vector_size(),
            SliceOriginExpand::Array(t) => t.vector_size(),
            SliceOriginExpand::SharedMemory(t) => t.vector_size(),
        }
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> Iterator for Slice<E, IO> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<'a, E: CubePrimitive, IO: SliceVisibility> Iterator for &'a Slice<E, IO> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

pub trait SliceVisibility: Clone + Copy + Send + Sync + 'static {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

#[derive(Clone, Copy)]
pub struct SliceExpand<E: CubePrimitive, IO: SliceVisibility> {
    pub(crate) origin: SliceOriginExpand<E>,
    pub(crate) io: PhantomData<IO>,
    pub(crate) offset: NativeExpand<usize>,
    pub(crate) length: NativeExpand<usize>,
    pub(crate) vector_size: Option<VectorSize>,
}

impl<E: CubePrimitive, IO: SliceVisibility> ExpandTypeClone for SliceExpand<E, IO> {
    fn clone_unchecked(&self) -> Self {
        SliceExpand {
            origin: self.origin.clone_unchecked(),
            io: self.io,
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        }
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceExpand<E, IO> {
    pub fn __to_raw_parts(&self) -> (Variable, Variable) {
        let expand = match self.origin {
            SliceOriginExpand::Tensor(expand) => expand.expand,
            SliceOriginExpand::Array(expand) => expand.expand,
            SliceOriginExpand::SharedMemory(expand) => expand.expand,
        };

        (expand, self.offset.expand)
    }
}

#[cube]
impl<E: Scalar, N: Size, IO: SliceVisibility> Slice<Vector<E, N>, IO> {
    /// Reinterprets how items are loaded and stored in memory.slicebase
    ///
    /// # Warning
    ///
    /// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
    #[allow(unused_variables)]
    pub fn with_vector_size<N2: Size>(&self) -> &Slice<Vector<E, N2>, IO> {
        intrinsic!(|scope| {
            let slice = self.clone().with_vector_size_inner::<N2>(scope);
            scope.create_kernel_ref(slice)
        })
    }

    /// Reinterprets how items are loaded and stored in memory.slicebase
    ///
    /// # Warning
    ///
    /// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
    #[allow(unused_variables)]
    pub fn with_vector_size_mut<N2: Size>(&mut self) -> &mut Slice<Vector<E, N2>, IO> {
        intrinsic!(|scope| {
            let slice = self.clone().with_vector_size_inner::<N2>(scope);
            scope.create_kernel_ref(slice)
        })
    }
}

impl<E: Scalar, N: Size, IO: SliceVisibility> SliceExpand<Vector<E, N>, IO> {
    fn with_vector_size_inner<N2: Size>(self, scope: &Scope) -> SliceExpand<Vector<E, N2>, IO> {
        let vector_size = N2::__expand_value(scope);
        let (input, _) = self.__to_raw_parts();
        let item = input.ty;

        let current = input.ty.vector_size();
        let mut out = self
            .clone()
            .__expand_downcast_unchecked_method::<Vector<E, N2>>(scope);

        if vector_size == item.vector_size() {
            return out;
        }

        if current < vector_size {
            let ratio = vector_size / current;
            let length = self.length.__expand_div_method(scope, ratio.into());
            let offset = self.offset.__expand_div_method(scope, ratio.into());
            out.length = length;
            out.offset = offset;
        } else {
            let ratio = current / vector_size;
            let length = self.length.__expand_mul_method(scope, ratio.into());
            let offset = self.offset.__expand_mul_method(scope, ratio.into());
            out.length = length;
            out.offset = offset;
        }

        out.vector_size = Some(vector_size);
        out
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    /// Returns the same slice, but with the type reinterpreted as `Vector`.
    /// Preserves existing vector size of the primitive.
    pub fn as_vectorized(&self) -> &Slice<Vector<E::Scalar, E::Size>, IO> {
        intrinsic!(|scope| {
            let slice = SliceExpand::<Vector<E::Scalar, E::Size>, IO> {
                origin: self.origin.clone().cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size,
            };
            scope.create_kernel_ref(slice)
        })
    }

    /// Returns the same slice, but with the type reinterpreted as `Vector`.
    /// Preserves existing vector size of the primitive.
    pub fn as_vectorized_mut(&mut self) -> &mut Slice<Vector<E::Scalar, E::Size>, IO> {
        intrinsic!(|scope| {
            let slice = SliceExpand::<Vector<E::Scalar, E::Size>, IO> {
                origin: self.origin.clone().cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size,
            };
            scope.create_kernel_ref(slice)
        })
    }

    /// Downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// This function should only be used to satisfy the Rust type system, when two generic
    /// types are supposed to be the same.
    pub fn downcast<T: CubePrimitive>(&self) -> Slice<T, IO> {
        intrinsic!(|scope| {
            if T::as_type(scope) != E::as_type(scope) && !is_tf32::<E, T>(scope) {
                let elems = [T::as_type(scope).elem_type(), E::as_type(scope).elem_type()];
                let is_flex32_cast = elems.contains(&ElemType::Float(FloatKind::F32))
                    && elems.contains(&ElemType::Float(FloatKind::Flex32));

                if !is_flex32_cast {
                    panic!("Downcast should only be used to satisfy the Rust type system.")
                }
            }

            unsafe { self.__expand_downcast_unchecked_method(scope) }
        })
    }

    /// Unsafely downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// # Safety
    /// This function converts unsafely, and should only be used for temporary storage with a dummy
    /// type (i.e. `ReinterpretSlice`)
    pub unsafe fn downcast_unchecked<T: CubePrimitive>(&self) -> Slice<T, IO> {
        intrinsic!(|scope| {
            SliceExpand::<T, IO> {
                origin: self.origin.clone().cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size.clone(),
            }
        })
    }

    pub fn as_ptr(&self) -> &E {
        intrinsic!(|scope| {
            as_ptr::expand(scope, &self.origin, self.offset, self.vector_size, true)
        })
    }

    pub fn as_ptr_mut(&mut self) -> &mut E {
        intrinsic!(|scope| {
            as_ptr_mut::expand(scope, &mut self.origin, self.offset, self.vector_size, true)
        })
    }
}

#[cube]
impl<E: CubePrimitive> Slice<E, ReadOnly> {
    pub fn as_mut_unchecked(&self) -> Slice<E, ReadWrite> {
        intrinsic!(|scope| {
            SliceExpand::<E, ReadWrite> {
                origin: self.origin.clone(),
                io: PhantomData,
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size.clone(),
            }
        })
    }
}

impl<E: CubePrimitive> SliceOriginExpand<E> {
    fn cast_unchecked<T: CubePrimitive>(self) -> SliceOriginExpand<T> {
        match self {
            SliceOriginExpand::Tensor(expand) => {
                SliceOriginExpand::<T>::Tensor(expand.expand.into())
            }
            SliceOriginExpand::Array(expand) => SliceOriginExpand::<T>::Array(expand.expand.into()),
            SliceOriginExpand::SharedMemory(expand) => {
                SliceOriginExpand::<T>::SharedMemory(expand.expand.into())
            }
        }
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    pub fn new(_origin: SliceOrigin<E>, _offset: usize, _length: usize) -> Self {
        unexpanded!()
    }
    pub fn __expand_new(
        scope: &Scope,
        origin: SliceOriginExpand<E>,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> SliceExpand<E, IO> {
        Self::__expand_new_expand(scope, origin, start, end)
    }
    pub fn __expand_new_expand(
        scope: &Scope,
        origin: SliceOriginExpand<E>,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> SliceExpand<E, IO> {
        let length = cubecl::frontend::sub::expand(scope, end, start);

        SliceExpand::<E, IO> {
            origin,
            io: PhantomData,
            offset: start,
            length,
            vector_size: None,
        }
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    /// Get the length of the slice.
    pub fn len(&self) -> usize {
        self.length
    }
    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeType for Slice<E, IO> {
    type ExpandType = SliceExpand<E, IO>;
}

impl<E: CubePrimitive, IO: SliceVisibility> AsRefExpand for SliceExpand<E, IO> {
    fn __expand_as_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<E: CubePrimitive, IO: SliceVisibility> AsMutExpand for SliceExpand<E, IO> {
    fn __expand_as_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> IntoExpand for SliceExpand<E, IO> {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}
impl<E: CubePrimitive, IO: SliceVisibility> IntoMut for SliceExpand<E, IO> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeDebug for SliceExpand<E, IO> {}

// TODO: Fix
impl<E: CubePrimitive> SizedContainer for Slice<E, ReadOnly> {
    type Item = E;
}

impl<E: CubePrimitive> Iterable for SliceExpand<E, ReadOnly> {
    type Item = E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = u32::as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a SliceExpand<E, ReadOnly> {
    type Item = &'a E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = u32::as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
        let item = self.__expand_index_method(&child, index);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a mut SliceExpand<E, ReadWrite> {
    type Item = &'a mut E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = u32::as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
        let item = self.__expand_index_mut_method(&child, index);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeIndex for Slice<E, IO> {
    type Output = E;
    type Idx = usize;
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeIndexExpand for SliceExpand<E, IO> {
    type Output = E::ExpandType;
    type Idx = NativeExpand<usize>;

    fn __expand_index_method(&self, scope: &Scope, index: NativeExpand<usize>) -> &Self::Output {
        self.__expand_read_method(scope, index)
    }
    fn __expand_index_unchecked_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> List<E> for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> ListExpand<E> for SliceExpand<E, IO> {
    fn __expand_read_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &<E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            &self.origin,
            self.offset,
            index,
            self.vector_size,
            true,
        )
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &<E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            &self.origin,
            self.offset,
            index,
            self.vector_size,
            false,
        )
    }

    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        Self::__expand_len(scope, self)
    }
}

impl<T: CubePrimitive, IO: SliceVisibility> Deref for Slice<T, IO> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for Slice<T, ReadWrite> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive, IO: SliceVisibility> DerefExpand for SliceExpand<T, IO> {
    type Target = Self;

    fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
        *self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> Vectorized for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> VectorizedExpand for SliceExpand<E, IO> {
    fn vector_size(&self) -> VectorSize {
        self.vector_size
            .unwrap_or_else(|| self.origin.vector_size())
    }
}

impl<E: CubePrimitive> CubeIndexMut for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> CubeIndexMutExpand for SliceExpand<E, ReadWrite> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut Self::Output {
        self.__expand_write_method(scope, index)
    }
}

impl<E: CubePrimitive> ListMut<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> ListMutExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut NativeExpand<E> {
        let mut origin = self.origin;
        let reference =
            write_offset::expand::<E>(scope, &mut origin, self.offset, index, self.vector_size);
        // Safety: Cloning origin only clones the reference, so this is safe
        unsafe { core::mem::transmute(reference) }
    }
}

mod as_ptr {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
        }
    }
}

mod as_ptr_mut {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a mut SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a mut <E as cubecl::prelude::CubeType>::ExpandType {
        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
        }
    }
}

mod read_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        index: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        let index = offset.__expand_add_method(scope, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
        }
    }
}

mod write_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a mut SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        index: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
    ) -> &'a mut E::ExpandType {
        let index = offset.__expand_add_method(scope, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, true)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, false)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, false)
            }
        }
    }
}
