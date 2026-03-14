use alloc::boxed::Box;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{Branch, ElemType, FloatKind, ManagedVariable, RangeLoop, Variable, VectorSize};
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

pub trait SliceVisibility: Clone + Copy + Send + Sync + 'static {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

pub struct SliceExpand<E: CubePrimitive, IO: SliceVisibility> {
    pub(crate) origin: SliceOriginExpand<E>,
    pub(crate) io: PhantomData<IO>,
    pub(crate) offset: NativeExpand<usize>,
    pub(crate) length: NativeExpand<usize>,
    pub(crate) vector_size: Option<VectorSize>,
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceExpand<E, IO> {
    pub fn __to_raw_parts(&self) -> (Variable, Variable) {
        let expand = match self.origin.clone() {
            SliceOriginExpand::Tensor(expand) => expand.expand,
            SliceOriginExpand::Array(expand) => expand.expand,
            SliceOriginExpand::SharedMemory(expand) => expand.expand,
        };

        (*expand, *self.offset.expand)
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
    pub fn with_vector_size<N2: Size>(&self) -> Slice<Vector<E, N2>, IO> {
        intrinsic!(|scope| {
            let vector_size = N2::__expand_value(scope);
            let (input, offset) = self.__to_raw_parts();
            let mut item = input.ty;

            let current = input.ty.vector_size();
            let mut out = self
                .clone()
                .__expand_downcast_unchecked_method::<Vector<E, N2>>(scope);

            if vector_size == item.vector_size() {
                return out;
            }

            if current < vector_size {
                let ratio = vector_size / current;
                let length = cubecl::frontend::div::expand(scope, self.length, ratio.into());
                let offset = cubecl::frontend::div::expand(scope, self.offset, ratio.into());
                out.length = length;
                out.offset = offset;
            } else {
                let ratio = current / vector_size;
                let length = cubecl::frontend::mul::expand(scope, self.length, ratio.into());
                let offset = cubecl::frontend::mul::expand(scope, self.offset, ratio.into());
                out.length = length;
                out.offset = offset;
            }

            out.vector_size = Some(vector_size);
            out
        })
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    /// Returns the same slice, but with the type reinterpreted as `Vector`.
    /// Preserves existing vector size of the primitive.
    pub fn into_vectorized(&self) -> Slice<Vector<E::Scalar, E::Size>, IO> {
        intrinsic!(|scope| {
            SliceExpand::<Vector<E::Scalar, E::Size>, IO> {
                origin: self.origin.cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size,
            }
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
                origin: self.origin.cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                vector_size: self.vector_size.clone(),
            }
        })
    }
}

#[cube]
impl<E: CubePrimitive> Slice<E, ReadOnly> {
    pub fn as_mut_unchecked(&self) -> Slice<E, ReadWrite> {
        intrinsic!(|scope| {
            SliceExpand::<E, ReadWrite> {
                origin: self.origin,
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
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> SliceExpand<E, IO> {
        Self::__expand_new_expand(scope, origin, start, end)
    }
    pub fn __expand_new_expand(
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> SliceExpand<E, IO> {
        let length = cubecl::frontend::sub::expand(scope, end, start.clone());

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

impl<E: CubePrimitive, IO: SliceVisibility> CubeType for &Slice<E, IO> {
    type ExpandType = SliceExpand<E, IO>;
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeType for &mut Slice<E, IO> {
    type ExpandType = SliceExpand<E, IO>;
}

impl<E: CubePrimitive, IO: SliceVisibility> IntoMut for SliceExpand<E, IO> {
    fn into_mut(self, _scope: &mut cubecl_ir::Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeDebug for SliceExpand<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> Clone for SliceExpand<E, IO> {
    fn clone(&self) -> Self {
        Self {
            origin: self.origin.clone(),
            offset: self.offset.clone(),
            length: self.length.clone(),
            vector_size: self.vector_size,
            io: PhantomData,
        }
    }
}

// TODO: Fix
impl<E: CubePrimitive> SizedContainer for Slice<E, ReadOnly> {
    type Item = E;
}

impl<E: CubePrimitive> Iterable<E> for SliceExpand<E, ReadOnly> {
    fn expand(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <E as CubeType>::ExpandType),
    ) {
        let index_ty = u32::as_type(scope);
        let len: ManagedVariable = self.length.clone().into();

        let mut child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.clone().into();
        let item = index::expand(&mut child, self, index);
        body(&mut child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start: 0usize.into(),
            end: *len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(
        self,
        _scope: &mut Scope,
        _body: impl FnMut(&mut Scope, <E as CubeType>::ExpandType),
    ) {
        unimplemented!("Can't unroll slice iterator")
    }
}
impl<E: CubePrimitive, IO: SliceVisibility> CubeIndex for Slice<E, IO> {
    type Output = E;
    type Idx = usize;

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: NativeExpand<usize>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeIndexExpand for SliceExpand<E, IO> {
    type Output = E::ExpandType;
    type Idx = NativeExpand<usize>;

    fn expand_index(self, scope: &mut Scope, index: NativeExpand<usize>) -> Self::Output {
        self.__expand_read_method(scope, index)
    }
    fn expand_index_unchecked(self, scope: &mut Scope, index: NativeExpand<usize>) -> Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> List<E> for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> ListExpand<E> for SliceExpand<E, IO> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.vector_size,
            true,
        )
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: NativeExpand<usize>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.vector_size,
            false,
        )
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> NativeExpand<usize> {
        Self::__expand_len(scope, self.clone())
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

impl<E: CubePrimitive, IO: SliceVisibility> Vectorized for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> VectorizedExpand for SliceExpand<E, IO> {
    fn vector_size(&self) -> VectorSize {
        self.vector_size
            .unwrap_or_else(|| self.origin.vector_size())
    }
}

impl<E: CubePrimitive> CubeIndexMut for Slice<E, ReadWrite> {
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: NativeExpand<usize>,
        value: NativeExpand<E>,
    ) {
        array.__expand_write_method(scope, index, value)
    }
}

impl<E: CubePrimitive> CubeIndexMutExpand for SliceExpand<E, ReadWrite> {
    fn expand_index_mut(self, scope: &mut Scope, index: NativeExpand<usize>, value: Self::Output) {
        self.__expand_write_method(scope, index, value)
    }
}

impl<E: CubePrimitive> ListMut<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> ListMutExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: NativeExpand<usize>,
        value: NativeExpand<E>,
    ) {
        write_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            value,
            self.vector_size,
        )
    }
}

mod read_offset {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut cubecl::prelude::Scope,
        origin: SliceOriginExpand<E>,
        offset: <usize as cubecl::prelude::CubeType>::ExpandType,
        index: <usize as cubecl::prelude::CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> <E as cubecl::prelude::CubeType>::ExpandType {
        let index = cubecl::frontend::add::expand(scope, offset, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native::<Tensor<E>>(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native::<Array<E>>(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native::<SharedMemory<E>>(scope, expand, index, vector_size, checked)
            }
        }
    }
}

mod write_offset {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut cubecl::prelude::Scope,
        origin: SliceOriginExpand<E>,
        offset: <usize as cubecl::prelude::CubeType>::ExpandType,
        index: <usize as cubecl::prelude::CubeType>::ExpandType,
        value: <E as cubecl::prelude::CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
    ) {
        let index = cubecl::frontend::add::expand(scope, offset, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => expand_index_assign_native::<Tensor<E>>(
                scope,
                expand,
                index,
                value,
                vector_size,
                true,
            ),
            SliceOriginExpand::Array(expand) => expand_index_assign_native::<Array<E>>(
                scope,
                expand,
                index,
                value,
                vector_size,
                false,
            ),
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_assign_native::<SharedMemory<E>>(
                    scope,
                    expand,
                    index,
                    value,
                    vector_size,
                    false,
                )
            }
        }
    }
}
