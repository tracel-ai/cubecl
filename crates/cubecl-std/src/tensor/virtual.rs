use alloc::sync::Arc;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};
use std::ops::{Deref, DerefMut};

use crate::tensor::{
    ViewExpand,
    layout::{
        Coordinates, Coords1d, Layout, VirtualLayout, VirtualLayoutExpand, simple::SimpleLayout,
    },
    view::View,
};

/// Tensor representation that is decoupled from how the tensor is stored.
#[derive(Clone)]
pub struct VirtualTensor<E: Numeric, N: Size, IO = ReadOnly> {
    // state: Arc<dyn VirtualTensorOperations<E>>,
    _e: PhantomData<E>,
    _n: PhantomData<N>,
    _p: PhantomData<IO>,
}

/// Expand type for [`VirtualTensor`].
#[derive(Clone)]
pub struct VirtualTensorExpand<E: Numeric, N: Size, IO> {
    state: Arc<dyn VirtualTensorOperationsExpand<E, N>>,
    _p: PhantomData<IO>,
}

impl<'a, E: Numeric, N: Size, IO: Clone> List<'a, Vector<E, N>> for VirtualTensor<E, N, IO> {
    fn __expand_read(
        scope: &Scope,
        this: &'a VirtualTensorExpand<E, N, IO>,
        index: <usize as CubeType>::ExpandType,
    ) -> &'a <Vector<E, N> as CubeType>::ExpandType {
        this.__expand_read_method(scope, index)
    }
}

impl<T: Numeric, N: Size, IO: Clone> Deref for VirtualTensor<T, N, IO> {
    type Target = [Vector<T, N>];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: Numeric, N: Size> DerefMut for VirtualTensor<T, N, ReadWrite> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<'a, E: Numeric, N: Size, IO: Clone> ListExpand<'a, Vector<E, N>>
    for VirtualTensorExpand<E, N, IO>
{
    fn __expand_read_method(
        &'a self,
        scope: &Scope,
        index: <usize as CubeType>::ExpandType,
    ) -> &'a <Vector<E, N> as CubeType>::ExpandType {
        self.state.__expand_read_method(scope, index)
    }

    fn __expand_read_unchecked_method(
        &'a self,
        _scope: &Scope,
        _index: NativeExpand<usize>,
    ) -> &'a <Vector<E, N> as CubeType>::ExpandType {
        todo!("VirtualTensor don't support read unchecked yet");
    }

    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.state.clone().__expand_len_method(scope)
    }
}

impl<E: Numeric, N: Size, IO: Clone> Vectorized for VirtualTensor<E, N, IO> {}
impl<E: Numeric, N: Size, IO: Clone> VectorizedExpand for VirtualTensorExpand<E, N, IO> {
    fn vector_size(&self) -> VectorSize {
        self.state.clone().vector_size()
    }
}

impl<'a, E: Numeric, N: Size, IO: Clone> SliceOperator<'a, Vector<E, N>>
    for VirtualTensor<E, N, IO>
{
}
impl<'a, E: Numeric, N: Size, IO: Clone> SliceOperatorExpand<'a, Vector<E, N>>
    for VirtualTensorExpand<E, N, IO>
{
    fn __expand_slice_method(
        &'a self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a SliceExpand<Vector<E, N>, ReadOnly> {
        self.state
            .clone()
            .__expand_read_window_method(scope, start, end)
    }

    fn __expand_to_slice_method(
        &'a self,
        scope: &Scope,
    ) -> &'a SliceExpand<Vector<E, N>, ReadOnly> {
        let end = self.clone().__expand_buffer_len_method(scope);
        self.state
            .clone()
            .__expand_read_window_method(scope, 0.into(), end)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric, N: Size, IO: Clone> VirtualTensor<E, N, IO> {
    pub fn as_tensor_map(&self) -> Option<TensorMap<E, Tiled>> {
        unexpanded!()
    }
    pub fn as_slice(&self, start: usize, end: usize) -> Slice<Vector<E, N>> {
        unexpanded!();
    }
    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: usize) -> usize {
        unexpanded!();
    }
    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: usize) -> usize {
        unexpanded!();
    }
    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        unexpanded!();
    }

    pub fn buffer_len(&self) -> usize {
        unexpanded!();
    }

    pub fn __expand_as_tensor_map(
        context: &Scope,
        this: &<Self as CubeType>::ExpandType,
    ) -> <ComptimeOption<TensorMap<E, Tiled>> as CubeType>::ExpandType {
        this.__expand_as_tensor_map_method(context)
    }
    pub fn __expand_as_slice<'a>(
        context: &Scope,
        this: &'a <Self as CubeType>::ExpandType,
        start: <usize as CubeType>::ExpandType,
        end: <usize as CubeType>::ExpandType,
    ) -> &'a <Slice<Vector<E, N>> as CubeType>::ExpandType {
        this.__expand_as_slice_method(context, start, end)
    }
    pub fn __expand_shape(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_shape_method(scope, axis)
    }
    pub fn __expand_stride(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_stride_method(scope, axis)
    }
    pub fn __expand_rank(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_rank_method(scope)
    }
    pub fn __expand_buffer_len(
        scope: &Scope,
        this: &<Self as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_buffer_len_method(scope)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric, N: Size, IO: Clone> VirtualTensorExpand<E, N, IO> {
    pub fn __expand_as_tensor_map_method(
        &self,
        context: &Scope,
    ) -> <ComptimeOption<TensorMap<E, Tiled>> as CubeType>::ExpandType {
        self.state.clone().__expand_as_tensor_map_method(context)
    }

    pub fn __expand_as_slice_method(
        &self,
        context: &Scope,
        start: <usize as CubeType>::ExpandType,
        end: <usize as CubeType>::ExpandType,
    ) -> &<Slice<Vector<E, N>> as CubeType>::ExpandType {
        self.state.__expand_read_window_method(context, start, end)
    }

    pub fn __expand_shape_method(
        &self,
        scope: &Scope,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_shape_method(scope, _arg_0.into())
    }

    pub fn __expand_stride_method(
        &self,
        scope: &Scope,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_stride_method(scope, _arg_0.into())
    }

    pub fn __expand_rank_method(&self, scope: &Scope) -> <usize as CubeType>::ExpandType {
        self.state.__expand_rank_method(scope)
    }

    pub fn __expand_buffer_len_method(&self, scope: &Scope) -> <usize as CubeType>::ExpandType {
        self.state.__expand_buffer_len_method(scope)
    }

    pub fn __expand_read<'a>(
        scope: &Scope,
        this: &'a Self,
        index: <usize as CubeType>::ExpandType,
    ) -> &'a <Vector<E, N> as CubeType>::ExpandType {
        VirtualTensor::<E, N, IO>::__expand_read(scope, this, index)
    }

    pub fn __expand_shape(
        scope: &Scope,
        this: &Self,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, N, IO>::__expand_shape(scope, this, axis)
    }

    pub fn __expand_stride(
        scope: &Scope,
        this: &Self,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, N, IO>::__expand_stride(scope, this, axis)
    }

    pub fn __expand_rank(scope: &Scope, this: &Self) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, N, IO>::__expand_rank(scope, this)
    }
}

impl<E: Numeric, N: Size, IO: Clone + 'static> VirtualTensor<E, N, IO> {
    /// Create a conceptual view over this tensor, allowing for multi-dimensional indexing with custom
    /// layouts
    pub fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<Vector<E, N>, C, ReadOnly> {
        View::new::<VirtualTensor<E, N, IO>, Coords1d>(self.clone(), layout)
    }
}

#[cube]
impl<E: Numeric, N: Size, IO: Clone + 'static> VirtualTensor<E, N, IO> {
    /// Create a conceptual view over this tensor, with a simple linear layout
    pub fn as_view(&self) -> View<Vector<E, N>, usize, ReadOnly> {
        let vector_size = self.vector_size();
        View::new::<VirtualTensor<E, N, IO>, usize>(
            self.clone(),
            SimpleLayout::new(self.len() * vector_size, vector_size),
        )
    }
}

impl<E: Numeric, N: Size, IO: Clone + 'static> VirtualTensorExpand<E, N, IO> {
    /// Create a conceptual view over this tensor, allowing for multi-dimensional indexing with custom
    /// layouts
    pub fn __expand_view_method<C: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Vector<E, N>, C, ReadOnly> {
        View::__expand_new::<VirtualTensor<E, N, IO>, Coords1d>(scope, self.clone(), layout)
    }
}

impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadWrite> {
    #[doc = " Create a mutable conceptual view over this tensor, allowing for multi-dimensional indexing"]
    #[doc = " with custom layouts"]
    pub fn view_mut<C: Coordinates + 'static>(
        &self,
        layout: impl Layout<Coordinates = C, SourceCoordinates = Coords1d> + 'static,
    ) -> View<Vector<E, N>, C, ReadWrite> {
        let this: VirtualTensor<E, N, ReadWrite> = self.clone();
        View::new_mut::<VirtualTensor<E, N, ReadWrite>, Coords1d>(this, layout)
    }
    pub fn __expand_view_mut<C: Coordinates + 'static>(
        scope: &Scope,
        this: VirtualTensorExpand<E, N, ReadWrite>,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Vector<E, N>, C, ReadWrite> {
        this.__expand_view_mut_method::<C>(scope, layout)
    }
}
impl<E: Numeric, N: Size> VirtualTensorExpand<E, N, ReadWrite> {
    pub fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Vector<E, N>, C, ReadWrite> {
        View::__expand_new_mut::<VirtualTensor<E, N, ReadWrite>, Coords1d>(scope, self, layout)
    }
}

#[cube]
impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadWrite> {
    /// Create a conceptual mutable view over this tensor, with a simple linear layout
    pub fn as_view_mut(&mut self) -> View<Vector<E, N>, usize, ReadWrite> {
        let vector_size = self.vector_size();
        View::new_mut::<VirtualTensor<E, N, ReadWrite>, usize>(
            self.clone(),
            SimpleLayout::new(self.len() * vector_size, vector_size),
        )
    }
}

#[cube]
impl<E: Numeric, N: Size, IO: Clone + 'static> VirtualTensor<E, N, IO> {
    pub fn coordinate(&self, index: usize, dim: usize) -> usize {
        let num_strides = index / self.stride(dim);
        num_strides % self.shape(dim)
    }
}

impl<'a, E: Numeric, N: Size> ListMut<'a, Vector<E, N>> for VirtualTensor<E, N, ReadWrite> {
    fn __expand_write(
        scope: &Scope,
        this: &'a VirtualTensorExpand<E, N, ReadWrite>,
        index: <usize as CubeType>::ExpandType,
    ) -> &'a mut <Vector<E, N> as CubeType>::ExpandType {
        this.__expand_write_method(scope, index)
    }
}

impl<'a, E: Numeric, N: Size> ListMutExpand<'a, Vector<E, N>>
    for VirtualTensorExpand<E, N, ReadWrite>
{
    fn __expand_write_method(
        &'a self,
        scope: &Scope,
        index: <usize as CubeType>::ExpandType,
    ) -> &'a mut <Vector<E, N> as CubeType>::ExpandType {
        self.state.clone().__expand_write_method(scope, index)
    }
}

impl<'a, E: Numeric, N: Size> SliceMutOperator<'a, Vector<E, N>>
    for VirtualTensor<E, N, ReadWrite>
{
}
impl<'a, E: Numeric, N: Size> SliceMutOperatorExpand<'a, Vector<E, N>>
    for VirtualTensorExpand<E, N, ReadWrite>
{
    #[allow(unused_variables)]
    fn __expand_slice_mut_method(
        &'a mut self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &'a mut SliceExpand<Vector<E, N>, cubecl_core::prelude::ReadWrite> {
        todo!("VirtualTensor don't support slice mut yet");
    }

    #[allow(unused_variables)]
    fn __expand_to_slice_mut_method(
        &'a mut self,
        scope: &Scope,
    ) -> &'a mut SliceExpand<Vector<E, N>, cubecl_core::prelude::ReadWrite> {
        todo!("VirtualTensor don't support slice mut yet");
    }
}

impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadOnly> {
    /// Create a new [read only](ReadOnly) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E, N> + 'static>(_v: &V) -> Self {
        unexpanded!()
    }

    /// Expand function of [`Self::new`].
    pub fn __expand_new<V: VirtualTensorOperations<E, N> + 'static>(
        _scope: &Scope,
        v: V::ExpandType,
    ) -> VirtualTensorExpand<E, N, ReadOnly> {
        VirtualTensorExpand {
            state: Arc::new(v),
            _p: PhantomData,
        }
    }
}

impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadWrite> {
    /// Create a new [read write](ReadWrite) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E, N> + 'static>(_v: &mut V) -> Self {
        unexpanded!()
    }

    /// Expand function of [`Self::new`].
    pub fn __expand_new<V: VirtualTensorOperations<E, N> + 'static>(
        _scope: &Scope,
        v: V::ExpandType,
    ) -> VirtualTensorExpand<E, N, ReadWrite> {
        VirtualTensorExpand {
            state: Arc::new(v),
            _p: PhantomData,
        }
    }
}

/// Trait to be implemented by a type that can become a [virtual tensor](VirtualTensor).
///
/// The [expand trait](VirtualTensorOperationsExpand) also need to be implemented for the type's
/// expand type.
///
/// # Warning
///
/// This trait is kind of unsafe, [`VirtualTensorOperations::write`] doesn't follow the mutability
/// rules, but it won't lead to any undefined behavior.
#[cube(expand_base_traits = "VectorizedExpand")]
pub trait VirtualTensorOperations<E: Numeric, N: Size>: Vectorized {
    fn as_tensor_map(&self) -> ComptimeOption<TensorMap<E, Tiled>> {
        unexpanded!()
    }
    /// Read the tensor at the given index.
    fn read(&self, _index: usize) -> &'static Vector<E, N> {
        unexpanded!()
    }
    fn read_window(&self, _start: usize, _end: usize) -> &'static Slice<Vector<E, N>, ReadOnly> {
        unexpanded!()
    }
    /// Write the tensor at the given index.
    fn write(&self, _index: usize) -> &'static mut Vector<E, N> {
        unexpanded!()
    }
    /// Get the shape of the tensor at the given axis.
    fn shape(&self, _axis: usize) -> usize {
        unexpanded!()
    }
    /// Get the stride of the tensor at the given axis.
    fn stride(&self, _axis: usize) -> usize {
        unexpanded!()
    }
    /// Get the rank of the tensor.
    fn rank(&self) -> usize {
        unexpanded!()
    }
    fn len(&self) -> usize {
        unexpanded!()
    }
    fn buffer_len(&self) -> usize {
        unexpanded!()
    }
}

/// Making [virtual tensors](VirtualTensor) a proper [cube type](CubeType).
mod __cube_type {
    use super::*;

    impl<E: Numeric, N: Size, IO: Clone> CubeType for VirtualTensor<E, N, IO> {
        type ExpandType = VirtualTensorExpand<E, N, IO>;
    }

    impl<E: Numeric, N: Size, IO: Clone> IntoExpand for VirtualTensorExpand<E, N, IO> {
        type Expand = VirtualTensorExpand<E, N, IO>;

        fn into_expand(self, _: &Scope) -> Self::Expand {
            self
        }
    }

    impl<E: Numeric, N: Size, IO: Clone> ExpandTypeClone for VirtualTensorExpand<E, N, IO> {
        fn clone_unchecked(&self) -> Self {
            self.clone()
        }
    }

    impl<E: Numeric, N: Size, IO> IntoMut for VirtualTensorExpand<E, N, IO> {
        fn into_mut(self, _scope: &Scope) -> Self {
            self
        }
    }

    impl<E: Numeric, N: Size, IO> CubeDebug for VirtualTensorExpand<E, N, IO> {}
}

/// Enable tensors to be virtual.
mod __tensor {
    use super::*;

    impl<E: Numeric, N: Size> VirtualTensorOperations<E, N> for Tensor<Vector<E, N>> {}
    impl<E: Numeric, N: Size> VirtualTensorOperationsExpand<E, N>
        for NativeExpand<Tensor<Vector<E, N>>>
    {
        fn __expand_read_method(
            &self,
            scope: &Scope,
            index: NativeExpand<usize>,
        ) -> &'static NativeExpand<Vector<E, N>> {
            unsafe { core::mem::transmute(self.__expand_index_unchecked_method(scope, index)) }
        }
        fn __expand_read_window_method(
            &self,
            context: &Scope,
            start: NativeExpand<usize>,
            end: NativeExpand<usize>,
        ) -> &'static SliceExpand<Vector<E, N>, ReadOnly> {
            unsafe { core::mem::transmute(self.__expand_slice_method(context, start, end)) }
        }

        fn __expand_write_method(
            &self,
            scope: &Scope,
            index: NativeExpand<usize>,
        ) -> &'static mut NativeExpand<Vector<E, N>> {
            let mut this = *self;
            unsafe {
                core::mem::transmute(this.__expand_index_assign_unchecked_method(scope, index))
            }
        }

        fn __expand_shape_method(
            &self,
            scope: &Scope,
            axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            self.clone().__expand_shape_method(scope, axis)
        }

        fn __expand_stride_method(
            &self,
            scope: &Scope,
            axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            self.clone().__expand_stride_method(scope, axis)
        }

        fn __expand_rank_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.clone().__expand_rank_method(scope)
        }
        fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.clone().__expand_len_method(scope)
        }
        fn __expand_buffer_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.clone().__expand_buffer_len_method(scope)
        }

        fn __expand_as_tensor_map_method(
            &self,
            scope: &Scope,
        ) -> ComptimeOptionExpand<TensorMap<E, Tiled>> {
            ComptimeOption::__expand_new_None(scope)
        }
    }
}

/// Enable tensor maps to be virtual.
mod __tensor_map {
    use super::*;

    impl<E: Numeric, N: Size> VirtualTensorOperations<E, N> for TensorMap<E, Tiled> {}
    impl<E: Numeric, N: Size> VirtualTensorOperationsExpand<E, N>
        for NativeExpand<TensorMap<E, Tiled>>
    {
        fn __expand_read_method(
            &self,
            _scope: &Scope,
            _index: NativeExpand<usize>,
        ) -> &'static NativeExpand<Vector<E, N>> {
            todo!()
        }
        fn __expand_read_window_method(
            &self,
            _context: &Scope,
            _start: NativeExpand<usize>,
            _end: NativeExpand<usize>,
        ) -> &'static SliceExpand<Vector<E, N>, ReadOnly> {
            todo!()
        }

        fn __expand_write_method(
            &self,
            _scope: &Scope,
            _index: NativeExpand<usize>,
        ) -> &'static mut NativeExpand<Vector<E, N>> {
            todo!()
        }

        fn __expand_shape_method(
            &self,
            _scope: &Scope,
            _axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            todo!()
        }

        fn __expand_stride_method(
            &self,
            _scope: &Scope,
            _axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            todo!()
        }

        fn __expand_rank_method(&self, _scope: &Scope) -> NativeExpand<usize> {
            todo!()
        }
        fn __expand_len_method(&self, _scope: &Scope) -> NativeExpand<usize> {
            todo!()
        }
        fn __expand_buffer_len_method(&self, _scope: &Scope) -> NativeExpand<usize> {
            todo!()
        }

        fn __expand_as_tensor_map_method(
            &self,
            scope: &Scope,
        ) -> ComptimeOptionExpand<TensorMap<E, Tiled>> {
            ComptimeOption::__expand_new_Some(scope, *self)
        }
    }
}
