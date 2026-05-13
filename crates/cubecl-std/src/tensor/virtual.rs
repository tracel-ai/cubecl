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

#[cube]
impl<E: Numeric, N: Size, IO: Clone> VirtualTensor<E, N, IO> {
    #[allow(unused)]
    pub fn read(&self, index: usize) -> Vector<E, N> {
        intrinsic!(|scope| { self.state.__expand_read_method(scope, index) })
    }

    #[allow(unused, clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        intrinsic!(|scope| { self.state.__expand_len_method(scope) })
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

impl<E: Numeric, N: Size, IO: Clone> Vectorized for VirtualTensor<E, N, IO> {}
impl<E: Numeric, N: Size, IO: Clone> VectorizedExpand for VirtualTensorExpand<E, N, IO> {
    fn vector_size(&self) -> VectorSize {
        self.state.clone().vector_size()
    }
}

impl<E: Numeric, N: Size, IO: Clone> SliceOperator<Vector<E, N>> for VirtualTensor<E, N, IO> {}
impl<E: Numeric, N: Size, IO: Clone> SliceOperatorExpand<Vector<E, N>>
    for VirtualTensorExpand<E, N, IO>
{
    fn __expand_slice_method(
        &self,
        scope: &Scope,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> &SliceExpand<Vector<E, N>> {
        self.state.__expand_read_window_method(scope, start, end)
    }

    fn __expand_slice_mut_method(
        &mut self,
        _: &Scope,
        _: NativeExpand<usize>,
        _: NativeExpand<usize>,
    ) -> &mut SliceExpand<Vector<E, N>> {
        todo!("VirtualTensor don't support slice mut yet");
    }
}

#[cube]
impl<E: Numeric, N: Size, IO: Clone> VirtualTensor<E, N, IO> {
    pub fn as_slice(&self) -> &[Vector<E, N>] {
        self.slice(0, self.len())
    }

    pub fn as_mut_slice(&mut self) -> &mut [Vector<E, N>] {
        self.slice_mut(0, self.len())
    }

    pub fn as_tensor_map(&self) -> ComptimeOption<TensorMap<E, Tiled>> {
        intrinsic!(|scope| self.state.__expand_as_tensor_map_method(scope))
    }
    #[allow(unused)]
    pub fn slice(&self, start: usize, end: usize) -> &[Vector<E, N>] {
        intrinsic!(|scope| self.state.__expand_read_window_method(scope, start, end))
    }
    /// Get the shape of the tensor at the given axis.
    #[allow(unused)]
    pub fn shape(&self, axis: usize) -> usize {
        intrinsic!(|scope| { self.state.__expand_shape_method(scope, axis) })
    }
    /// Get the stride of the tensor at the given axis.
    #[allow(unused)]
    pub fn stride(&self, axis: usize) -> usize {
        intrinsic!(|scope| self.state.__expand_stride_method(scope, axis))
    }
    /// Get the rank of the tensor.
    pub fn rank(&self) -> usize {
        intrinsic!(|scope| self.state.__expand_rank_method(scope))
    }

    pub fn buffer_len(&self) -> usize {
        intrinsic!(|scope| self.state.__expand_buffer_len_method(scope))
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
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Vector<E, N>, C, ReadWrite> {
        View::__expand_new_mut::<VirtualTensor<E, N, ReadWrite>, Coords1d>(
            scope,
            self.clone(),
            layout,
        )
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

#[cube]
impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadWrite> {
    #[allow(unused)]
    pub fn write(&self, index: usize, value: Vector<E, N>) {
        intrinsic!(|scope| self.state.__expand_write_method(scope, index, value))
    }
}

impl<E: Numeric, N: Size> VirtualTensor<E, N, ReadOnly> {
    /// Create a new [read only](ReadOnly) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E, N> + 'static>(_v: V) -> Self {
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
    pub fn new<V: VirtualTensorOperations<E, N> + 'static>(_v: V) -> Self {
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
    fn read(&self, _index: usize) -> Vector<E, N> {
        unexpanded!()
    }
    fn read_window(&self, _start: usize, _end: usize) -> &[Vector<E, N>] {
        unexpanded!()
    }
    /// Write the tensor at the given index.
    #[allow(unused)]
    fn write(&self, _index: usize, value: Vector<E, N>) {
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

    impl<E: Numeric, N: Size, IO: Clone> AsRefExpand for VirtualTensorExpand<E, N, IO> {
        fn __expand_ref_method(&self, _: &Scope) -> &Self {
            self
        }
    }

    impl<E: Numeric, N: Size, IO: Clone> AsMutExpand for VirtualTensorExpand<E, N, IO> {
        fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
            self
        }
    }
}

/// Enable tensors to be virtual.
mod __tensor {
    use super::*;

    impl<E: Numeric, N: Size> VirtualTensorOperations<E, N> for Tensor<Vector<E, N>> {}
    impl<E: Numeric, N: Size> VirtualTensorOperationsExpand<E, N> for TensorExpand<Vector<E, N>> {
        fn __expand_read_method(
            &self,
            scope: &Scope,
            index: NativeExpand<usize>,
        ) -> NativeExpand<Vector<E, N>> {
            unsafe {
                self.__expand_get_unchecked_method(scope, index)
                    .__expand_deref_method(scope)
            }
        }
        fn __expand_read_window_method(
            &self,
            context: &Scope,
            start: NativeExpand<usize>,
            end: NativeExpand<usize>,
        ) -> &'static SliceExpand<Vector<E, N>> {
            unsafe { core::mem::transmute(self.__expand_slice_method(context, start, end)) }
        }

        fn __expand_write_method(
            &self,
            scope: &Scope,
            index: NativeExpand<usize>,
            value: NativeExpand<Vector<E, N>>,
        ) {
            let mut this = self.clone_unchecked();
            unsafe {
                this.__expand_get_unchecked_mut_method(scope, index)
                    .__expand_assign_method(scope, value)
            };
        }

        fn __expand_shape_method(
            &self,
            scope: &Scope,
            axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            self.__expand_shape_method(scope, axis)
        }

        fn __expand_stride_method(
            &self,
            scope: &Scope,
            axis: NativeExpand<usize>,
        ) -> NativeExpand<usize> {
            self.__expand_stride_method(scope, axis)
        }

        fn __expand_rank_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.__expand_rank_method(scope)
        }
        fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.__expand_len_method(scope)
        }
        fn __expand_buffer_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
            self.__expand_buffer_len_method(scope)
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
        ) -> NativeExpand<Vector<E, N>> {
            todo!()
        }
        fn __expand_read_window_method(
            &self,
            _context: &Scope,
            _start: NativeExpand<usize>,
            _end: NativeExpand<usize>,
        ) -> &'static SliceExpand<Vector<E, N>> {
            todo!()
        }

        fn __expand_write_method(
            &self,
            _scope: &Scope,
            _index: NativeExpand<usize>,
            _value: NativeExpand<Vector<E, N>>,
        ) {
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
