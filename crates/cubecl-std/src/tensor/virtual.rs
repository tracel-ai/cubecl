use alloc::sync::Arc;
use core::marker::PhantomData;
use cubecl::prelude::{CubeType, Scope, *};
use cubecl_core::{self as cubecl, unexpanded};
use std::ops::{Deref, DerefMut};

use crate::{
    CubeOption,
    tensor::{
        ViewExpand,
        layout::{
            Coordinates, Coords1d, Layout, VirtualLayout, VirtualLayoutExpand, simple::SimpleLayout,
        },
        view::View,
    },
};

/// Tensor representation that is decoupled from how the tensor is stored.
#[derive(Clone)]
pub struct VirtualTensor<E: Numeric, IO = ReadOnly> {
    // state: Arc<dyn VirtualTensorOperations<E>>,
    _e: PhantomData<E>,
    _p: PhantomData<IO>,
}

impl<E: Numeric, IO: Clone> Copy for VirtualTensor<E, IO> {}

/// Expand type for [VirtualTensor].
#[derive(Clone)]
pub struct VirtualTensorExpand<E: Numeric, IO> {
    state: Arc<dyn VirtualTensorOperationsExpand<E>>,
    _p: PhantomData<IO>,
}

impl<E: Numeric, IO: Clone> List<Line<E>> for VirtualTensor<E, IO> {
    fn __expand_read(
        scope: &mut Scope,
        this: VirtualTensorExpand<E, IO>,
        index: <usize as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        this.__expand_read_method(scope, index)
    }
}

impl<T: Numeric, IO: Clone> Deref for VirtualTensor<T, IO> {
    type Target = [Line<T>];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: Numeric> DerefMut for VirtualTensor<T, ReadWrite> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<E: Numeric, IO: Clone> ListExpand<Line<E>> for VirtualTensorExpand<E, IO> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: <usize as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        self.state.clone().__expand_read_method(scope, index)
    }

    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<usize>,
    ) -> <Line<E> as CubeType>::ExpandType {
        todo!("VirtualTensor don't support read unchecked yet");
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        self.state.clone().__expand_len_method(scope)
    }
}

impl<E: Numeric, IO: Clone> Lined for VirtualTensor<E, IO> {}
impl<E: Numeric, IO: Clone> LinedExpand for VirtualTensorExpand<E, IO> {
    fn line_size(&self) -> LineSize {
        self.state.clone().line_size()
    }
}

impl<E: Numeric, IO: Clone> SliceOperator<Line<E>> for VirtualTensor<E, IO> {}
impl<E: Numeric, IO: Clone> SliceOperatorExpand<Line<E>> for VirtualTensorExpand<E, IO> {
    fn __expand_slice_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        self.state
            .clone()
            .__expand_read_window_method(scope, start, end)
    }

    fn __expand_to_slice_method(&self, scope: &mut Scope) -> SliceExpand<Line<E>, ReadOnly> {
        let end = self.clone().__expand_buffer_len_method(scope);
        self.state
            .clone()
            .__expand_read_window_method(scope, 0.into(), end)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric, IO: Clone> VirtualTensor<E, IO> {
    pub fn as_tensor_map(&self) -> CubeOption<TensorMap<E, Tiled>> {
        unexpanded!()
    }
    pub fn as_slice(&self, start: usize, end: usize) -> Slice<Line<E>> {
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
        context: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <CubeOption<TensorMap<E, Tiled>> as CubeType>::ExpandType {
        this.__expand_as_tensor_map_method(context)
    }
    pub fn __expand_as_slice(
        context: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        start: <usize as CubeType>::ExpandType,
        end: <usize as CubeType>::ExpandType,
    ) -> <Slice<Line<E>> as CubeType>::ExpandType {
        this.__expand_as_slice_method(context, start, end)
    }
    pub fn __expand_shape(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_shape_method(scope, axis)
    }
    pub fn __expand_stride(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_stride_method(scope, axis)
    }
    pub fn __expand_rank(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_rank_method(scope)
    }
    pub fn __expand_buffer_len(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        this.__expand_buffer_len_method(scope)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric, IO: Clone> VirtualTensorExpand<E, IO> {
    pub fn __expand_as_tensor_map_method(
        self,
        context: &mut Scope,
    ) -> <CubeOption<TensorMap<E, Tiled>> as CubeType>::ExpandType {
        self.state.clone().__expand_as_tensor_map_method(context)
    }

    pub fn __expand_as_slice_method(
        self,
        context: &mut Scope,
        start: <usize as CubeType>::ExpandType,
        end: <usize as CubeType>::ExpandType,
    ) -> <Slice<Line<E>> as CubeType>::ExpandType {
        self.state
            .clone()
            .__expand_read_window_method(context, start, end)
    }

    pub fn __expand_shape_method(
        self,
        scope: &mut Scope,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_shape_method(scope, _arg_0.into())
    }

    pub fn __expand_stride_method(
        self,
        scope: &mut Scope,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_stride_method(scope, _arg_0.into())
    }

    pub fn __expand_rank_method(self, scope: &mut Scope) -> <usize as CubeType>::ExpandType {
        self.state.clone().__expand_rank_method(scope)
    }

    pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> <usize as CubeType>::ExpandType {
        self.state.clone().__expand_buffer_len_method(scope)
    }

    pub fn __expand_read(
        scope: &mut Scope,
        this: Self,
        index: <usize as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_read(scope, this, index)
    }

    pub fn __expand_shape(
        scope: &mut Scope,
        this: Self,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_shape(scope, this, axis)
    }

    pub fn __expand_stride(
        scope: &mut Scope,
        this: Self,
        axis: <usize as CubeType>::ExpandType,
    ) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_stride(scope, this, axis)
    }

    pub fn __expand_rank(scope: &mut Scope, this: Self) -> <usize as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_rank(scope, this)
    }
}

impl<E: Numeric, IO: Clone + 'static> VirtualTensor<E, IO> {
    /// Create a conceptual view over this tensor, allowing for multi-dimensional indexing with custom
    /// layouts
    pub fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<Line<E>, C, ReadOnly> {
        View::new::<VirtualTensor<E, IO>, Coords1d>(self, layout)
    }
}

#[cube]
impl<E: Numeric, IO: Clone + 'static> VirtualTensor<E, IO> {
    /// Create a conceptual view over this tensor, with a simple linear layout
    pub fn as_view(&self) -> View<Line<E>, usize, ReadOnly> {
        let line_size = self.line_size();
        View::new::<VirtualTensor<E, IO>, usize>(
            self,
            SimpleLayout::new(self.len() * line_size, line_size),
        )
    }
}

impl<E: Numeric, IO: Clone + 'static> VirtualTensorExpand<E, IO> {
    /// Create a conceptual view over this tensor, allowing for multi-dimensional indexing with custom
    /// layouts
    pub fn __expand_view_method<C: Coordinates + 'static>(
        &self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Line<E>, C, ReadOnly> {
        View::__expand_new::<VirtualTensor<E, IO>, Coords1d>(scope, self.clone(), layout)
    }
}

impl<E: Numeric> VirtualTensor<E, ReadWrite> {
    #[doc = " Create a mutable conceptual view over this tensor, allowing for multi-dimensional indexing"]
    #[doc = " with custom layouts"]
    pub fn view_mut<C: Coordinates + 'static>(
        &self,
        layout: impl Layout<Coordinates = C, SourceCoordinates = Coords1d> + 'static,
    ) -> View<Line<E>, C, ReadWrite> {
        let mut this: VirtualTensor<E, ReadWrite> = *self;
        View::new_mut::<VirtualTensor<E, ReadWrite>, Coords1d>(&mut this, layout)
    }
    pub fn __expand_view_mut<C: Coordinates + 'static>(
        scope: &mut Scope,
        this: VirtualTensorExpand<E, ReadWrite>,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Line<E>, C, ReadWrite> {
        this.__expand_view_mut_method::<C>(scope, layout)
    }
}
impl<E: Numeric> VirtualTensorExpand<E, ReadWrite> {
    pub fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<Line<E>, C, ReadWrite> {
        View::__expand_new_mut::<VirtualTensor<E, ReadWrite>, Coords1d>(scope, self, layout)
    }
}

#[cube]
impl<E: Numeric> VirtualTensor<E, ReadWrite> {
    /// Create a conceptual mutable view over this tensor, with a simple linear layout
    pub fn as_view_mut(&mut self) -> View<Line<E>, usize, ReadWrite> {
        let line_size = self.line_size();
        View::new_mut::<VirtualTensor<E, ReadWrite>, usize>(
            self,
            SimpleLayout::new(self.len() * line_size, line_size),
        )
    }
}

#[cube]
impl<E: Numeric, IO: Clone + 'static> VirtualTensor<E, IO> {
    pub fn coordinate(&self, index: usize, dim: usize) -> usize {
        let num_strides = index / self.stride(dim);
        num_strides % self.shape(dim)
    }
}

impl<E: Numeric> ListMut<Line<E>> for VirtualTensor<E, ReadWrite> {
    fn __expand_write(
        scope: &mut Scope,
        this: VirtualTensorExpand<E, ReadWrite>,
        index: <usize as CubeType>::ExpandType,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <() as CubeType>::ExpandType {
        this.__expand_write_method(scope, index, value)
    }
}

impl<E: Numeric> ListMutExpand<Line<E>> for VirtualTensorExpand<E, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: <usize as CubeType>::ExpandType,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <() as CubeType>::ExpandType {
        self.state
            .clone()
            .__expand_write_method(scope, index, value)
    }
}

impl<E: Numeric> SliceMutOperator<Line<E>> for VirtualTensor<E, ReadWrite> {}
impl<E: Numeric> SliceMutOperatorExpand<Line<E>> for VirtualTensorExpand<E, ReadWrite> {
    #[allow(unused_variables)]
    fn __expand_slice_mut_method(
        &self,
        scope: &mut Scope,
        start: ExpandElementTyped<usize>,
        end: ExpandElementTyped<usize>,
    ) -> SliceExpand<Line<E>, cubecl_core::prelude::ReadWrite> {
        todo!("VirtualTensor don't support slice mut yet");
    }

    #[allow(unused_variables)]
    fn __expand_to_slice_mut_method(
        &self,
        scope: &mut Scope,
    ) -> SliceExpand<Line<E>, cubecl_core::prelude::ReadWrite> {
        todo!("VirtualTensor don't support slice mut yet");
    }
}

impl<E: Numeric> VirtualTensor<E, ReadOnly> {
    /// Create a new [read only](Read) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E> + 'static>(_v: &V) -> Self {
        unexpanded!()
    }

    /// Expand function of [Self::new].
    pub fn __expand_new<V: VirtualTensorOperations<E> + 'static>(
        _scope: &mut Scope,
        v: V::ExpandType,
    ) -> VirtualTensorExpand<E, ReadOnly> {
        VirtualTensorExpand {
            state: Arc::new(v),
            _p: PhantomData,
        }
    }
}

impl<E: Numeric> VirtualTensor<E, ReadWrite> {
    /// Create a new [read write](ReadWrite) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E> + 'static>(_v: &mut V) -> Self {
        unexpanded!()
    }

    /// Expand function of [Self::new].
    pub fn __expand_new<V: VirtualTensorOperations<E> + 'static>(
        _scope: &mut Scope,
        v: V::ExpandType,
    ) -> VirtualTensorExpand<E, ReadWrite> {
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
/// This trait is kind of unsafe, [VirtualTensorOperations::write] doesn't follow the mutability
/// rules, but it won't lead to any undefined behavior.
#[cube(self_type = "ref", expand_base_traits = "LinedExpand")]
pub trait VirtualTensorOperations<E: Numeric>: Lined {
    fn as_tensor_map(&self) -> CubeOption<TensorMap<E, Tiled>> {
        unexpanded!()
    }
    /// Read the tensor at the given index.
    fn read(&self, _index: usize) -> Line<E> {
        unexpanded!()
    }
    fn read_window(&self, _start: usize, _end: usize) -> Slice<Line<E>, ReadOnly> {
        unexpanded!()
    }
    /// Write the tensor at the given index.
    fn write(&self, _index: usize, _value: Line<E>) {
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

    impl<E: Numeric, IO: Clone> CubeType for VirtualTensor<E, IO> {
        type ExpandType = VirtualTensorExpand<E, IO>;
    }

    impl<E: Numeric, IO> IntoMut for VirtualTensorExpand<E, IO> {
        fn into_mut(self, _scope: &mut Scope) -> Self {
            self
        }
    }

    impl<E: Numeric, IO> CubeDebug for VirtualTensorExpand<E, IO> {}
}

/// Enable tensors to be virtual.
mod __tensor {
    use crate::CubeOptionExpand;

    use super::*;

    impl<E: Numeric> VirtualTensorOperations<E> for Tensor<Line<E>> {}
    impl<E: Numeric> VirtualTensorOperationsExpand<E> for ExpandElementTyped<Tensor<Line<E>>> {
        fn __expand_read_method(
            &self,
            scope: &mut Scope,
            index: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<Line<E>> {
            self.clone().__expand_index_unchecked_method(scope, index)
        }
        fn __expand_read_window_method(
            &self,
            context: &mut Scope,
            start: ExpandElementTyped<usize>,
            end: ExpandElementTyped<usize>,
        ) -> SliceExpand<Line<E>, ReadOnly> {
            self.clone().__expand_slice_method(context, start, end)
        }

        fn __expand_write_method(
            &self,
            scope: &mut Scope,
            index: ExpandElementTyped<usize>,
            value: ExpandElementTyped<Line<E>>,
        ) {
            self.clone()
                .__expand_index_assign_unchecked_method(scope, index, value)
        }

        fn __expand_shape_method(
            &self,
            scope: &mut Scope,
            axis: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            self.clone().__expand_shape_method(scope, axis)
        }

        fn __expand_stride_method(
            &self,
            scope: &mut Scope,
            axis: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            self.clone().__expand_stride_method(scope, axis)
        }

        fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            self.clone().__expand_rank_method(scope)
        }
        fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            self.clone().__expand_len_method(scope)
        }
        fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
            self.clone().__expand_buffer_len_method(scope)
        }

        fn __expand_as_tensor_map_method(
            &self,
            scope: &mut Scope,
        ) -> CubeOptionExpand<TensorMap<E, Tiled>> {
            CubeOption::__expand_new_None(scope)
        }
    }
}

/// Enable tensor maps to be virtual.
mod __tensor_map {
    use crate::CubeOptionExpand;

    use super::*;

    impl<E: Numeric> VirtualTensorOperations<E> for TensorMap<E, Tiled> {}
    impl<E: Numeric> VirtualTensorOperationsExpand<E> for ExpandElementTyped<TensorMap<E, Tiled>> {
        fn __expand_read_method(
            &self,
            _scope: &mut Scope,
            _index: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<Line<E>> {
            todo!()
        }
        fn __expand_read_window_method(
            &self,
            _context: &mut Scope,
            _start: ExpandElementTyped<usize>,
            _end: ExpandElementTyped<usize>,
        ) -> SliceExpand<Line<E>, ReadOnly> {
            todo!()
        }

        fn __expand_write_method(
            &self,
            _scope: &mut Scope,
            _index: ExpandElementTyped<usize>,
            _value: ExpandElementTyped<Line<E>>,
        ) {
            todo!()
        }

        fn __expand_shape_method(
            &self,
            _scope: &mut Scope,
            _axis: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            todo!()
        }

        fn __expand_stride_method(
            &self,
            _scope: &mut Scope,
            _axis: ExpandElementTyped<usize>,
        ) -> ExpandElementTyped<usize> {
            todo!()
        }

        fn __expand_rank_method(&self, _scope: &mut Scope) -> ExpandElementTyped<usize> {
            todo!()
        }
        fn __expand_len_method(&self, _scope: &mut Scope) -> ExpandElementTyped<usize> {
            todo!()
        }
        fn __expand_buffer_len_method(&self, _scope: &mut Scope) -> ExpandElementTyped<usize> {
            todo!()
        }

        fn __expand_as_tensor_map_method(
            &self,
            scope: &mut Scope,
        ) -> CubeOptionExpand<TensorMap<E, Tiled>> {
            CubeOption::__expand_new_Some(scope, self.clone())
        }
    }
}
