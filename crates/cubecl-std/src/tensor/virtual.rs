use cubecl::prelude::{CubeType, Scope, *};
use cubecl_core::{self as cubecl, unexpanded};
use std::{marker::PhantomData, sync::Arc};

/// The read tag for [virtual tensor](VirtualTensor).
#[derive(Clone)]
pub struct Read;

/// The read write tag for [virtual tensor](VirtualTensor).
#[derive(Clone)]
pub struct ReadWrite;

/// Tensor representation that is decoupled from how the tensor is stored.
#[derive(Clone)]
pub struct VirtualTensor<E: Numeric, IO = Read> {
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

#[allow(unused, clippy::all)]
impl<E: Numeric, IO: Clone> VirtualTensor<E, IO> {
    /// Read the tensor at the given index.
    pub fn read(&self, index: u32) -> Line<E> {
        unexpanded!();
    }
    pub fn as_slice(&self, start: u32, end: u32) -> Slice<Line<E>> {
        unexpanded!();
    }
    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unexpanded!();
    }
    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unexpanded!();
    }
    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unexpanded!();
    }

    pub fn len(&self) -> u32 {
        unexpanded!();
    }

    pub fn buffer_len(&self) -> u32 {
        unexpanded!();
    }

    pub fn __expand_read(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        index: <u32 as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        this.__expand_read_method(scope, index)
    }
    pub fn __expand_as_slice(
        context: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        start: <u32 as CubeType>::ExpandType,
        end: <u32 as CubeType>::ExpandType,
    ) -> <Slice<Line<E>> as CubeType>::ExpandType {
        this.__expand_as_slice_method(context, start, end)
    }
    pub fn __expand_shape(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        this.__expand_shape_method(scope, axis)
    }
    pub fn __expand_stride(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        this.__expand_stride_method(scope, axis)
    }
    pub fn __expand_rank(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        this.__expand_rank_method(scope)
    }
    pub fn __expand_len(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        this.__expand_len_method(scope)
    }
    pub fn __expand_buffer_len(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        this.__expand_buffer_len_method(scope)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric, IO: Clone> VirtualTensorExpand<E, IO> {
    pub fn __expand_read_method(
        self,
        scope: &mut Scope,
        index: <u32 as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        let _arg_0 = index;
        self.state
            .clone()
            .__expand_read_method(scope, _arg_0.into())
    }
    pub fn __expand_as_slice_method(
        self,
        context: &mut Scope,
        start: <u32 as CubeType>::ExpandType,
        end: <u32 as CubeType>::ExpandType,
    ) -> <Slice<Line<E>> as CubeType>::ExpandType {
        self.state
            .clone()
            .__expand_read_window_method(context, start, end)
    }

    pub fn __expand_shape_method(
        self,
        scope: &mut Scope,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_shape_method(scope, _arg_0.into())
    }

    pub fn __expand_stride_method(
        self,
        scope: &mut Scope,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        let _arg_0 = axis;
        self.state
            .clone()
            .__expand_stride_method(scope, _arg_0.into())
    }

    pub fn __expand_rank_method(self, scope: &mut Scope) -> <u32 as CubeType>::ExpandType {
        self.state.clone().__expand_rank_method(scope)
    }

    pub fn __expand_len_method(self, scope: &mut Scope) -> <u32 as CubeType>::ExpandType {
        self.state.clone().__expand_len_method(scope)
    }

    pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> <u32 as CubeType>::ExpandType {
        self.state.clone().__expand_buffer_len_method(scope)
    }

    pub fn __expand_read(
        scope: &mut Scope,
        this: Self,
        index: <u32 as CubeType>::ExpandType,
    ) -> <Line<E> as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_read(scope, this, index)
    }

    pub fn __expand_shape(
        scope: &mut Scope,
        this: Self,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_shape(scope, this, axis)
    }

    pub fn __expand_stride(
        scope: &mut Scope,
        this: Self,
        axis: <u32 as CubeType>::ExpandType,
    ) -> <u32 as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_stride(scope, this, axis)
    }

    pub fn __expand_rank(scope: &mut Scope, this: Self) -> <u32 as CubeType>::ExpandType {
        VirtualTensor::<E, IO>::__expand_rank(scope, this)
    }
}

#[cube]
impl<E: Numeric, IO: Clone> VirtualTensor<E, IO> {
    pub fn coordinate(&self, index: u32, dim: u32) -> u32 {
        let num_strides = index / self.stride(dim);
        num_strides % self.shape(dim)
    }
}

#[allow(unused, clippy::all)]
impl<E: Numeric> VirtualTensor<E, ReadWrite> {
    #[doc = " Write the tensor at the given index."]
    pub fn write(&mut self, index: u32, value: Line<E>) {
        unexpanded!()
    }

    pub fn __expand_write(
        scope: &mut Scope,
        this: <Self as CubeType>::ExpandType,
        index: <u32 as CubeType>::ExpandType,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <() as CubeType>::ExpandType {
        this.__expand_write_method(scope, index, value)
    }
}
impl<E: Numeric> VirtualTensorExpand<E, ReadWrite> {
    pub fn __expand_write_method(
        self,
        scope: &mut Scope,
        index: <u32 as CubeType>::ExpandType,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <() as CubeType>::ExpandType {
        let _arg_0 = index;
        let _arg_1 = value;

        self.state
            .clone()
            .__expand_write_method(scope, _arg_0, _arg_1)
    }

    pub fn __expand_write(
        scope: &mut Scope,
        this: Self,
        index: <u32 as CubeType>::ExpandType,
        value: <Line<E> as CubeType>::ExpandType,
    ) -> <() as CubeType>::ExpandType {
        VirtualTensor::<E, ReadWrite>::__expand_write(scope, this, index, value)
    }
}
impl<E: Numeric> VirtualTensor<E, Read> {
    /// Create a new [read only](Read) [virtual tensor](VirtualTensor).
    pub fn new<V: VirtualTensorOperations<E> + 'static>(_v: &V) -> Self {
        unexpanded!()
    }

    /// Expand function of [Self::new].
    pub fn __expand_new<V>(_scope: &mut Scope, v: V::ExpandType) -> VirtualTensorExpand<E, Read>
    where
        V::ExpandType: VirtualTensorOperationsExpand<E>,
        V: VirtualTensorOperations<E> + CubeType + 'static,
    {
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
    pub fn __expand_new<V>(
        _scope: &mut Scope,
        v: V::ExpandType,
    ) -> VirtualTensorExpand<E, ReadWrite>
    where
        V::ExpandType: VirtualTensorOperationsExpand<E>,
        V: VirtualTensorOperations<E> + CubeType + 'static,
    {
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
pub trait VirtualTensorOperations<E: Numeric> {
    /// Read the tensor at the given index.
    fn read(&self, _index: u32) -> Line<E> {
        unexpanded!()
    }
    /// Write the tensor at the given index.
    fn write(&self, _index: u32, _value: Line<E>) {
        unexpanded!()
    }
    /// Get the shape of the tensor at the given axis.
    fn shape(&self, _axis: u32) -> u32 {
        unexpanded!()
    }
    /// Get the stride of the tensor at the given axis.
    fn stride(&self, _axis: u32) -> u32 {
        unexpanded!()
    }
    /// Get the rank of the tensor.
    fn rank(&self) -> u32 {
        unexpanded!()
    }
}

/// Expand trait for [VirtualTensorOperations].
///
/// For now this needs to be manually implemented for any type that needs to become a virtual
/// tensor.
pub trait VirtualTensorOperationsExpand<E: Numeric> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<E>>;
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<E>>>;
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<E>>,
    );
    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32>;
    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32>;
    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32>;
    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32>;
    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32>;
}

/// Making [virtual tensors](VirtualTensor) a proper [cube type](CubeType).
mod __cube_type {
    use super::*;

    impl<E: Numeric, IO: Clone> CubeType for VirtualTensor<E, IO> {
        type ExpandType = VirtualTensorExpand<E, IO>;
    }

    impl<E: Numeric, IO> Init for VirtualTensorExpand<E, IO> {
        fn init(self, _scope: &mut Scope) -> Self {
            self
        }
    }

    impl<E: Numeric, IO> CubeDebug for VirtualTensorExpand<E, IO> {}

    impl<E: Numeric, IO: Clone> IntoRuntime for VirtualTensor<E, IO> {
        fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
            panic!("Virtual tensors don't exist at comptime.")
        }
    }
}

/// Enable tensors to be virtual.
mod __tensor {
    use super::*;

    impl<E: Numeric> VirtualTensorOperations<E> for Tensor<Line<E>> {}
    impl<E: Numeric> VirtualTensorOperationsExpand<E> for ExpandElementTyped<Tensor<Line<E>>> {
        fn __expand_read_method(
            &self,
            scope: &mut Scope,
            index: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<Line<E>> {
            self.clone().__expand_index_unchecked_method(scope, index)
        }
        fn __expand_read_window_method(
            &self,
            context: &mut Scope,
            start: ExpandElementTyped<u32>,
            end: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<Slice<Line<E>>> {
            self.clone().__expand_slice_method(context, start, end)
        }

        fn __expand_write_method(
            &self,
            scope: &mut Scope,
            index: ExpandElementTyped<u32>,
            value: ExpandElementTyped<Line<E>>,
        ) {
            self.clone()
                .__expand_index_assign_unchecked_method(scope, index, value)
        }

        fn __expand_shape_method(
            &self,
            scope: &mut Scope,
            axis: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            self.clone().__expand_shape_method(scope, axis)
        }

        fn __expand_stride_method(
            &self,
            scope: &mut Scope,
            axis: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<u32> {
            self.clone().__expand_stride_method(scope, axis)
        }

        fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            self.clone().__expand_rank_method(scope)
        }
        fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            self.clone().__expand_len_method(scope)
        }
        fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
            self.clone().__expand_buffer_len_method(scope)
        }
    }
}
