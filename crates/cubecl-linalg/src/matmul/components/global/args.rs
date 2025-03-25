use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand};

use super::Quantization;

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<EG: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<EG: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<EG: Numeric>: CubeType;

    /// Init the state.
    fn init_state<EG: Numeric>(
        input: &Self::Input<EG>,
        output: &mut Self::Output<EG>,
    ) -> Self::State<EG>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_window_lhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>>;

    /// Reinterpret lhs as tensor map
    fn as_tensor_map_lhs<EG: Numeric>(state: &Self::State<EG>) -> TensorMap<EG>;
    /// Reinterpret rhs as tensor map
    fn as_tensor_map_rhs<EG: Numeric>(state: &Self::State<EG>) -> TensorMap<EG>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>);

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32;

    /// Get the length of the lhs tensor using the state.
    fn len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the length of the rhs tensor using the state.
    fn len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the length of the out tensor using the state.
    fn len_out<EG: Numeric>(state: &Self::State<EG>) -> u32;

    /// Get the buffer length of the lhs tensor using the state.
    fn buffer_len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the buffer length of the rhs tensor using the state.
    fn buffer_len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<EG: Numeric>(state: &Self::State<EG>) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;

    /// It is the responsibility of the caller to ensure it is safe to call this function.
    /// That is, when a matmul is indeed quantized. Else, it will most likely results in
    /// out-of-bound memory access.
    fn quantization<EG: Numeric>(state: &Self::State<EG>) -> Quantization<EG>;
}

#[derive(Clone, Copy)]
/// Identification of the [tensor input](TensorInput).
pub enum TensorInputIdent {
    Lhs,
    Rhs,
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorInput<EG: Numeric, GA: MatmulArgs> {
    state: *const GA::State<EG>,
    ident: TensorInputIdent,
}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperations<EG> for TensorInput<EG, MA> {}
impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperations<EG> for TensorOutput<EG, MA> {}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EG> for TensorOutputExpand<EG, MA> {
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EG>> {
        panic!("Can't read output tensor");
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EG>>> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<EG>>,
    ) {
        TensorOutputExpand::__expand_write_method(self.clone(), scope, index, value)
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        _scope: &mut Scope,
    ) -> ExpandElementTyped<TensorMap<EG>> {
        unimplemented!("TensorOutputExpand can't be turned into a tensor map");
    }
}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EG> for TensorInputExpand<EG, MA> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EG>> {
        TensorInputExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EG>>> {
        TensorInputExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<EG>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ExpandElementTyped<TensorMap<EG>> {
        TensorInputExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<EG: Numeric, GA: MatmulArgs> {
    state: *mut GA::State<EG>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<EG: Numeric, GA: MatmulArgs> {
    state: <GA::State<EG> as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<EG: Numeric, GA: MatmulArgs> {
    state: <GA::State<EG> as CubeType>::ExpandType,
}

#[cube]
impl<EG: Numeric, MA: MatmulArgs> TensorInput<EG, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<EG>, #[comptime] ident: TensorInputIdent) -> TensorInput<EG, MA> {
        TensorInput::<EG, MA> { state, ident }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<EG>> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::read_window_lhs(&(*self.state), start, end),
                TensorInputIdent::Rhs => MA::read_window_rhs(&(*self.state), start, end),
            }
        }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<EG> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::read_lhs(&(*self.state), coordinate),
                TensorInputIdent::Rhs => MA::read_rhs(&(*self.state), coordinate),
            }
        }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::shape_lhs(&(*self.state), axis),
                TensorInputIdent::Rhs => MA::shape_rhs(&(*self.state), axis),
            }
        }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::stride_lhs(&(*self.state), axis),
                TensorInputIdent::Rhs => MA::stride_rhs(&(*self.state), axis),
            }
        }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::rank_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::rank_rhs(&(*self.state)),
            }
        }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::len_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::len_rhs(&(*self.state)),
            }
        }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::buffer_len_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::buffer_len_rhs(&(*self.state)),
            }
        }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> TensorMap<EG> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::as_tensor_map_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::as_tensor_map_rhs(&(*self.state)),
            }
        }
    }
}

#[cube]
impl<EG: Numeric, GA: MatmulArgs> TensorOutput<EG, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<EG>) -> TensorOutput<EG, GA> {
        TensorOutput::<EG, GA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<EG>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, value) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { GA::shape_out(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { GA::stride_out(&(*self.state), dim) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { GA::rank_out(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { GA::len_out(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { GA::len_out(&(*self.state)) }
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<EG: Numeric> {
    /// The lhs tensor.
    pub lhs: Tensor<Line<EG>>,
    /// The rhs tensor.
    pub rhs: Tensor<Line<EG>>,
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EG: Numeric> = Tensor<Line<EG>>;
    type Input<EG: Numeric> = TensorInputs<EG>;
    type State<EG: Numeric> = (
        *const Tensor<Line<EG>>,
        *const Tensor<Line<EG>>,
        *mut Tensor<Line<EG>>,
    );

    fn init_state<EG: Numeric>(
        input: &Self::Input<EG>,
        output: &mut Self::Output<EG>,
    ) -> Self::State<EG> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        unsafe { (*state.1)[coordinate] }
    }

    fn read_window_lhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        unsafe { (*state.0).slice(start, end) }
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        unsafe { (*state.1).slice(start, end) }
    }

    fn as_tensor_map_lhs<EG: Numeric>(_state: &Self::State<EG>) -> TensorMap<EG> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn as_tensor_map_rhs<EG: Numeric>(_state: &Self::State<EG>) -> TensorMap<EG> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn shape_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.1).len() }
    }

    fn len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn buffer_len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }

    fn quantization<EG: Numeric>(state: &Self::State<EG>) -> Quantization<EG> {
        let (lhs, rhs, out) = *state;
        unsafe {
            Quantization::<EG> {
                lhs: (*lhs).slice(Self::len_lhs(state), Self::buffer_len_lhs(state)),
                rhs: (*rhs).slice(Self::len_rhs(state), Self::buffer_len_rhs(state)),
                out: (*out).slice_mut(Self::len_out(state), Self::buffer_len_out(state)),
            }
            // TODO Currently I assume that buffer_len = metadata_len + len.
            //      That is, all the data within the tensors are contiguous and there are no hole
            //      in the stride pattern.
        }
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<EG: Numeric> {
    /// The lhs tensor.
    pub lhs: TensorMap<EG>,
    /// The rhs tensor.
    pub rhs: TensorMap<EG>,
}

#[cube]
impl MatmulArgs for TensorMapArgs {
    type Output<EG: Numeric> = Tensor<Line<EG>>;
    type Input<EG: Numeric> = TensorMapInputs<EG>;
    type State<EG: Numeric> = (
        *const TensorMap<EG>,
        *const TensorMap<EG>,
        *mut Tensor<Line<EG>>,
    );

    fn init_state<EG: Numeric>(
        input: &Self::Input<EG>,
        output: &mut Self::Output<EG>,
    ) -> Self::State<EG> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<EG: Numeric>(_state: &Self::State<EG>, _coordinate: u32) -> Line<EG> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        Line::empty(1)
    }

    fn read_rhs<EG: Numeric>(_state: &Self::State<EG>, _coordinate: u32) -> Line<EG> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        Line::empty(1)
    }

    #[allow(unused)]
    fn read_window_lhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        unsafe { &*state.2 }.slice(start, end)
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    #[allow(unused)]
    fn read_window_rhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        unsafe { &*state.2 }.slice(start, end)
    }

    fn as_tensor_map_lhs<EG: Numeric>(state: &Self::State<EG>) -> TensorMap<EG> {
        unsafe { *state.0 }
    }

    fn as_tensor_map_rhs<EG: Numeric>(state: &Self::State<EG>) -> TensorMap<EG> {
        unsafe { *state.1 }
    }

    fn shape_lhs<EG: Numeric>(_state: &Self::State<EG>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn shape_rhs<EG: Numeric>(_state: &Self::State<EG>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn shape_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { &*state.2 }.shape(dim)
    }

    fn stride_lhs<EG: Numeric>(_state: &Self::State<EG>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn stride_rhs<EG: Numeric>(_state: &Self::State<EG>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn stride_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { &*state.2 }.stride(dim)
    }

    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        3u32
    }

    fn rank_rhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        3u32
    }

    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn len_rhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn buffer_len_rhs<EG: Numeric>(_state: &Self::State<EG>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn buffer_len_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }

    fn quantization<EG: Numeric>(_state: &Self::State<EG>) -> Quantization<EG> {
        comptime!(todo!("Quantized TMA not yet supported"));
        #[allow(unreachable_code)]
        unsafe {
            Quantization::<EG> {
                lhs: (*_state.2).slice(Self::len_out(_state), Self::buffer_len_out(_state)),
                rhs: (*_state.2).slice(Self::len_out(_state), Self::buffer_len_out(_state)),
                out: (*_state.2).slice_mut(Self::len_out(_state), Self::buffer_len_out(_state)),
            }
        }
    }
}

mod __input {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs> CubeType for TensorInput<EG, GA> {
        type ExpandType = TensorInputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorInputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident,
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Init for TensorInputExpand<EG, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> CubeDebug for TensorInputExpand<EG, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorInput<EG, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> Copy for TensorInput<EG, GA> {}
}

mod __output {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs> CubeType for TensorOutput<EG, GA> {
        type ExpandType = TensorOutputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorOutput<EG, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorOutputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Init for TensorOutputExpand<EG, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> CubeDebug for TensorOutputExpand<EG, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Copy for TensorOutput<EG, GA> {}
}
