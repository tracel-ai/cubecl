use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand};

use super::Quantization;

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<EI: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<EO: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<EI: Numeric, EO: Numeric>: CubeType;

    /// Init the state.
    fn init_state<EI: Numeric, EO: Numeric>(
        input: &Self::Input<EI>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<EI, EO>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, coordinate: u32)
    -> Line<EI>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, coordinate: u32)
    -> Line<EI>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_window_lhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>>;

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>>;

    /// Reinterpret lhs as tensor map
    fn as_tensor_map_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI>;

    /// Reinterpret rhs as tensor map
    fn as_tensor_map_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<EI: Numeric, EO: Numeric>(
        state: &mut Self::State<EI, EO>,
        coordinate: u32,
        value: Line<EO>,
    );

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the length of the lhs tensor using the state.
    fn len_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the length of the rhs tensor using the state.
    fn len_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the length of the out tensor using the state.
    fn len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the buffer length of the lhs tensor using the state.
    fn buffer_len_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the buffer length of the rhs tensor using the state.
    fn buffer_len_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;

    /// It is the responsibility of the caller to ensure it is safe to call this function.
    /// That is, when a matmul is indeed quantized. Else, it will most likely results in
    /// out-of-bound memory access.
    fn quantization<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> Quantization<EI, EO>;
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
pub struct TensorInput<EI: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: *const GA::State<EI, EO>,
    ident: TensorInputIdent,
}

impl<EI: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperations<EI>
    for TensorInput<EI, EO, MA>
{
}

impl<EI: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperations<EO>
    for TensorOutput<EI, EO, MA>
{
}

impl<EI: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EO>
    for TensorOutputExpand<EI, EO, MA>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EO>> {
        panic!("Can't read output tensor");
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EO>>> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<EO>>,
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
    ) -> ExpandElementTyped<TensorMap<EO>> {
        unimplemented!("TensorOutputExpand can't be turned into a tensor map");
    }
}

impl<EI: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EI>
    for TensorInputExpand<EI, EO, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EI>> {
        TensorInputExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EI>>> {
        TensorInputExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<EI>>,
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
    ) -> ExpandElementTyped<TensorMap<EI>> {
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
pub struct TensorOutput<EI: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: *mut GA::State<EI, EO>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<EI: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: <GA::State<EI, EO> as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<EI: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: <GA::State<EI, EO> as CubeType>::ExpandType,
}

#[cube]
impl<EI: Numeric, EO: Numeric, MA: MatmulArgs> TensorInput<EI, EO, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(
        state: &MA::State<EI, EO>,
        #[comptime] ident: TensorInputIdent,
    ) -> TensorInput<EI, EO, MA> {
        TensorInput::<EI, EO, MA> { state, ident }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<EI>> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::read_window_lhs(&(*self.state), start, end),
                TensorInputIdent::Rhs => MA::read_window_rhs(&(*self.state), start, end),
            }
        }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<EI> {
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
    pub fn as_tensor_map(&self) -> TensorMap<EI> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::as_tensor_map_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::as_tensor_map_rhs(&(*self.state)),
            }
        }
    }
}

#[cube]
impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> TensorOutput<EI, EO, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<EI, EO>) -> TensorOutput<EI, EO, GA> {
        TensorOutput::<EI, EO, GA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<EO>) {
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
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type Input<EI: Numeric> = TensorInputs<EI>;
    type State<EI: Numeric, EO: Numeric> = (
        *const Tensor<Line<EI>>,
        *const Tensor<Line<EI>>,
        *mut Tensor<Line<EO>>,
    );

    fn init_state<EI: Numeric, EO: Numeric>(
        input: &Self::Input<EI>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<EI, EO> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI> {
        unsafe { (*state.1)[coordinate] }
    }

    fn read_window_lhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        unsafe { (*state.0).slice(start, end) }
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        unsafe { (*state.1).slice(start, end) }
    }

    fn as_tensor_map_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> TensorMap<EI> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn as_tensor_map_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> TensorMap<EI> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn shape_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out<EI: Numeric, EO: Numeric>(
        state: &mut Self::State<EI, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.1).len() }
    }

    fn len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn buffer_len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }

    fn quantization<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> Quantization<EI, EO> {
        let (lhs, rhs, out) = *state;
        unsafe {
            Quantization::<EI, EO> {
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
    type Input<EI: Numeric> = TensorMapInputs<EI>;
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type State<EI: Numeric, EO: Numeric> = (
        *const TensorMap<EI>,
        *const TensorMap<EI>,
        *mut Tensor<Line<EO>>,
    );

    fn init_state<EI: Numeric, EO: Numeric>(
        input: &Self::Input<EI>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<EI, EO> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<EI: Numeric, EO: Numeric>(
        _state: &Self::State<EI, EO>,
        _coordinate: u32,
    ) -> Line<EI> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        Line::empty(1)
    }

    fn read_rhs<EI: Numeric, EO: Numeric>(
        _state: &Self::State<EI, EO>,
        _coordinate: u32,
    ) -> Line<EI> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        Line::empty(1)
    }

    #[allow(unused)]
    fn read_window_lhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        let a = Array::new(0);
        a.to_slice()
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    #[allow(unused)]
    fn read_window_rhs<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        comptime!(unimplemented!("Can't directly read from TensorMap"));
        #[allow(unreachable_code)]
        let a = Array::new(0);
        a.to_slice()
    }

    fn as_tensor_map_lhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI> {
        unsafe { *state.0 }
    }

    fn as_tensor_map_rhs<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI> {
        unsafe { *state.1 }
    }

    fn shape_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn shape_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn shape_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { &*state.2 }.shape(dim)
    }

    fn stride_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn stride_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>, _dim: u32) -> u32 {
        // This is invalid on purpose, to allow for unified batch offset that's ignored for TMA
        // anyways. Eventually the API should change to only calculate batch offset when necessary.
        // Set to 0 to allow CUDA to remove the calculation entirely.
        0u32
    }

    fn stride_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { &*state.2 }.stride(dim)
    }

    fn write_out<EI: Numeric, EO: Numeric>(
        state: &mut Self::State<EI, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        3u32
    }

    fn rank_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        3u32
    }

    fn rank_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn len_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn buffer_len_rhs<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> u32 {
        comptime!(unimplemented!("Can't read metadata from TensorMap"));
        #[allow(unreachable_code)]
        1u32
    }

    fn buffer_len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }

    fn quantization<EI: Numeric, EO: Numeric>(
        _state: &Self::State<EI, EO>,
    ) -> Quantization<EI, EO> {
        comptime!(todo!("Quantized TMA not yet supported"));
        #[allow(unreachable_code)]
        let a = Array::new(0);
        unsafe {
            Quantization::<EI, EO> {
                lhs: a.to_slice(),
                rhs: a.to_slice(),
                out: (*_state.2).slice_mut(Self::len_out(_state), Self::buffer_len_out(_state)),
            }
        }
    }
}

mod __input {
    use super::*;

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> CubeType for TensorInput<EI, EO, GA> {
        type ExpandType = TensorInputExpand<EI, EO, GA>;
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Clone for TensorInputExpand<EI, EO, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident,
            }
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Init for TensorInputExpand<EI, EO, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug for TensorInputExpand<EI, EO, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Clone for TensorInput<EI, EO, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorInput<EI, EO, GA> {}
}

mod __output {
    use super::*;

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> CubeType for TensorOutput<EI, EO, GA> {
        type ExpandType = TensorOutputExpand<EI, EO, GA>;
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Clone for TensorOutput<EI, EO, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Clone for TensorOutputExpand<EI, EO, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Init for TensorOutputExpand<EI, EO, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug for TensorOutputExpand<EI, EO, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorOutput<EI, EO, GA> {}
}
