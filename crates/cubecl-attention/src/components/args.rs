use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand};

use crate::components::{
    line_size::AttentionLineSizes, problem::AttentionProblem, selection::AttentionSelection,
};

/// Create the input runtime arguments for a attention kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        query: &'a TensorHandleRef<'a, R>,
        key: &'a TensorHandleRef<'a, R>,
        value: &'a TensorHandleRef<'a, R>,
        // mask: &'a TensorHandleRef<'a, R>,
        selection: &AttentionSelection,
        problem: &AttentionProblem,
        line_sizes: &AttentionLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime argument for a attention kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        selection: &AttentionSelection,
        problem: &AttentionProblem,
        line_sizes: &AttentionLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

#[cube]
/// Arguments for the attention algorithm.
pub trait AttentionArgs: Send + Sync + 'static + Clone {
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

    /// Read the line of the query tensor using the state at the given coordinate.
    fn read_query<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI>;
    /// Read the line of the key tensor using the state at the given coordinate.
    fn read_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, coordinate: u32)
    -> Line<EI>;
    /// Read the line of the value tensor using the state at the given coordinate.
    fn read_value<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI>;

    /// Read the line of the query tensor using the state at the given coordinate.
    fn read_window_query<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>>;

    /// Read the line of the key tensor using the state at the given coordinate.
    fn read_window_key<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>>;

    /// Read the line of the value tensor using the state at the given coordinate.
    fn read_window_value<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>>;

    /// Reinterpret query as tensor map
    fn as_tensor_map_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI>;

    /// Reinterpret key as tensor map
    fn as_tensor_map_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI>;

    /// Reinterpret value as tensor map
    fn as_tensor_map_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> TensorMap<EI>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<EI: Numeric, EO: Numeric>(
        state: &mut Self::State<EI, EO>,
        coordinate: u32,
        value: Line<EO>,
    );

    /// Get the rank of the query tensor using the state.
    fn rank_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the rank of the key tensor using the state.
    fn rank_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the rank of the value tensor using the state.
    fn rank_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the length of the query tensor using the state.
    fn len_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the length of the key tensor using the state.
    fn len_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the length of the value tensor using the state.
    fn len_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the length of the out tensor using the state.
    fn len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the buffer length of the query tensor using the state.
    fn buffer_len_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the buffer length of the key tensor using the state.
    fn buffer_len_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the buffer length of the value tensor using the state.
    fn buffer_len_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32;

    /// Get the shape of the query tensor using the state.
    fn shape_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the shape of the key tensor using the state.
    fn shape_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the shape of the value tensor using the state.
    fn shape_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;

    /// Get the stride of the query tensor using the state.
    fn stride_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the stride of the key tensor using the state.
    fn stride_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the stride of the value tensor using the state.
    fn stride_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, axis: u32) -> u32;
}

#[derive(Clone, Copy)]
/// Identification of the [tensor input](TensorInput).
pub enum TensorInputIdent {
    Query,
    Key,
    Value,
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorInput<EI: Numeric, EO: Numeric, GA: AttentionArgs> {
    state: *const GA::State<EI, EO>,
    ident: TensorInputIdent,
}

impl<EI: Numeric, EO: Numeric, MA: AttentionArgs> VirtualTensorOperations<EI>
    for TensorInput<EI, EO, MA>
{
}

impl<EI: Numeric, EO: Numeric, MA: AttentionArgs> VirtualTensorOperations<EO>
    for TensorOutput<EI, EO, MA>
{
}

impl<EI: Numeric, EO: Numeric, MA: AttentionArgs> VirtualTensorOperationsExpand<EO>
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
    ) -> SliceExpand<Line<EO>, ReadOnly> {
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

impl<EI: Numeric, EO: Numeric, MA: AttentionArgs> VirtualTensorOperationsExpand<EI>
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
    ) -> SliceExpand<Line<EI>, ReadOnly> {
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
pub struct TensorOutput<EI: Numeric, EO: Numeric, GA: AttentionArgs> {
    state: *mut GA::State<EI, EO>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<EI: Numeric, EO: Numeric, GA: AttentionArgs> {
    state: <GA::State<EI, EO> as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<EI: Numeric, EO: Numeric, GA: AttentionArgs> {
    state: <GA::State<EI, EO> as CubeType>::ExpandType,
}

#[cube]
impl<EI: Numeric, EO: Numeric, MA: AttentionArgs> TensorInput<EI, EO, MA> {
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
                TensorInputIdent::Query => MA::read_window_query(&(*self.state), start, end),
                TensorInputIdent::Key => MA::read_window_key(&(*self.state), start, end),
                TensorInputIdent::Value => MA::read_window_value(&(*self.state), start, end),
            }
        }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<EI> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::read_query(&(*self.state), coordinate),
                TensorInputIdent::Key => MA::read_key(&(*self.state), coordinate),
                TensorInputIdent::Value => MA::read_value(&(*self.state), coordinate),
            }
        }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::shape_query(&(*self.state), axis),
                TensorInputIdent::Key => MA::shape_key(&(*self.state), axis),
                TensorInputIdent::Value => MA::shape_value(&(*self.state), axis),
            }
        }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::stride_query(&(*self.state), axis),
                TensorInputIdent::Key => MA::stride_key(&(*self.state), axis),
                TensorInputIdent::Value => MA::stride_value(&(*self.state), axis),
            }
        }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::rank_query(&(*self.state)),
                TensorInputIdent::Key => MA::rank_key(&(*self.state)),
                TensorInputIdent::Value => MA::rank_value(&(*self.state)),
            }
        }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::len_query(&(*self.state)),
                TensorInputIdent::Key => MA::len_key(&(*self.state)),
                TensorInputIdent::Value => MA::len_value(&(*self.state)),
            }
        }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::buffer_len_query(&(*self.state)),
                TensorInputIdent::Key => MA::buffer_len_key(&(*self.state)),
                TensorInputIdent::Value => MA::buffer_len_value(&(*self.state)),
            }
        }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> TensorMap<EI> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Query => MA::as_tensor_map_query(&(*self.state)),
                TensorInputIdent::Key => MA::as_tensor_map_key(&(*self.state)),
                TensorInputIdent::Value => MA::as_tensor_map_value(&(*self.state)),
            }
        }
    }
}

#[cube]
impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> TensorOutput<EI, EO, GA> {
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
/// Type implementing [AttentionArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [AttentionArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [AttentionArgs].
pub struct TensorInputs<EG: Numeric> {
    pub query: Tensor<Line<EG>>,
    pub key: Tensor<Line<EG>>,
    pub value: Tensor<Line<EG>>,
    // pub mask: CubeOption<Tensor<Line<EM>>>,
}

impl<EG: Numeric> ConcreteInputsFactory for TensorInputs<EG> {
    fn create<'a, R: Runtime>(
        query: &'a TensorHandleRef<'a, R>,
        key: &'a TensorHandleRef<'a, R>,
        value: &'a TensorHandleRef<'a, R>,
        _selection: &AttentionSelection,
        _problem: &AttentionProblem,
        line_sizes: &AttentionLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            query.as_tensor_arg(line_sizes.query),
            key.as_tensor_arg(line_sizes.key),
            value.as_tensor_arg(line_sizes.value),
            // mask.as_tensor_arg(line_sizes.value),
        )
    }
}

impl<EG: Numeric> ConcreteOutputFactory for Tensor<Line<EG>> {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        _selection: &AttentionSelection,
        _problem: &AttentionProblem,
        line_sizes: &AttentionLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        out.as_tensor_arg(line_sizes.out)
    }
}

#[derive(CubeType)]
pub struct AttentionState<EI: Numeric, EO: Numeric> {
    pub query: *const Tensor<Line<EI>>,
    pub key: *const Tensor<Line<EI>>,
    pub value: *const Tensor<Line<EI>>,
    pub output: *mut Tensor<Line<EO>>,
}

#[cube]
impl AttentionArgs for TensorArgs {
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type Input<EI: Numeric> = TensorInputs<EI>;
    type State<EI: Numeric, EO: Numeric> = AttentionState<EI, EO>;

    fn init_state<EI: Numeric, EO: Numeric>(
        input: &Self::Input<EI>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<EI, EO> {
        AttentionState::<EI, EO> {
            query: &input.query,
            key: &input.key,
            value: &input.value,
            output,
        }
    }

    fn read_query<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI> {
        unsafe { (*state.query)[coordinate] }
    }

    fn read_key<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI> {
        unsafe { (*state.key)[coordinate] }
    }

    fn read_value<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        coordinate: u32,
    ) -> Line<EI> {
        unsafe { (*state.value)[coordinate] }
    }

    fn read_window_query<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        unsafe { (*state.query).slice(start, end) }
    }

    fn read_window_key<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        unsafe { (*state.key).slice(start, end) }
    }

    fn read_window_value<EI: Numeric, EO: Numeric>(
        state: &Self::State<EI, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EI>> {
        unsafe { (*state.value).slice(start, end) }
    }

    fn as_tensor_map_query<EI: Numeric, EO: Numeric>(
        _state: &Self::State<EI, EO>,
    ) -> TensorMap<EI> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn as_tensor_map_key<EI: Numeric, EO: Numeric>(_state: &Self::State<EI, EO>) -> TensorMap<EI> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn as_tensor_map_value<EI: Numeric, EO: Numeric>(
        _state: &Self::State<EI, EO>,
    ) -> TensorMap<EI> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn shape_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.query).shape(dim) }
    }

    fn shape_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.key).shape(dim) }
    }

    fn shape_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.value).shape(dim) }
    }

    fn shape_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.output).shape(dim) }
    }

    fn stride_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.query).stride(dim) }
    }

    fn stride_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.key).stride(dim) }
    }

    fn stride_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.value).stride(dim) }
    }

    fn stride_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>, dim: u32) -> u32 {
        unsafe { (*state.output).stride(dim) }
    }

    fn write_out<EI: Numeric, EO: Numeric>(
        state: &mut Self::State<EI, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.output)[coordinate] = value }
    }

    fn rank_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.query).rank() }
    }

    fn rank_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.key).rank() }
    }

    fn rank_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.value).rank() }
    }

    fn rank_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.output).rank() }
    }

    fn len_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.query).len() }
    }

    fn len_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.key).len() }
    }

    fn len_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.value).len() }
    }

    fn len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.output).len() }
    }

    fn buffer_len_query<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.query).buffer_len() }
    }

    fn buffer_len_key<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.key).buffer_len() }
    }

    fn buffer_len_value<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.value).buffer_len() }
    }

    fn buffer_len_out<EI: Numeric, EO: Numeric>(state: &Self::State<EI, EO>) -> u32 {
        unsafe { (*state.output).buffer_len() }
    }
}

mod __input {
    use super::*;

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> CubeType for TensorInput<EI, EO, GA> {
        type ExpandType = TensorInputExpand<EI, EO, GA>;
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Clone for TensorInputExpand<EI, EO, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident,
            }
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> IntoMut for TensorInputExpand<EI, EO, GA> {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> CubeDebug for TensorInputExpand<EI, EO, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Clone for TensorInput<EI, EO, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Copy for TensorInput<EI, EO, GA> {}
}

mod __output {
    use super::*;

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> CubeType for TensorOutput<EI, EO, GA> {
        type ExpandType = TensorOutputExpand<EI, EO, GA>;
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Clone for TensorOutput<EI, EO, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Clone for TensorOutputExpand<EI, EO, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> IntoMut for TensorOutputExpand<EI, EO, GA> {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> CubeDebug for TensorOutputExpand<EI, EO, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<EI: Numeric, EO: Numeric, GA: AttentionArgs> Copy for TensorOutput<EI, EO, GA> {}
}
