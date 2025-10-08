use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand},
};

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
        mask: &'a TensorHandleRef<'a, R>,
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
    type Input<Q: Float, K: Float, V: Float>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<O: Float>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<Q: Float, K: Float, V: Float, O: Float>: CubeType;

    /// Init the state.
    fn init_state<Q: Float, K: Float, V: Float, O: Float>(
        input: &Self::Input<Q, K, V>,
        output: &mut Self::Output<O>,
    ) -> Self::State<Q, K, V, O>;

    /// Read the line of the query tensor using the state at the given coordinate.
    fn read_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<Q>;
    /// Read the line of the key tensor using the state at the given coordinate.
    fn read_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<K>;
    /// Read the line of the value tensor using the state at the given coordinate.
    fn read_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<V>;

    /// Read the line of the query tensor using the state at the given coordinate.
    fn read_window_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Q>>;

    /// Read the line of the key tensor using the state at the given coordinate.
    fn read_window_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<K>>;

    /// Read the line of the value tensor using the state at the given coordinate.
    fn read_window_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<V>>;

    /// Reinterpret query as tensor map
    fn as_tensor_map_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<Q>>;

    /// Reinterpret key as tensor map
    fn as_tensor_map_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<K>>;

    /// Reinterpret value as tensor map
    fn as_tensor_map_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<V>>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &mut Self::State<Q, K, V, O>,
        coordinate: u32,
        value: Line<O>,
    );

    /// Get the rank of the query tensor using the state.
    fn rank_query<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the rank of the key tensor using the state.
    fn rank_key<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the rank of the value tensor using the state.
    fn rank_value<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;

    /// Get the length of the query tensor using the state.
    fn len_query<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the length of the key tensor using the state.
    fn len_key<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the length of the value tensor using the state.
    fn len_value<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;
    /// Get the length of the out tensor using the state.
    fn len_out<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32;

    /// Get the buffer length of the query tensor using the state.
    fn buffer_len_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32;
    /// Get the buffer length of the key tensor using the state.
    fn buffer_len_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32;
    /// Get the buffer length of the value tensor using the state.
    fn buffer_len_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32;

    /// Get the shape of the query tensor using the state.
    fn shape_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the shape of the key tensor using the state.
    fn shape_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the shape of the value tensor using the state.
    fn shape_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;

    /// Get the stride of the query tensor using the state.
    fn stride_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the stride of the key tensor using the state.
    fn stride_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the stride of the value tensor using the state.
    fn stride_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        axis: u32,
    ) -> u32;

    fn line_size_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32);
    fn line_size_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32);
    fn line_size_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32);
    fn line_size_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32);
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorQuery<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: *const GA::State<Q, K, V, O>,
}

pub struct TensorKey<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: *const GA::State<Q, K, V, O>,
}

pub struct TensorValue<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: *const GA::State<Q, K, V, O>,
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperations<Q>
    for TensorQuery<Q, K, V, O, MA>
{
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperations<K>
    for TensorKey<Q, K, V, O, MA>
{
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperations<V>
    for TensorValue<Q, K, V, O, MA>
{
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperations<O>
    for TensorOutput<Q, K, V, O, MA>
{
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperationsExpand<O>
    for TensorOutputExpand<Q, K, V, O, MA>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<O>> {
        panic!("Can't read output tensor");
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<O>, ReadOnly> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<O>>,
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

    fn __expand_as_tensor_map_method(&self, scope: &mut Scope) -> CubeOptionExpand<TensorMap<O>> {
        CubeOption::__expand_new_None(scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> Lined
    for TensorOutput<Q, K, V, O, MA>
{
}
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> LinedExpand
    for TensorOutputExpand<Q, K, V, O, MA>
{
    fn line_size(&self) -> u32 {
        let mut scope = Scope::root(false);
        TensorOutputExpand::__expand_line_size_method(self.clone(), &mut scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperationsExpand<Q>
    for TensorQueryExpand<Q, K, V, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<Q>> {
        TensorQueryExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<Q>, ReadOnly> {
        TensorQueryExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<Q>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorQueryExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorQueryExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorQueryExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorQueryExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorQueryExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(&self, scope: &mut Scope) -> CubeOptionExpand<TensorMap<Q>> {
        TensorQueryExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> Lined
    for TensorQuery<Q, K, V, O, MA>
{
}
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> LinedExpand
    for TensorQueryExpand<Q, K, V, O, MA>
{
    fn line_size(&self) -> u32 {
        let mut scope = Scope::root(false);
        TensorQueryExpand::__expand_line_size_method(self.clone(), &mut scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperationsExpand<K>
    for TensorKeyExpand<Q, K, V, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<K>> {
        TensorKeyExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<K>, ReadOnly> {
        TensorKeyExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<K>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorKeyExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorKeyExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorKeyExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorKeyExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorKeyExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(&self, scope: &mut Scope) -> CubeOptionExpand<TensorMap<K>> {
        TensorKeyExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> Lined
    for TensorKey<Q, K, V, O, MA>
{
}
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> LinedExpand
    for TensorKeyExpand<Q, K, V, O, MA>
{
    fn line_size(&self) -> u32 {
        let mut scope = Scope::root(false);
        TensorKeyExpand::__expand_line_size_method(self.clone(), &mut scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> VirtualTensorOperationsExpand<V>
    for TensorValueExpand<Q, K, V, O, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<V>> {
        TensorValueExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<V>, ReadOnly> {
        TensorValueExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<V>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorValueExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorValueExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorValueExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorValueExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorValueExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(&self, scope: &mut Scope) -> CubeOptionExpand<TensorMap<V>> {
        TensorValueExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> Lined
    for TensorValue<Q, K, V, O, MA>
{
}
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> LinedExpand
    for TensorValueExpand<Q, K, V, O, MA>
{
    fn line_size(&self) -> u32 {
        let mut scope = Scope::root(false);
        TensorValueExpand::__expand_line_size_method(self.clone(), &mut scope)
    }
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: *mut GA::State<Q, K, V, O>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorQueryExpand<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: <GA::State<Q, K, V, O> as CubeType>::ExpandType,
}

pub struct TensorKeyExpand<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: <GA::State<Q, K, V, O> as CubeType>::ExpandType,
}

pub struct TensorValueExpand<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: <GA::State<Q, K, V, O> as CubeType>::ExpandType,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> {
    state: <GA::State<Q, K, V, O> as CubeType>::ExpandType,
}

#[cube]
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> TensorQuery<Q, K, V, O, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<Q, K, V, O>) -> TensorQuery<Q, K, V, O, MA> {
        TensorQuery::<Q, K, V, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<Q>> {
        unsafe { MA::read_window_query(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<Q> {
        unsafe { MA::read_query(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_query(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe { MA::stride_query(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_query(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { MA::len_query(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { MA::buffer_len_query(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> CubeOption<TensorMap<Q>> {
        unsafe { MA::as_tensor_map_query(&(*self.state)) }
    }

    /// Get the line size of the tensor.
    pub fn line_size(&self) -> comptime_type!(u32) {
        unsafe { MA::line_size_query(&(*self.state)) }
    }
}

#[cube]
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> TensorKey<Q, K, V, O, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<Q, K, V, O>) -> TensorKey<Q, K, V, O, MA> {
        TensorKey::<Q, K, V, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<K>> {
        unsafe { MA::read_window_key(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<K> {
        unsafe { MA::read_key(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_key(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe { MA::stride_key(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_key(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { MA::len_key(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { MA::buffer_len_key(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> CubeOption<TensorMap<K>> {
        unsafe { MA::as_tensor_map_key(&(*self.state)) }
    }

    /// Get the line size of the tensor.
    pub fn line_size(&self) -> comptime_type!(u32) {
        unsafe { MA::line_size_key(&(*self.state)) }
    }
}

#[cube]
impl<Q: Float, K: Float, V: Float, O: Float, MA: AttentionArgs> TensorValue<Q, K, V, O, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<Q, K, V, O>) -> TensorValue<Q, K, V, O, MA> {
        TensorValue::<Q, K, V, O, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<V>> {
        unsafe { MA::read_window_value(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<V> {
        unsafe { MA::read_value(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_value(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe { MA::stride_value(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_value(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { MA::len_value(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { MA::buffer_len_value(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> CubeOption<TensorMap<V>> {
        unsafe { MA::as_tensor_map_value(&(*self.state)) }
    }

    /// Get the line size of the tensor.
    pub fn line_size(&self) -> comptime_type!(u32) {
        unsafe { MA::line_size_value(&(*self.state)) }
    }
}

#[cube]
impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> TensorOutput<Q, K, V, O, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<Q, K, V, O>) -> TensorOutput<Q, K, V, O, GA> {
        TensorOutput::<Q, K, V, O, GA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<O>) {
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
        unsafe { GA::buffer_len_out(&(*self.state)) }
    }

    /// Get the line size of the tensor.
    pub fn line_size(&self) -> comptime_type!(u32) {
        unsafe { GA::line_size_out(&(*self.state)) }
    }
}

#[derive(Clone)]
/// Type implementing [AttentionArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [AttentionArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [AttentionArgs].
pub struct TensorInputs<Q: Float, K: Float, V: Float, M: Numeric> {
    pub query: Tensor<Line<Q>>,
    pub key: Tensor<Line<K>>,
    pub value: Tensor<Line<V>>,
    pub mask: CubeOption<Tensor<Line<M>>>,
}

impl<Q: Float, K: Float, V: Float, M: Numeric> ConcreteInputsFactory for TensorInputs<Q, K, V, M> {
    fn create<'a, R: Runtime>(
        query: &'a TensorHandleRef<'a, R>,
        key: &'a TensorHandleRef<'a, R>,
        value: &'a TensorHandleRef<'a, R>,
        mask: &'a TensorHandleRef<'a, R>,
        _selection: &AttentionSelection,
        _problem: &AttentionProblem,
        line_sizes: &AttentionLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            query.as_tensor_arg(line_sizes.query),
            key.as_tensor_arg(line_sizes.key),
            value.as_tensor_arg(line_sizes.value),
            // TODO CubeOptionArgs
            mask.as_tensor_arg(line_sizes.mask),
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
pub struct AttentionState<Q: Float, K: Float, V: Float, O: Float> {
    pub query: *const Tensor<Line<Q>>,
    pub key: *const Tensor<Line<K>>,
    pub value: *const Tensor<Line<V>>,
    pub output: *mut Tensor<Line<O>>,
}

#[cube]
impl AttentionArgs for TensorArgs {
    type Input<Q: Float, K: Float, V: Float> = TensorInputs<Q, K, V>;
    type Output<O: Float> = Tensor<Line<O>>;
    type State<Q: Float, K: Float, V: Float, O: Float> = AttentionState<Q, K, V, O>;

    fn init_state<Q: Float, K: Float, V: Float, O: Float>(
        input: &Self::Input<Q, K, V>,
        output: &mut Self::Output<O>,
    ) -> Self::State<Q, K, V, O> {
        AttentionState::<Q, K, V, O> {
            query: &input.query,
            key: &input.key,
            value: &input.value,
            output,
        }
    }

    fn read_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<Q> {
        unsafe { (*state.query)[coordinate] }
    }

    fn read_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<K> {
        unsafe { (*state.key)[coordinate] }
    }

    fn read_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        coordinate: u32,
    ) -> Line<V> {
        unsafe { (*state.value)[coordinate] }
    }

    fn read_window_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Q>> {
        unsafe { (*state.query).slice(start, end) }
    }

    fn read_window_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<K>> {
        unsafe { (*state.key).slice(start, end) }
    }

    fn read_window_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        start: u32,
        end: u32,
    ) -> Slice<Line<V>> {
        unsafe { (*state.value).slice(start, end) }
    }

    fn as_tensor_map_query<Q: Float, K: Float, V: Float, O: Float>(
        _state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<Q>> {
        CubeOption::new_None()
    }

    fn as_tensor_map_key<Q: Float, K: Float, V: Float, O: Float>(
        _state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<K>> {
        CubeOption::new_None()
    }

    fn as_tensor_map_value<Q: Float, K: Float, V: Float, O: Float>(
        _state: &Self::State<Q, K, V, O>,
    ) -> CubeOption<TensorMap<V>> {
        CubeOption::new_None()
    }

    fn shape_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.query).shape(dim) }
    }

    fn shape_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.key).shape(dim) }
    }

    fn shape_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.value).shape(dim) }
    }

    fn shape_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.output).shape(dim) }
    }

    fn stride_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.query).stride(dim) }
    }

    fn stride_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.key).stride(dim) }
    }

    fn stride_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.value).stride(dim) }
    }

    fn stride_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.output).stride(dim) }
    }

    fn write_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &mut Self::State<Q, K, V, O>,
        coordinate: u32,
        value: Line<O>,
    ) {
        unsafe { (*state.output)[coordinate] = value }
    }

    fn rank_query<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.query).rank() }
    }

    fn rank_key<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.key).rank() }
    }

    fn rank_value<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.value).rank() }
    }

    fn rank_out<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.output).rank() }
    }

    fn len_query<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.query).len() }
    }

    fn len_key<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.key).len() }
    }

    fn len_value<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.value).len() }
    }

    fn len_out<Q: Float, K: Float, V: Float, O: Float>(state: &Self::State<Q, K, V, O>) -> u32 {
        unsafe { (*state.output).len() }
    }

    fn buffer_len_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32 {
        unsafe { (*state.query).buffer_len() }
    }

    fn buffer_len_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32 {
        unsafe { (*state.key).buffer_len() }
    }

    fn buffer_len_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32 {
        unsafe { (*state.value).buffer_len() }
    }

    fn buffer_len_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> u32 {
        unsafe { (*state.output).buffer_len() }
    }

    fn line_size_query<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.query).line_size() }
    }

    fn line_size_key<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.key).line_size() }
    }

    fn line_size_value<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.value).line_size() }
    }

    fn line_size_out<Q: Float, K: Float, V: Float, O: Float>(
        state: &Self::State<Q, K, V, O>,
    ) -> comptime_type!(u32) {
        unsafe { (*state.output).line_size() }
    }
}

mod __query {
    use super::*;

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeType
        for TensorQuery<Q, K, V, O, GA>
    {
        type ExpandType = TensorQueryExpand<Q, K, V, O, GA>;
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorQueryExpand<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> IntoMut
        for TensorQueryExpand<Q, K, V, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeDebug
        for TensorQueryExpand<Q, K, V, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorQuery<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Copy
        for TensorQuery<Q, K, V, O, GA>
    {
    }
}

mod __key {
    use super::*;

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeType
        for TensorKey<Q, K, V, O, GA>
    {
        type ExpandType = TensorKeyExpand<Q, K, V, O, GA>;
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorKeyExpand<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> IntoMut
        for TensorKeyExpand<Q, K, V, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeDebug
        for TensorKeyExpand<Q, K, V, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorKey<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Copy for TensorKey<Q, K, V, O, GA> {}
}

mod __value {
    use super::*;

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeType
        for TensorValue<Q, K, V, O, GA>
    {
        type ExpandType = TensorValueExpand<Q, K, V, O, GA>;
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorValueExpand<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> IntoMut
        for TensorValueExpand<Q, K, V, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeDebug
        for TensorValueExpand<Q, K, V, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorValue<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Copy
        for TensorValue<Q, K, V, O, GA>
    {
    }
}

mod __output {
    use super::*;

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeType
        for TensorOutput<Q, K, V, O, GA>
    {
        type ExpandType = TensorOutputExpand<Q, K, V, O, GA>;
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorOutput<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Clone
        for TensorOutputExpand<Q, K, V, O, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> IntoMut
        for TensorOutputExpand<Q, K, V, O, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> CubeDebug
        for TensorOutputExpand<Q, K, V, O, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<Q: Float, K: Float, V: Float, O: Float, GA: AttentionArgs> Copy
        for TensorOutput<Q, K, V, O, GA>
    {
    }
}
