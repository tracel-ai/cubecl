use std::any::TypeId;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, server::TensorMapMeta};
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::r#virtual::{VirtualTensorOperations, VirtualTensorOperationsExpand},
};

use crate::{
    MatmulInputHandleRef,
    components::{self, MatmulLineSizes, MatmulProblem, MatmulSelection},
};

/// Create the input runtime arguments for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteInputsFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

/// Create the output runtime argument for a matmul kernel that works on concrete inputs and
/// output (not fused).
pub trait ConcreteOutputFactory: LaunchArg {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R>;
}

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<Lhs: Numeric, Rhs: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<EO: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric>: CubeType;

    /// Init the state.
    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Lhs>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Rhs>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_window_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Lhs>>;

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Rhs>>;

    /// Reinterpret lhs as tensor map
    fn as_tensor_map_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Lhs>;

    /// Reinterpret rhs as tensor map
    fn as_tensor_map_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Rhs>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
        value: Line<EO>,
    );

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;

    /// Get the length of the lhs tensor using the state.
    fn len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;
    /// Get the length of the rhs tensor using the state.
    fn len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;
    /// Get the length of the out tensor using the state.
    fn len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32;

    /// Get the buffer length of the lhs tensor using the state.
    fn buffer_len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32;
    /// Get the buffer length of the rhs tensor using the state.
    fn buffer_len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32;
    /// Get the buffer length of the out tensor using the state.
    fn buffer_len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        axis: u32,
    ) -> u32;
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
pub struct TensorLhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: *const GA::State<Lhs, Rhs, EO>,
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorRhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: *const GA::State<Lhs, Rhs, EO>,
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperations<Lhs>
    for TensorLhs<Lhs, Rhs, EO, MA>
{
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperations<Rhs>
    for TensorRhs<Lhs, Rhs, EO, MA>
{
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperations<EO>
    for TensorOutput<Lhs, Rhs, EO, MA>
{
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EO>
    for TensorOutputExpand<Lhs, Rhs, EO, MA>
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

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<Lhs>
    for TensorLhsExpand<Lhs, Rhs, EO, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<Lhs>> {
        TensorLhsExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<Lhs>, ReadOnly> {
        TensorLhsExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<Lhs>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorLhsExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorLhsExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorLhsExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorLhsExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorLhsExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ExpandElementTyped<TensorMap<Lhs>> {
        TensorLhsExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<Rhs>
    for TensorRhsExpand<Lhs, Rhs, EO, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<Rhs>> {
        TensorRhsExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<Rhs>, ReadOnly> {
        TensorRhsExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<Rhs>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorRhsExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorRhsExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorRhsExpand::__expand_rank_method(self.clone(), scope)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorRhsExpand::__expand_len_method(self.clone(), scope)
    }

    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorRhsExpand::__expand_buffer_len_method(self.clone(), scope)
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> ExpandElementTyped<TensorMap<Rhs>> {
        TensorRhsExpand::__expand_as_tensor_map_method(self.clone(), scope)
    }
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: *mut GA::State<Lhs, Rhs, EO>,
}

/// Expand type for [tensor lhs](TensorLhs).
pub struct TensorLhsExpand<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: <GA::State<Lhs, Rhs, EO> as CubeType>::ExpandType,
}

/// Expand type for [tensor rhs](TensorRhs).
pub struct TensorRhsExpand<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: <GA::State<Lhs, Rhs, EO> as CubeType>::ExpandType,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> {
    state: <GA::State<Lhs, Rhs, EO> as CubeType>::ExpandType,
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> TensorLhs<Lhs, Rhs, EO, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<Lhs, Rhs, EO>) -> TensorLhs<Lhs, Rhs, EO, MA> {
        TensorLhs::<Lhs, Rhs, EO, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<Lhs>> {
        unsafe { MA::read_window_lhs(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<Lhs> {
        unsafe { MA::read_lhs(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_lhs(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe { MA::stride_lhs(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_lhs(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { MA::len_lhs(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { MA::buffer_len_lhs(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> TensorMap<Lhs> {
        unsafe { MA::as_tensor_map_lhs(&(*self.state)) }
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, MA: MatmulArgs> TensorRhs<Lhs, Rhs, EO, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<Lhs, Rhs, EO>) -> TensorRhs<Lhs, Rhs, EO, MA> {
        TensorRhs::<Lhs, Rhs, EO, MA> { state }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<Rhs>> {
        unsafe { MA::read_window_rhs(&(*self.state), start, end) }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<Rhs> {
        unsafe { MA::read_rhs(&(*self.state), coordinate) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_rhs(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe { MA::stride_rhs(&(*self.state), axis) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_rhs(&(*self.state)) }
    }

    /// Get the length of the tensor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unsafe { MA::len_rhs(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn buffer_len(&self) -> u32 {
        unsafe { MA::buffer_len_rhs(&(*self.state)) }
    }

    /// Get the buffer length of the tensor.
    pub fn as_tensor_map(&self) -> TensorMap<Rhs> {
        unsafe { MA::as_tensor_map_rhs(&(*self.state)) }
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> TensorOutput<Lhs, Rhs, EO, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<Lhs, Rhs, EO>) -> TensorOutput<Lhs, Rhs, EO, GA> {
        TensorOutput::<Lhs, Rhs, EO, GA> { state }
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
pub struct TensorInputs<Lhs: Numeric, Rhs: Numeric> {
    /// The lhs tensor.
    pub lhs: Tensor<Line<Lhs>>,
    pub lhs_scale: CubeOption<Tensor<f32>>,
    /// The rhs tensor.
    pub rhs: Tensor<Line<Rhs>>,
    pub rhs_scale: CubeOption<Tensor<f32>>,
}

impl<Lhs: Numeric, Rhs: Numeric> ConcreteInputsFactory for TensorInputs<Lhs, Rhs> {
    fn create<'a, R: Runtime>(
        lhs: &'a MatmulInputHandleRef<'a, R>,
        rhs: &'a MatmulInputHandleRef<'a, R>,
        _selection: &MatmulSelection,
        _problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        TensorInputsLaunch::new(
            lhs.data().as_tensor_arg(line_sizes.lhs),
            lhs.scale().map(|it| it.as_tensor_arg(1)).into(),
            rhs.data().as_tensor_arg(line_sizes.rhs),
            rhs.scale().map(|it| it.as_tensor_arg(1)).into(),
        )
    }
}

impl<EG: Numeric> ConcreteOutputFactory for Tensor<Line<EG>> {
    fn create<'a, R: Runtime>(
        out: &'a TensorHandleRef<'a, R>,
        _selection: &MatmulSelection,
        _problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        out.as_tensor_arg(line_sizes.out)
    }
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type Input<Lhs: Numeric, Rhs: Numeric> = TensorInputs<Lhs, Rhs>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        *const Tensor<Line<Lhs>>,
        *const Tensor<Line<Rhs>>,
        *mut Tensor<Line<EO>>,
        CubeOption<f32>,
        CubeOption<f32>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO> {
        let lhs_scale = match &input.lhs_scale {
            CubeOption::Some(scale) => CubeOption::new_Some(scale[0]),
            CubeOption::None => CubeOption::new_None(),
        };
        let rhs_scale = match &input.rhs_scale {
            CubeOption::Some(scale) => CubeOption::new_Some(scale[0]),
            CubeOption::None => CubeOption::new_None(),
        };
        (&input.lhs, &input.rhs, output, lhs_scale, rhs_scale)
    }

    fn read_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Lhs> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
    ) -> Line<Rhs> {
        unsafe { (*state.1)[coordinate] }
    }

    fn read_window_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Lhs>> {
        unsafe { (*state.0).slice(start, end) }
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Rhs>> {
        unsafe { (*state.1).slice(start, end) }
    }

    fn as_tensor_map_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Lhs> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn as_tensor_map_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Rhs> {
        comptime!(unimplemented!("Can't use `TensorArgs` as `TensorMap`"));
        #[allow(unreachable_code)]
        TensorMap::dummy()
    }

    fn shape_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.1).len() }
    }

    fn len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn buffer_len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensor maps.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorMapArgs;

#[derive(CubeLaunch, CubeType)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorMapInputs<Lhs: Numeric, Rhs: Numeric> {
    /// The lhs tensor.
    pub lhs: TensorMap<Lhs>,
    /// The rhs tensor.
    pub rhs: TensorMap<Rhs>,
}

impl<Lhs: Numeric, Rhs: Numeric> ConcreteInputsFactory for TensorMapInputs<Lhs, Rhs> {
    fn create<'a, R: Runtime>(
        lhs_handle: &'a MatmulInputHandleRef<'a, R>,
        rhs_handle: &'a MatmulInputHandleRef<'a, R>,
        selection: &MatmulSelection,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
    ) -> Self::RuntimeArg<'a, R> {
        let lhs = lhs_handle.data();
        let rhs = rhs_handle.data();

        let tiling_scheme = selection.tiling_scheme;
        let stage_m = tiling_scheme.elements_in_stage_m();
        let stage_n = tiling_scheme.elements_in_stage_n();
        let stage_k = tiling_scheme.elements_in_stage_k();
        let stage_size_lhs = match problem.lhs_layout {
            components::MatrixLayout::RowMajor => {
                vec![1, stage_m, tiling_scheme.elements_in_tile_k()]
            }
            components::MatrixLayout::ColMajor => {
                vec![1, stage_k, tiling_scheme.elements_in_tile_m()]
            }
        };
        let stage_size_rhs = match problem.rhs_layout {
            components::MatrixLayout::RowMajor => {
                vec![1, stage_k, tiling_scheme.elements_in_tile_n()]
            }
            components::MatrixLayout::ColMajor => {
                vec![1, stage_n, tiling_scheme.elements_in_tile_k()]
            }
        };

        let lhs_elem_size = size_of::<Lhs>();
        let rhs_elem_size = size_of::<Rhs>();

        let lhs_rank = lhs.shape.len();
        let mut lhs_shape = vec![
            problem.lhs_batches[0],
            lhs.shape[lhs_rank - 2],
            lhs.shape[lhs_rank - 1],
        ];
        let mut lhs_strides = if lhs_rank > 2 {
            lhs.strides[lhs_rank - 3..].to_vec()
        } else {
            vec![1, lhs.strides[lhs_rank - 2], lhs.strides[lhs_rank - 1]]
        };

        let rhs_rank = rhs.shape.len();
        let mut rhs_shape = vec![
            problem.rhs_batches[0],
            rhs.shape[rhs_rank - 2],
            rhs.shape[rhs_rank - 1],
        ];
        let mut rhs_strides = if rhs_rank > 2 {
            rhs.strides[rhs_rank - 3..].to_vec()
        } else {
            vec![1, rhs.strides[rhs_rank - 2], rhs.strides[rhs_rank - 1]]
        };

        // TMA assumes the last stride is contiguous and won't even take it, so we need to map it
        // with transposed shape and stride. Tensor metadata still has the normal layout.
        if matches!(problem.lhs_layout, components::MatrixLayout::ColMajor) {
            lhs_shape.swap(lhs_rank - 1, lhs_rank - 2);
            lhs_strides.swap(lhs_rank - 1, lhs_rank - 2);
        }
        if matches!(problem.rhs_layout, components::MatrixLayout::ColMajor) {
            rhs_shape.swap(rhs_rank - 1, rhs_rank - 2);
            rhs_strides.swap(rhs_rank - 1, rhs_rank - 2);
        }

        fn prefetch(bytes: usize) -> TensorMapPrefetch {
            match bytes {
                ..64 => TensorMapPrefetch::None,
                64..128 => TensorMapPrefetch::B64,
                128..256 => TensorMapPrefetch::B128,
                256.. => TensorMapPrefetch::B256,
            }
        }

        let prefetch_lhs = prefetch(stage_size_lhs[2] as usize * lhs_elem_size);
        let prefetch_rhs = prefetch(stage_size_rhs[2] as usize * rhs_elem_size);

        // f32 gets remapped to tf32 for the tensor map just to ensure CUDA loads them correctly.
        // It shouldn't matter, but it's better to be safe.
        let lhs_elem = if TypeId::of::<Lhs>() == TypeId::of::<f32>() {
            tf32::as_elem_native_unchecked()
        } else {
            Lhs::as_elem_native_unchecked()
        };
        let rhs_elem = if TypeId::of::<Rhs>() == TypeId::of::<f32>() {
            tf32::as_elem_native_unchecked()
        } else {
            Rhs::as_elem_native_unchecked()
        };

        let meta_lhs = TensorMapMeta {
            format: TensorMapFormat::Tiled {
                tile_size: stage_size_lhs,
            },
            rank: 3,
            shape: lhs_shape,
            strides: lhs_strides,
            elem_stride: vec![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: prefetch_lhs,
            oob_fill: OobFill::Zero,
            elem: lhs_elem,
        };

        let meta_rhs = TensorMapMeta {
            format: TensorMapFormat::Tiled {
                tile_size: stage_size_rhs,
            },
            rank: 3,
            shape: rhs_shape,
            strides: rhs_strides,
            elem_stride: vec![1, 1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            prefetch: prefetch_rhs,
            oob_fill: OobFill::Zero,
            elem: rhs_elem,
        };

        let lhs = TensorMapArg {
            tensor: lhs.as_tensor_arg(line_sizes.lhs),
            metadata: meta_lhs,
        };
        let rhs = TensorMapArg {
            tensor: rhs.as_tensor_arg(line_sizes.rhs),
            metadata: meta_rhs,
        };

        TensorMapInputsLaunch::new(lhs, rhs)
    }
}

#[cube]
impl MatmulArgs for TensorMapArgs {
    type Input<Lhs: Numeric, Rhs: Numeric> = TensorMapInputs<Lhs, Rhs>;
    type Output<EO: Numeric> = Tensor<Line<EO>>;
    type State<Lhs: Numeric, Rhs: Numeric, EO: Numeric> = (
        *const TensorMap<Lhs>,
        *const TensorMap<Rhs>,
        *mut Tensor<Line<EO>>,
    );

    fn init_state<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        input: &Self::Input<Lhs, Rhs>,
        output: &mut Self::Output<EO>,
    ) -> Self::State<Lhs, Rhs, EO> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _coordinate: u32,
    ) -> Line<Lhs> {
        unimplemented!("Can't directly read from TensorMap")
    }

    fn read_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
        _coordinate: u32,
    ) -> Line<Rhs> {
        unimplemented!("Can't directly read from TensorMap")
    }

    #[allow(unused)]
    fn read_window_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Lhs>> {
        unimplemented!("Can't directly read from TensorMap")
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    #[allow(unused)]
    fn read_window_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        start: u32,
        end: u32,
    ) -> Slice<Line<Rhs>> {
        unimplemented!("Can't directly read from TensorMap")
    }

    fn as_tensor_map_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Lhs> {
        unsafe { *state.0 }
    }

    fn as_tensor_map_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> TensorMap<Rhs> {
        unsafe { *state.1 }
    }

    fn shape_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.2 }.shape(dim)
    }

    fn stride_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.0 }.stride(dim)
    }

    fn stride_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.1 }.stride(dim)
    }

    fn stride_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
        dim: u32,
    ) -> u32 {
        unsafe { &*state.2 }.stride(dim)
    }

    fn write_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &mut Self::State<Lhs, Rhs, EO>,
        coordinate: u32,
        value: Line<EO>,
    ) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn rank_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn rank_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(_state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(_state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(state: &Self::State<Lhs, Rhs, EO>) -> u32 {
        unsafe { (*state.2).len() }
    }

    fn buffer_len_lhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn buffer_len_rhs<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        _state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unimplemented!("Can't read metadata from TensorMap")
    }

    fn buffer_len_out<Lhs: Numeric, Rhs: Numeric, EO: Numeric>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> u32 {
        unsafe { (*state.2).buffer_len() }
    }
}

mod __lhs {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorLhs<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorLhsExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorLhsExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorLhs<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorLhs<Lhs, Rhs, EO, GA> {}
}

mod __rhs {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorRhs<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorRhsExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorRhsExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorRhs<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy for TensorRhs<Lhs, Rhs, EO, GA> {}
}

mod __output {
    use super::*;

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeType
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
        type ExpandType = TensorOutputExpand<Lhs, Rhs, EO, GA>;
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Clone
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> IntoMut
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn into_mut(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.into_mut(scope);
            self
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> CubeDebug
        for TensorOutputExpand<Lhs, Rhs, EO, GA>
    {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<Lhs: Numeric, Rhs: Numeric, EO: Numeric, GA: MatmulArgs> Copy
        for TensorOutput<Lhs, Rhs, EO, GA>
    {
    }
}
