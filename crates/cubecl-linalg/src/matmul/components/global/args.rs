use crate::tensor::{VirtualTensorOperations, VirtualTensorOperationsExpand};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<In: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<Out: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<In: Numeric, Out: Numeric>: CubeType;

    /// Init the state.
    fn init_state<In: Numeric, Out: Numeric>(
        input: &Self::Input<In>,
        output: &mut Self::Output<Out>,
    ) -> Self::State<In, Out>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs<In: Numeric, Out: Numeric>(
        state: &Self::State<In, Out>,
        coordinate: u32,
    ) -> Line<In>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs<In: Numeric, Out: Numeric>(
        state: &Self::State<In, Out>,
        coordinate: u32,
    ) -> Line<In>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<In: Numeric, Out: Numeric>(
        state: &mut Self::State<In, Out>,
        coordinate: u32,
        value: Line<Out>,
    );

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, axis: u32) -> u32;
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
pub struct TensorInput<In: Numeric, Out: Numeric, MA: MatmulArgs> {
    state: *const MA::State<In, Out>,
    ident: TensorInputIdent,
}

impl<In: Numeric, Out: Numeric, MA: MatmulArgs> VirtualTensorOperations<In>
    for TensorInput<In, Out, MA>
{
}

impl<In: Numeric, Out: Numeric, MA: MatmulArgs> VirtualTensorOperations<Out>
    for TensorOutput<In, Out, MA>
{
}

impl<In: Numeric, Out: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<Out>
    for TensorOutputExpand<In, Out, MA>
{
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<Out>> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<Out>>,
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
}

impl<In: Numeric, Out: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<In>
    for TensorInputExpand<In, Out, MA>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<In>> {
        TensorInputExpand::__expand_read_method(self.clone(), scope, index)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<In>>,
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
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<In: Numeric, Out: Numeric, MA: MatmulArgs> {
    state: *mut MA::State<In, Out>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<In: Numeric, Out: Numeric, MA: MatmulArgs> {
    state: <MA::State<In, Out> as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<In: Numeric, Out: Numeric, MA: MatmulArgs> {
    state: <MA::State<In, Out> as CubeType>::ExpandType,
}

#[cube]
impl<In: Numeric, Out: Numeric, MA: MatmulArgs> TensorInput<In, Out, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(
        state: &MA::State<In, Out>,
        #[comptime] ident: TensorInputIdent,
    ) -> TensorInput<In, Out, MA> {
        TensorInput::<In, Out, MA> { state, ident }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<In> {
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
}

#[cube]
impl<In: Numeric, Out: Numeric, MA: MatmulArgs> TensorOutput<In, Out, MA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut MA::State<In, Out>) -> TensorOutput<In, Out, MA> {
        TensorOutput::<In, Out, MA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<Out>) {
        unsafe { MA::write_out(&mut (*self.state), coordinate, value) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { MA::shape_out(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { MA::stride_out(&(*self.state), dim) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { MA::rank_out(&(*self.state)) }
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<EG: Numeric> {
    /// The lhs tensor.
    pub lhs: Tensor<Line<EG>>,
    /// The rhs tensor.
    pub rhs: Tensor<Line<EG>>,
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<Out: Numeric> = Tensor<Line<Out>>;
    type Input<In: Numeric> = TensorInputs<In>;
    type State<In: Numeric, Out: Numeric> = (
        *const Tensor<Line<In>>,
        *const Tensor<Line<In>>,
        *mut Tensor<Line<Out>>,
    );

    fn init_state<In: Numeric, Out: Numeric>(
        input: &Self::Input<In>,
        output: &mut Self::Output<Out>,
    ) -> Self::State<In, Out> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<In: Numeric, Out: Numeric>(
        state: &Self::State<In, Out>,
        coordinate: u32,
    ) -> Line<In> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<In: Numeric, Out: Numeric>(
        state: &Self::State<In, Out>,
        coordinate: u32,
    ) -> Line<In> {
        unsafe { (*state.1)[coordinate] }
    }

    fn shape_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out<In: Numeric, Out: Numeric>(
        state: &mut Self::State<In, Out>,
        coordinate: u32,
        value: Line<Out>,
    ) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out<In: Numeric, Out: Numeric>(state: &Self::State<In, Out>) -> u32 {
        unsafe { (*state.2).rank() }
    }
}

mod __input {
    use super::*;

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> CubeType for TensorInput<In, Out, MA> {
        type ExpandType = TensorInputExpand<In, Out, MA>;
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Clone for TensorInputExpand<In, Out, MA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident,
            }
        }
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Init for TensorInputExpand<In, Out, MA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }
    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Clone for TensorInput<In, Out, MA> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Copy for TensorInput<In, Out, MA> {}

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> IntoRuntime for TensorInput<In, Out, MA> {
        fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}

mod __output {
    use super::*;

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> CubeType for TensorOutput<In, Out, MA> {
        type ExpandType = TensorOutputExpand<In, Out, MA>;
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Clone for TensorOutput<In, Out, MA> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Clone for TensorOutputExpand<In, Out, MA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Init for TensorOutputExpand<In, Out, MA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> Copy for TensorOutput<In, Out, MA> {}

    impl<In: Numeric, Out: Numeric, MA: MatmulArgs> IntoRuntime for TensorOutput<In, Out, MA> {
        fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}
