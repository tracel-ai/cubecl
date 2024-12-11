use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs<EG: Numeric>: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input: LaunchArg + CubeType;
    /// Type used for the output.
    type Output: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State: CubeType;

    /// Init the state.
    fn init_state(input: &Self::Input, output: &mut Self::Output) -> Self::State;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>);

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs(state: &Self::State) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs(state: &Self::State) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out(state: &Self::State) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs(state: &Self::State, axis: u32) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs(state: &Self::State, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out(state: &Self::State, axis: u32) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs(state: &Self::State, axis: u32) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs(state: &Self::State, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out(state: &Self::State, axis: u32) -> u32;
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
pub struct TensorInput<EG: Numeric, GA: MatmulArgs<EG>> {
    state: *const GA::State,
    ident: TensorInputIdent,
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<EG: Numeric, GA: MatmulArgs<EG>> {
    state: *mut GA::State,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<EG: Numeric, GA: MatmulArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<EG: Numeric, GA: MatmulArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
}

#[cube]
impl<EG: Numeric, MA: MatmulArgs<EG>> TensorInput<EG, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State, #[comptime] ident: TensorInputIdent) -> TensorInput<EG, MA> {
        TensorInput::<EG, MA> { state, ident }
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
}

#[cube]
impl<EG: Numeric, GA: MatmulArgs<EG>> TensorOutput<EG, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State) -> TensorOutput<EG, GA> {
        TensorOutput::<EG, GA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<EG>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, value) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { GA::shape_out(&mut (*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { GA::stride_out(&mut (*self.state), dim) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { GA::rank_out(&mut (*self.state)) }
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
impl<EG: Numeric> MatmulArgs<EG> for TensorArgs {
    type Output = Tensor<Line<EG>>;
    type Input = TensorInputs<EG>;
    type State = (
        *const Tensor<Line<EG>>,
        *const Tensor<Line<EG>>,
        *mut Tensor<Line<EG>>,
    );

    fn init_state(input: &Self::Input, output: &mut Self::Output) -> Self::State {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG> {
        unsafe { (*state.1)[coordinate] }
    }

    fn shape_lhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out(state: &Self::State, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs(state: &Self::State) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs(state: &Self::State) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out(state: &Self::State) -> u32 {
        unsafe { (*state.2).rank() }
    }
}

mod __input {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs<EG>> CubeType for TensorInput<EG, GA> {
        type ExpandType = TensorInputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Clone for TensorInputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident.clone(),
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Init for TensorInputExpand<EG, GA> {
        fn init(mut self, context: &mut CubeContext) -> Self {
            self.state = self.state.init(context);
            self
        }
    }
    impl<EG: Numeric, GA: MatmulArgs<EG>> Clone for TensorInput<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident.clone(),
            }
        }
    }
    impl<EG: Numeric, GA: MatmulArgs<EG>> Copy for TensorInput<EG, GA> {}

    impl<EG: Numeric, GA: MatmulArgs<EG>> IntoRuntime for TensorInput<EG, GA> {
        fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}

mod __output {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs<EG>> CubeType for TensorOutput<EG, GA> {
        type ExpandType = TensorOutputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Clone for TensorOutput<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Clone for TensorOutputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Init for TensorOutputExpand<EG, GA> {
        fn init(mut self, context: &mut CubeContext) -> Self {
            self.state = self.state.init(context);
            self
        }
    }

    impl<EG: Numeric, GA: MatmulArgs<EG>> Copy for TensorOutput<EG, GA> {}

    impl<EG: Numeric, GA: MatmulArgs<EG>> IntoRuntime for TensorOutput<EG, GA> {
        fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}
