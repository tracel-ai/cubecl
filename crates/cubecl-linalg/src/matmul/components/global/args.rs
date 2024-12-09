use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
pub trait GmmArgs<EG: Numeric>: Send + Sync + 'static + Clone {
    type Output: LaunchArg + CubeType;
    type Input: LaunchArg + CubeType;
    type State: CubeType;

    fn init_state(input: &Self::Input, output: &mut Self::Output) -> Self::State;

    fn read_lhs(state: &Self::State, coordinate: u32) -> Line<EG>;
    fn read_rhs(state: &Self::State, coordinate: u32) -> Line<EG>;

    fn shape_lhs(state: &Self::State, dim: u32) -> u32;
    fn shape_rhs(state: &Self::State, dim: u32) -> u32;
    fn shape_out(state: &Self::State, dim: u32) -> u32;

    fn stride_lhs(state: &Self::State, dim: u32) -> u32;
    fn stride_rhs(state: &Self::State, dim: u32) -> u32;
    fn stride_out(state: &Self::State, dim: u32) -> u32;

    fn write_out(state: &mut Self::State, coordinate: u32, value: Line<EG>);
}

#[derive(Clone, Copy)]
pub enum Ident {
    Lhs,
    Rhs,
}

pub struct TensorInput<EG: Numeric, GA: GmmArgs<EG>> {
    state: *const GA::State,
    ident: Ident,
}

impl<EG: Numeric, GA: GmmArgs<EG>> IntoRuntime for TensorInput<EG, GA> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        panic!("Can't")
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> IntoRuntime for TensorOutput<EG, GA> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        panic!("Can't")
    }
}

pub struct TensorInputExpand<EG: Numeric, GA: GmmArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
    ident: Ident,
}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorInputExpand<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            ident: self.ident.clone(),
        }
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> Init for TensorInputExpand<EG, GA> {
    fn init(mut self, context: &mut CubeContext) -> Self {
        self.state = self.state.init(context);
        self
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> CubeType for TensorOutput<EG, GA> {
    type ExpandType = TensorOutputExpand<EG, GA>;
}

pub struct TensorOutput<EG: Numeric, GA: GmmArgs<EG>> {
    state: *mut GA::State,
}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorInput<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            ident: self.ident.clone(),
        }
    }
}
impl<EG: Numeric, GA: GmmArgs<EG>> Copy for TensorInput<EG, GA> {}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorOutput<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}
impl<EG: Numeric, GA: GmmArgs<EG>> Copy for TensorOutput<EG, GA> {}

pub struct TensorOutputExpand<EG: Numeric, GA: GmmArgs<EG>> {
    state: <GA::State as CubeType>::ExpandType,
}

impl<EG: Numeric, GA: GmmArgs<EG>> Clone for TensorOutputExpand<EG, GA> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> Init for TensorOutputExpand<EG, GA> {
    fn init(mut self, context: &mut CubeContext) -> Self {
        self.state = self.state.init(context);
        self
    }
}

impl<EG: Numeric, GA: GmmArgs<EG>> CubeType for TensorInput<EG, GA> {
    type ExpandType = TensorInputExpand<EG, GA>;
}

#[cube]
impl<EG: Numeric, GA: GmmArgs<EG>> TensorInput<EG, GA> {
    pub fn new(state: &GA::State, #[comptime] ident: Ident) -> TensorInput<EG, GA> {
        TensorInput::<EG, GA> { state, ident }
    }

    pub fn read(&self, coordinate: u32) -> Line<EG> {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::read_lhs(&(*self.state), coordinate),
                Ident::Rhs => GA::read_rhs(&(*self.state), coordinate),
            }
        }
    }
    pub fn shape(&self, dim: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::shape_lhs(&(*self.state), dim),
                Ident::Rhs => GA::shape_rhs(&(*self.state), dim),
            }
        }
    }
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                Ident::Lhs => GA::stride_lhs(&(*self.state), dim),
                Ident::Rhs => GA::stride_rhs(&(*self.state), dim),
            }
        }
    }
}

#[cube]
impl<EG: Numeric, GA: GmmArgs<EG>> TensorOutput<EG, GA> {
    pub fn new(state: &mut GA::State) -> TensorOutput<EG, GA> {
        TensorOutput::<EG, GA> { state }
    }

    pub fn write(&self, coordinate: u32, value: Line<EG>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, value) }
    }
    pub fn shape(&self, dim: u32) -> u32 {
        unsafe { GA::shape_out(&mut (*self.state), dim) }
    }
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { GA::stride_out(&mut (*self.state), dim) }
    }
}

#[derive(CubeLaunch)]
pub struct TensorInputs<EG: Numeric> {
    pub lhs: Tensor<Line<EG>>,
    pub rhs: Tensor<Line<EG>>,
}

#[derive(Clone)]
pub struct TensorArgs;

#[cube]
impl<EG: Numeric> GmmArgs<EG> for TensorArgs {
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
}
