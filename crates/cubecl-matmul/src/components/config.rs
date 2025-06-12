use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::kernels::MatmulAvailabilityError;

use super::MatmulPrecision;

pub type InvalidConfigError = Box<dyn Display>;

pub struct FormattedConfigError {
    func: Box<dyn Fn() -> String>,
}

impl FormattedConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + 'static>(func: F) -> Box<dyn Display> {
        Box::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for FormattedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}

/// Provides configuration for a matmul kernel at any level
pub trait MatmulChecker: Send + Sync + 'static {
    type Config: MatmulConfig;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError>;

    /// Checks if the client can handle the features used in this computation
    #[allow(clippy::result_large_err)]
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError>;
}

/// A config for a matmul
///
/// Useful to aggregate many trait bounds
pub trait MatmulConfig:
    Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug
{
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum Ident {
    Lhs,
    Rhs,
    Out,
}

impl Ident {
    pub fn as_input_ident(&self) -> InputIdent {
        match self {
            Ident::Lhs => InputIdent::Lhs,
            Ident::Rhs => InputIdent::Rhs,
            Ident::Out => panic!("Out is not an input."),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for the two input tensors in a matmul.
///
/// Useful to specialize some functions depending on the tensor
pub enum InputIdent {
    Lhs,
    Rhs,
}

impl InputIdent {
    pub fn as_ident(&self) -> Ident {
        match self {
            InputIdent::Lhs => Ident::Lhs,
            InputIdent::Rhs => Ident::Rhs,
        }
    }
}

impl From<InputIdent> for Ident {
    fn from(value: InputIdent) -> Self {
        value.as_ident()
    }
}

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout of a 2D structure such as a tensor, shared memory or slice,
/// used within any matmul kernel level
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

#[cube]
/// Maps the matmul MatrixLayout to cmma's MatrixLayout, for use in Cmma API.
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}

pub trait TensorIdent:
    Clone + Copy + Debug + Hash + PartialEq + Eq + Send + Sync + 'static
{
    const IDENT: Ident;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lhs;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Rhs;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Out;

impl TensorIdent for Lhs {
    const IDENT: Ident = Ident::Lhs;
}

impl TensorIdent for Rhs {
    const IDENT: Ident = Ident::Rhs;
}

impl TensorIdent for Out {
    const IDENT: Ident = Ident::Out;
}
