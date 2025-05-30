use cubecl_core::{CubeCount, CubeDim, ir::Elem};
use std::fmt::Debug;

use crate::matmul::components::{InvalidConfigError, TileSize};

pub enum MatmulLaunchError {
    Unavailable(MatmulAvailabilityError),
    InvalidProblem(MatmulInvalidProblem),
    InvalidConfig(InvalidConfigError),
    Unimplemented(MatmulUnimplementedError),
}

pub enum MatmulAvailabilityError {
    PlaneDimUnknown,
    CubeCountTooBig(CubeCount),
    CubeDimTooBig(CubeDim),
    PlaneDimUnsupported {
        plane_dim: u32,
    },
    PlaneOperationsUnavailable,
    TypesUnavailable {
        input: Elem,
        output: Elem,
    },
    CmmaInstructionUnavailable {
        input: Elem,
        output: Elem,
        size: Option<TileSize>,
    },
    PipelineUnavailable,
    BarrierUnavailable,
    TmaUnavailable,
    DynamicLineSizeUnavailable,
}

pub enum MatmulInvalidProblem {
    ExceededMSize { m: u32, max_m: u32 },
    ExceededNSize { n: u32, max_n: u32 },
    ExceededBatchSize { b: u32, max_b: u32 },
    InvalidLineSizeLhs { size: u32, line_size: u8 },
    InvalidLineSizeRhs { size: u32, line_size: u8 },
    InvalidLineSizeOut { size: u32, line_size: u8 },
}

impl From<MatmulInvalidProblem> for MatmulLaunchError {
    fn from(value: MatmulInvalidProblem) -> Self {
        Self::InvalidProblem(value)
    }
}

impl From<MatmulAvailabilityError> for MatmulLaunchError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Unavailable(value)
    }
}

impl From<InvalidConfigError> for MatmulLaunchError {
    fn from(value: InvalidConfigError) -> Self {
        Self::InvalidConfig(value)
    }
}

impl From<MatmulUnimplementedError> for MatmulLaunchError {
    fn from(value: MatmulUnimplementedError) -> Self {
        Self::Unimplemented(value)
    }
}

impl Debug for MatmulLaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulLaunchError::Unavailable(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because a required feature is unavailable: {err:?}"
                )
            }
            MatmulLaunchError::InvalidProblem(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because the problem isn't correctly defined: {err:?}"
                )
            }
            MatmulLaunchError::InvalidConfig(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because the config is invalid: {:?}",
                    err.to_string()
                )
            }
            MatmulLaunchError::Unimplemented(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because the feature is not ready: {err:?}"
                )
            }
        }
    }
}

impl Debug for MatmulInvalidProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulInvalidProblem::ExceededMSize { m, max_m } => write!(
                f,
                "Problem has m={m} but these configs can only have m<={max_m}"
            ),
            MatmulInvalidProblem::ExceededNSize { n, max_n } => write!(
                f,
                "Problem has n={n} but these configs can only have n<={max_n}",
            ),
            MatmulInvalidProblem::ExceededBatchSize { b, max_b } => write!(
                f,
                "Problem has {b} batches but these configs can only have batches<={max_b}",
            ),
            MatmulInvalidProblem::InvalidLineSizeLhs { size, line_size } => write!(
                f,
                "the lhs tensor can't be read with line size={line_size} and dimension={size}"
            ),
            MatmulInvalidProblem::InvalidLineSizeRhs { size, line_size } => write!(
                f,
                "The rhs tensor can't be read with line size={line_size} and dimension={size}"
            ),
            MatmulInvalidProblem::InvalidLineSizeOut { size, line_size } => write!(
                f,
                "The out tensor can't be written with line size={line_size} and dimension={size}"
            ),
        }
    }
}

impl Debug for MatmulAvailabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulAvailabilityError::PlaneOperationsUnavailable => {
                writeln!(f, "Plane operations not supported.")
            }
            MatmulAvailabilityError::CubeCountTooBig(count) => {
                writeln!(f, "Cube count too big {count:?}")
            }
            MatmulAvailabilityError::CubeDimTooBig(dim) => {
                writeln!(f, "Cube dim too big {dim:?}")
            }
            MatmulAvailabilityError::PlaneDimUnknown => {
                writeln!(f, "Plane dimension unknown.")
            }
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim } => {
                writeln!(
                    f,
                    "Plane dimension unsupported: {plane_dim}. Only 32 & 64 are supported."
                )
            }
            MatmulAvailabilityError::TypesUnavailable { input, output } => {
                writeln!(
                    f,
                    "Types input={input:?} and/or output={output:?} not supported.",
                )
            }
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                input,
                output,
                size: Some(size),
            } => writeln!(
                f,
                "Cmma on inputs {:?} and outputs {:?} with shape m={:?}, n={:?}, k={:?} not supported.",
                input,
                output,
                size.m(),
                size.n(),
                size.k()
            ),
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                input,
                output,
                size: None,
            } => writeln!(f, "Cmma on inputs {input:?} and outputs {output:?}.",),
            MatmulAvailabilityError::PipelineUnavailable => {
                writeln!(f, "Pipeline is not available.")
            }
            MatmulAvailabilityError::BarrierUnavailable => {
                writeln!(f, "Barrier is not available.")
            }
            MatmulAvailabilityError::TmaUnavailable => {
                writeln!(f, "TMA is not available.")
            }
            MatmulAvailabilityError::DynamicLineSizeUnavailable => {
                writeln!(f, "Dynamic line size is not available.")
            }
        }
    }
}

pub enum MatmulUnimplementedError {
    Quantization,
}

impl Debug for MatmulUnimplementedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulUnimplementedError::Quantization => {
                writeln!(f, "Quantization")
            }
        }
    }
}
