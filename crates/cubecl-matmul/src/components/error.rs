use cubecl_core::{CubeCount, CubeDim, LineSizeError, ir::StorageType};
use std::fmt::{Debug, Display};

use crate::components::{MatrixLayout, TileSize};

/// Errors that can occur during the setup phase of a matmul operation.
pub enum MatmulSetupError {
    /// A required hardware or runtime feature is not available.
    Unavailable(MatmulAvailabilityError),

    /// The provided configuration is invalid or rejected by a component.
    InvalidConfig(InvalidConfigError),

    /// No compatible line size could be found for the given constraints.
    LineSize(LineSizeError),
}

/// A specific feature required for matmul is not available in the current runtime or hardware.
pub enum MatmulAvailabilityError {
    /// The requested cube count exceeds what the runtime or hardware supports.
    CubeCountTooBig(CubeCount),

    /// The requested cube dimensions are too large for the current runtime or hardware.
    CubeDimTooBig(CubeDim),

    /// The requested plane dimension is not supported.
    PlaneDimUnsupported { plane_dim: u32 },

    /// The required data types for input or output are not supported.
    TypesUnavailable {
        lhs: StorageType,
        rhs: StorageType,
        output: StorageType,
    },

    /// The required CMMA instruction is not supported for the given element types and tile size.
    CmmaInstructionUnavailable {
        lhs: StorageType,
        rhs: StorageType,
        output: StorageType,
        size: Option<TileSize>,
    },

    /// The layout of the matmul is unsupported
    LayoutUnsupported {
        lhs: MatrixLayout,
        rhs: MatrixLayout,
    },

    /// Barrier synchronization is not available in the runtime.
    BarrierUnavailable,

    /// TMA (Tensor Memory Access) is not available in the runtime.
    TmaUnavailable,

    /// Dynamic selection of line size is unsupported in the current runtime.
    DynamicLineSizeUnavailable,

    /// Plane operations like plane_sum are unavailable
    PlaneOpsUnavailable,
}
impl From<MatmulAvailabilityError> for MatmulSetupError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Unavailable(value)
    }
}

impl From<InvalidConfigError> for MatmulSetupError {
    fn from(value: InvalidConfigError) -> Self {
        Self::InvalidConfig(value)
    }
}

impl From<LineSizeError> for MatmulSetupError {
    fn from(value: LineSizeError) -> Self {
        Self::LineSize(value)
    }
}

impl Display for MatmulSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Debug for MatmulSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulSetupError::Unavailable(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because a required feature is unavailable: {err:?}"
                )
            }
            MatmulSetupError::InvalidConfig(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because the config is invalid: {:?}",
                    err.to_string()
                )
            }
            MatmulSetupError::LineSize(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because could not find supported line size: {err:?}"
                )
            }
        }
    }
}

impl Debug for MatmulAvailabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulAvailabilityError::CubeCountTooBig(count) => {
                writeln!(f, "Cube count too big {count:?}")
            }
            MatmulAvailabilityError::CubeDimTooBig(dim) => {
                writeln!(f, "Cube dim too big {dim:?}")
            }
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim } => {
                writeln!(
                    f,
                    "Plane dimension unsupported: {plane_dim}. Only 32 & 64 are supported."
                )
            }
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output } => {
                writeln!(
                    f,
                    "Types lhs={lhs:?}, rhs={rhs:?} and/or output={output:?} not supported.",
                )
            }
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output,
                size: Some(size),
            } => writeln!(
                f,
                "Cmma on lhs {:?} rhs {:?} and output {:?} with shape m={:?}, n={:?}, k={:?} not supported.",
                lhs,
                rhs,
                output,
                size.m(),
                size.n(),
                size.k()
            ),
            MatmulAvailabilityError::LayoutUnsupported { lhs, rhs } => {
                writeln!(
                    f,
                    "Cmma with layouts lhs {lhs:?} and rhs {rhs:?} not supported."
                )
            }
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output,
                size: None,
            } => writeln!(
                f,
                "Cmma on inputs lhs {lhs:?} rhs {rhs:?} and output {output:?} not supported.",
            ),
            MatmulAvailabilityError::BarrierUnavailable => {
                writeln!(f, "Barrier is not available.")
            }
            MatmulAvailabilityError::TmaUnavailable => {
                writeln!(f, "TMA is not available.")
            }
            MatmulAvailabilityError::DynamicLineSizeUnavailable => {
                writeln!(f, "Dynamic line size is not available.")
            }
            MatmulAvailabilityError::PlaneOpsUnavailable => {
                writeln!(f, "Plane-wide operations like plane_sum are not available.")
            }
        }
    }
}

/// Error that arises from invalid configurations
pub type InvalidConfigError = Box<dyn Display>;

/// Error that arises from invalid configurations
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
