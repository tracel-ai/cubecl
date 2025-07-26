use std::fmt::Debug;

use cubecl_core::tune::AutotuneError;
use cubecl_matmul::components::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_runtime::storage::AllocError;

#[allow(clippy::large_enum_variant)]
pub enum ConvLaunchError {
    Matmul(MatmulSetupError),
    Groups(usize),
    Memory(AllocError),
    Unknown,
}

impl Debug for ConvLaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvLaunchError::Matmul(err) => {
                write!(f, "{err:?}")
            }
            ConvLaunchError::Groups(groups) => {
                writeln!(
                    f,
                    "Unable to launch matmul because groups must be one, is actually {groups}",
                )
            }
            ConvLaunchError::Memory(err) => {
                write!(f, "Failed to allocate memory: {err:?}")
            }
            ConvLaunchError::Unknown => write!(f, "Unknown"),
        }
    }
}

impl From<MatmulSetupError> for ConvLaunchError {
    fn from(value: MatmulSetupError) -> Self {
        Self::Matmul(value)
    }
}

impl From<MatmulAvailabilityError> for ConvLaunchError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Matmul(MatmulSetupError::Unavailable(value))
    }
}

impl From<AllocError> for ConvLaunchError {
    fn from(value: AllocError) -> Self {
        Self::Memory(value)
    }
}

#[allow(clippy::from_over_into)]
impl Into<AutotuneError> for ConvLaunchError {
    fn into(self) -> AutotuneError {
        AutotuneError::Unknown(format!("{self:?}"))
    }
}
