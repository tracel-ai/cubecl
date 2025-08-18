use std::fmt::Debug;

use cubecl_core::tune::AutotuneError;
use cubecl_matmul::components::{MatmulAvailabilityError, MatmulSetupError};

#[allow(clippy::large_enum_variant)]
pub enum ConvSetupError {
    Matmul(MatmulSetupError),
    Groups(usize),
    Unknown,
}

impl Debug for ConvSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvSetupError::Matmul(err) => {
                write!(f, "{err:?}")
            }
            ConvSetupError::Groups(groups) => {
                writeln!(
                    f,
                    "Unable to launch matmul because groups must be one, is actually {groups}",
                )
            }
            ConvSetupError::Unknown => write!(f, "Unknown"),
        }
    }
}

impl From<MatmulSetupError> for ConvSetupError {
    fn from(value: MatmulSetupError) -> Self {
        Self::Matmul(value)
    }
}

impl From<MatmulAvailabilityError> for ConvSetupError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Matmul(MatmulSetupError::Unavailable(value))
    }
}

#[allow(clippy::from_over_into)]
impl Into<AutotuneError> for ConvSetupError {
    fn into(self) -> AutotuneError {
        AutotuneError::Unknown(format!("{self:?}"))
    }
}
