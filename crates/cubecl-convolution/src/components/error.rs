use core::fmt::Debug;
use cubecl_core::server::LaunchError;
use cubecl_matmul::components::{MatmulAvailabilityError, MatmulSetupError};

#[allow(clippy::large_enum_variant)]
pub enum ConvSetupError {
    Matmul(MatmulSetupError),
    Groups(usize),
    Unknown,
    Launch(LaunchError),
}

impl From<LaunchError> for ConvSetupError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
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
            ConvSetupError::Launch(err) => write!(f, "Launch error {err:?}"),
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
impl Into<String> for ConvSetupError {
    fn into(self) -> String {
        format!("{self:?}")
    }
}
