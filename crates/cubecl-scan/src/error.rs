use core::fmt;
use cubecl_core::ir::StorageType;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ScanError {
    /// Indicate that the hardware / API doesn't support SIMT plane instructions.
    PlanesUnavailable,
    /// When the cube count is bigger than the max supported.
    CubeCountTooLarge,
    /// Indicate that min_plane_dim != max_plane_dim, thus the exact plane_dim is not fixed.
    ImprecisePlaneDim,
    MismatchSize {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
    /// Indicates that the buffer type is not supported by the backend.
    UnsupportedType(StorageType),
    /// Indicates that we can't launch a decoupled look-back scan
    /// because the atomic load/store operations are not supported.
    MissingAtomicLoadStore(StorageType),
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlanesUnavailable => write!(
                f,
                "Trying to launch a kernel using plane instructions, but there are not supported by the hardware."
            ),
            Self::CubeCountTooLarge => write!(f, "The cube count is larger than the max supported"),
            Self::ImprecisePlaneDim => write!(
                f,
                "Trying to launch a kernel using plane instructions, but the min and max plane dimensions are different."
            ),
            Self::MismatchSize { shape_a, shape_b } => write!(
                f,
                "The tensor of shape {shape_a:?} should have the same number of elements as the one with shape {shape_b:?}."
            ),
            Self::UnsupportedType(ty) => {
                write!(f, "The type {ty} is not supported by the client")
            }
            Self::MissingAtomicLoadStore(ty) => {
                write!(f, "Atomic load/store not supported by the client for {ty}")
            }
        }
    }
}
