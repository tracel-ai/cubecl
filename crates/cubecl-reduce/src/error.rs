use core::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum ReduceError {
    PlanesUnavailable, // Indicate that the hardware / API doesn't support SIMT plane instructions.
    ImprecisePlaneDim, // Indicate that min_plane_dim != max_plane_dim, thus the exact plane_dim is not fixed.
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlanesUnavailable => write!(f, "Trying to launch a kernel using plane instructions, but there are not supported by the hardware."),
            Self::ImprecisePlaneDim => write!(f, "Trying to launch a kernel using plane instructions, but the min and max plane dimensions are differents.")
        }
    }
}

impl std::error::Error for ReduceError {}
