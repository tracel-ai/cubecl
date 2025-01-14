use core::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ReduceError {
    /// Indicate that the hardware / API doesn't support SIMT plane instructions.
    PlanesUnavailable,
    /// When the cube count is bigger than the max supported.
    CubeCountTooLarge,
    /// Indicate that min_plane_dim != max_plane_dim, thus the exact plane_dim is not fixed.
    ImprecisePlaneDim,
    /// Indicate the axis is too large.
    InvalidAxis { axis: usize, rank: usize },
    /// Indicate that the shape of the output tensor is invalid for the given input and axis.
    MismatchShape {
        expected_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlanesUnavailable => write!(f, "Trying to launch a kernel using plane instructions, but there are not supported by the hardware."),
            Self::CubeCountTooLarge => write!(f, "The cube count is larger than the max supported."),
            Self::ImprecisePlaneDim => write!(f, "Trying to launch a kernel using plane instructions, but the min and max plane dimensions are different."),
            Self::InvalidAxis{axis, rank} => write!(f, "The provided axis ({axis}) must be smaller than the input tensor rank ({rank})."),
            Self::MismatchShape { expected_shape, output_shape } => {
                write!(f, "The output shape (currently {output_shape:?}) should be {expected_shape:?}.")
            }
        }
    }
}

impl std::error::Error for ReduceError {}
