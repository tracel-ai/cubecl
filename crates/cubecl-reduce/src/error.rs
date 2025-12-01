use cubecl_core::{ir::StorageType, server::LaunchError};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
/// This error should be catched and properly handled.
pub enum ReduceError {
    /// Indicate that the hardware / API doesn't support SIMT plane instructions.
    #[error(
        "Trying to launch a kernel using plane instructions, but there are not supported by the hardware."
    )]
    PlanesUnavailable,
    /// When the cube count is bigger than the max supported.
    #[error("The cube count is larger than the max supported.")]
    CubeCountTooLarge,
    /// Indicate that min_plane_dim != max_plane_dim, thus the exact plane_dim is not fixed.
    #[error(
        "Trying to launch a kernel using plane instructions, but the min and max plane dimensions are different."
    )]
    ImprecisePlaneDim,
    /// Indicate the axis is too large.
    #[error("The provided axis ({axis}) must be smaller than the input tensor rank ({rank}).")]
    InvalidAxis { axis: usize, rank: usize },
    /// Indicate that the shape of the output tensor is invalid for the given input and axis.
    #[error("The output shape (currently {output_shape:?}) should be {expected_shape:?}.")]
    MismatchShape {
        expected_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Indicate that we can't launch a shared sum because the atomic addition is not supported.
    #[error("Atomic add not supported by the client for {0}")]
    MissingAtomicAdd(StorageType),

    /// An error happened during launch.
    #[error("An error happened during launch: {0}")]
    Launch(LaunchError),
}
