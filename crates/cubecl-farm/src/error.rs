use thiserror::Error;

pub type Result<T> = core::result::Result<T, FarmError>;

#[derive(Error, Debug)]
pub enum FarmError {
    #[error("The choosen split configuration device count is not equal to your devices")]
    InvalidSplitConfiguration,
    #[error("Warning: Proportional split has no valid proportions. Falling back to SingleGroup.")]
    ProportionFallback,
    #[error("Unit index not found")]
    InvalidDevice,
    #[error(
        "Warning: Explicit split {sum} does not match device count {device_count}. Falling back to SingleGroup."
    )]
    ExplicitFallback { sum: usize, device_count: usize },
}
