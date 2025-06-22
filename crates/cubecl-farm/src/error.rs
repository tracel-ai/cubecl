#[derive(Debug)]
pub enum FarmError {
    DriverError(cudarc::driver::DriverError),
    InvalidDevice,
    InvalidConfiguration,
    InvalidHandle,
    NcclError(cudarc::nccl::result::NcclError),
    InvalidGroup(usize),
    InvalidUnit(usize),
    InvalidDataCount { expected: usize, got: usize },
    GroupAlreadyStarted(usize),
    GroupNotStarted(usize),
    NoNcclLinks(usize),
    ChannelClosed,
    RuntimeError(String),
    InvalidSplitConfiguration,
}

impl From<cudarc::driver::DriverError> for FarmError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        FarmError::DriverError(err)
    }
}

impl From<cudarc::nccl::result::NcclError> for FarmError {
    fn from(err: cudarc::nccl::result::NcclError) -> Self {
        FarmError::NcclError(err)
    }
}
