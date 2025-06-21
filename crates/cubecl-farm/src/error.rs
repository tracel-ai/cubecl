#[derive(Debug)]
pub enum FarmError {
    DriverError(cudarc::driver::DriverError),
    InvalidDevice,
    InvalidConfiguration,
    NcclError(cudarc::nccl::result::NcclError),
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
