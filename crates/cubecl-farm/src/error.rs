#[derive(Debug)]
pub enum CudaError {
    DriverError(cudarc::driver::DriverError),
    InvalidDevice,
    InvalidConfiguration,
    NcclError(cudarc::nccl::result::NcclError),
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        CudaError::DriverError(err)
    }
}

impl From<cudarc::nccl::result::NcclError> for CudaError {
    fn from(err: cudarc::nccl::result::NcclError) -> Self {
        CudaError::NcclError(err)
    }
}
