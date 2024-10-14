#[derive(new, Clone, PartialEq, Eq, Default, Hash)]
pub struct CudaDevice {
    pub index: usize,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cuda({})", self.index)
    }
}
