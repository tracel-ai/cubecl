use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    prelude::{CubePrimitive, TensorHandleRef},
};

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub clear_cube_dim: CubeDim,
    pub line_size: u32,
    pub clear_line_size: u32,
    pub inclusive: bool,
}

impl ScanConfig {
    pub(crate) fn generate<R: Runtime, N: CubePrimitive>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: &TensorHandleRef<R>,
        output: &TensorHandleRef<R>,
    ) -> ScanConfig {
        // ToDo
        ScanConfig::empty()
    }

    fn empty() -> Self {
        Self {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            clear_cube_dim: CubeDim::new_single(),
            line_size: 1,
            clear_line_size: 1,
            inclusive: false,
        }
    }

    pub fn with_inclusive(mut self, inclusive: bool) -> Self {
        self.inclusive = inclusive;
        self
    }
}
