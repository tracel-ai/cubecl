use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{InputRuntimeArg, MatmulConfigFactory, MatmulSpec, OutputRuntimeArg};

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
// TODO type alias for different uses
// Also those for numbers always <255 should be u8
pub struct MatmulSize {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl From<MatmulSize> for (u32, u32, u32) {
    fn from(matmul_size: MatmulSize) -> Self {
        (matmul_size.m, matmul_size.n, matmul_size.k)
    }
}

impl From<(u32, u32, u32)> for MatmulSize {
    fn from(sizes: (u32, u32, u32)) -> Self {
        MatmulSize {
            m: sizes.0,
            n: sizes.1,
            k: sizes.2,
        }
    }
}

/// Provides launch entry point to solve a matmul
pub trait MatmulLaunch: MatmulConfigFactory {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        size_k: ScalarArg<u32>,
        config: <Self as MatmulConfigFactory>::Config,
    );
}
