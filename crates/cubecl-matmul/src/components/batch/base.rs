use crate::{
    components::{
        AvailableLineSizes, InputRuntimeArg, MatmulLineSizes, MatmulPrecision, MatmulProblem,
        MatmulSpec, OutputRuntimeArg, TilingScheme,
        batch::Partitioner,
        config::MatmulConfig,
        global::{self, GlobalConfig as _, Quantization},
    },
    kernels::{MatmulSetupError, matmul::MatmulSelection},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily: 'static + Send + Sync {
    type Matmul<MP: MatmulPrecision>: BatchMatmul<MP, Config = Self::Config>;
    type Partitioner: Partitioner;
    type Config: BatchConfig;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

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
        config: Self::Config,
    );
}

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes can collaborate to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
/// - Enough Cubes are launched to perform the whole computation.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important to use an underlying global matmul that performs check bounds,
/// and to not launch more Cubes than necessary.
pub trait BatchMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig: MatmulConfig {
    /// Underlying Global matmul config
    type GlobalConfig: global::GlobalConfig;

    /// Convert itself to the underlying global matmul config
    fn global_config(&self) -> Self::GlobalConfig;

    /// Returns true if the matmul is quantized.
    fn quantized(&self) -> bool;

    fn tiling_scheme(&self) -> TilingScheme {
        self.global_config().tiling_scheme()
    }

    // TODO make a launch config over batch config
    fn cube_dim(&self) -> CubeDim;
    // fn cube_count(&self) -> CubeCount;
    fn line_sizes(&self) -> MatmulLineSizes;
}
