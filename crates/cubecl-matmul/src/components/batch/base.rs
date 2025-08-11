use crate::components::{
    AvailableLineSizes, InputRuntimeArg, LhsG, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSelection, MatmulSpec, OutputRuntimeArg, RhsG, TilingScheme,
    batch::{CubeCountInput, CubeCountInputArgs, HypercubeConfig},
    error::MatmulSetupError,
    global::{self, GlobalConfig as _},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::{fmt::Debug, hash::Hash};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily: 'static + Send + Sync {
    /// The specific [BatchMatmul] implementation associated with this family.
    type Matmul<MP: MatmulPrecision>: BatchMatmul<MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: BatchConfig;

    /// Constructs the configuration based on the matmul problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
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
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
    );

    /// Filters out line sizes that are incompatible with this matmul family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
}

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes are used to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// - It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
///   It is therefore important to use an underlying global matmul that performs check bounds,
/// - It is accepted to launch more Cube than necessary, providing a CubeCountInput that states
///   the max cube position
pub trait BatchMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: VirtualTensor<LhsG<MP>>,
        rhs: VirtualTensor<RhsG<MP>>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    /// Underlying Global matmul config
    type GlobalConfig: global::GlobalConfig;

    /// Convert itself to the underlying global matmul config
    fn global_config(&self) -> Self::GlobalConfig;

    /// Returns the [TilingScheme]
    fn tiling_scheme(&self) -> TilingScheme {
        self.global_config().tiling_scheme()
    }

    /// Returns the [CubeDim]
    fn cube_dim(&self) -> CubeDim;

    /// Returns the line sizes for Lhs, Rhs and output
    fn line_sizes(&self) -> MatmulLineSizes;

    /// Returns the [HypercubeConfig]
    fn hypercube_config(&self) -> HypercubeConfig;

    /// Whether it may launch more cubes than the minimum required
    fn can_yield_extra_cubes(&self) -> bool;
}
