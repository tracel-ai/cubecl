use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    AccG, AvailableLineSizes, LhsG, MatmulLineSizes, MatmulPrecision, MatmulSelection,
    MatmulSetupError, RhsG,
    global::StageWriter,
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
};
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};

use crate::{
    components::{ConvGemmConfig, ConvolutionProblem, global::entry_point::ConvolutionLaunch},
    kernels::layered::selector::RuntimeArgs,
};

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

pub type GlobalConfig<F> = <F as GlobalConvolutionFamily>::Config;

pub trait GlobalConvolutionFamily: ConvolutionLaunch<Self::Config> + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: ConvGemmConfig;
    type Convolution<MP: MatmulPrecision>: GlobalConvolution<MP, Config = Self::Config>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes;

    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;
}

#[cube]
pub trait GlobalConvolution<MP: MatmulPrecision>: 'static + Send + Sync {
    /// The loader for the Lhs (input feature map) tensor
    type LhsStageLoader: CubeType;
    /// The loader for the Rhs (weight) tensor
    type RhsStageLoader: CubeType;
    /// The loader for the accumulator (bias) tensor
    type AccStageLoader: CubeType;
    /// The config type of the convolution
    type Config: ConvGemmConfig;

    /// The writer used to write the results to the output feature map
    type StageWriter: StageWriter<AccG<MP>>;
    /// The type of the tile matmul accumulator
    type Accumulators: CubeType;

    /// Performs the convolution over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::LhsStageLoader,
        rhs_loader: Self::RhsStageLoader,
        acc_loader: Self::AccStageLoader,
        writer: Self::StageWriter,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initializes the loader for the input feature map with an appropriate layout
    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsStageLoader;

    /// Initializes the loader for the weights with an appropriate layout
    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsStageLoader;

    /// Initializes the loader for the bias with an appropriate layout
    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<AccG<MP>>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccStageLoader;

    /// Initializes the output feature map loader with an appropriate layout
    fn init_writer(
        out: VirtualTensor<AccG<MP>, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::StageWriter;

    /// Initializes a new accumulator for the tile matmul
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators;
}
