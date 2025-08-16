use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    AvailableLineSizes, LhsG, MatmulLineSizes, MatmulPrecision, MatmulSelection, MatmulSetupError,
    RhsG,
    global::{AccumulatorLoader, GlobalWriter},
};
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use crate::{
    components::{ConvGemmConfig, ConvolutionProblem, global::entry_point::ConvolutionLaunch},
    kernels::layered::selector::RuntimeArgs,
};

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
    type LhsLoader: CubeType;
    type RhsLoader: CubeType;
    type Config: ConvGemmConfig;
    type AccumulatorLoader: AccumulatorLoader<MP>;

    type Writer: GlobalWriter<MP::EO>;
    type Accumulator: CubeType;

    /// Performs the convolution over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        acc_loader: Self::AccumulatorLoader,
        writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    fn init_lhs_loader(
        lhs: VirtualTensor<LhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    fn init_rhs_loader(
        rhs: VirtualTensor<RhsG<MP>>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader;

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<MP::EO>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccumulatorLoader;

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Writer;

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;
}
