use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    AccG, AvailableLineSizes, LhsG, MatmulElems, MatmulLineSizes, MatmulPrecision, MatmulSelection,
    MatmulSetupError, RhsG,
    global::GlobalWriter,
    stage::{ContiguousTilingLayout, RowMajorTilingOrder},
};
use cubecl_std::{
    CubeOption,
    tensor::{View, layout::Coords2d},
};

use crate::components::{
    ConvGemmConfig, ConvolutionProblem,
    global::{args::RuntimeArgs, entry_point::ConvolutionLaunch},
};

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

pub type GlobalConfig<F> = <F as GlobalConvolutionFamily>::Config;

pub trait GlobalConvolutionFamily: ConvolutionLaunch<Self::Config> + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: ConvGemmConfig;
    type Convolution<MP: MatmulPrecision>: GlobalConvolution<MP, Config = Self::Config>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes;

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError>;
}

#[cube]
pub trait GlobalConvolution<MP: MatmulPrecision>: 'static + Send + Sync {
    /// The global reader for the Lhs (input feature map) tensor
    type LhsGlobalReader: CubeType;
    /// The global reader for the Rhs (weight) tensor
    type RhsGlobalReader: CubeType;
    /// The global reader for the accumulator (bias) tensor
    type AccGlobalReader: CubeType;
    /// The config type of the convolution
    type Config: ConvGemmConfig;

    /// The writer used to write the results to the output feature map
    type GlobalWriter: GlobalWriter<MP::Acc>;
    /// The type of the tile matmul accumulator
    type Accumulators: CubeType;

    /// Performs the convolution over data loaded by the
    /// LHS and RHS readers, over the range given for K, and stores with
    /// using the output writer.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_reader: Self::LhsGlobalReader,
        rhs_reader: Self::RhsGlobalReader,
        acc_reader: Self::AccGlobalReader,
        writer: Self::GlobalWriter,
        acc: &mut Self::Accumulators,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    );

    /// Initializes the global reader for the input feature map with an appropriate layout
    fn init_lhs_global_reader(
        lhs: View<Line<LhsG<MP>>, Coords2d>,
        offset: Coords2d,
        slice_size: Coords2d,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsGlobalReader;

    /// Initializes the global reader for the weights with an appropriate layout
    fn init_rhs_global_reader(
        rhs: View<Line<RhsG<MP>>, Coords2d>,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::RhsGlobalReader;

    /// Initializes the global reader for the bias with an appropriate layout
    fn init_bias_global_reader(
        bias: CubeOption<View<Line<AccG<MP>>, Coords2d>>,
        #[comptime] config: Self::Config,
    ) -> Self::AccGlobalReader;

    /// Initializes the output feature map global writer with an appropriate layout
    fn init_global_writer(
        out: View<Line<AccG<MP>>, Coords2d, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::GlobalWriter;

    /// Initializes a new accumulator for the tile matmul
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulators;
}
