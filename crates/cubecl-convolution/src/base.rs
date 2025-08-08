use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    AvailableLineSizes, InputRuntimeArg, LhsG, MatmulLineSizes, MatmulPrecision, MatmulProblem,
    MatmulSelection, MatmulSetupError, MatmulSpec, MatrixLayout, OutputRuntimeArg, RhsG,
    global::{AccumulatorLoader, GlobalWriter},
};
use cubecl_std::{
    CubeOption, FastDivmod,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::ConvGemmConfig;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub size_m: u32,
    pub size_n: u32,
    pub size_k: u32,
    pub padded_channels: FastDivmod,
    pub out_shape: Sequence<FastDivmod>,
}

pub trait ConvolutionFamily:
    ConvolutionConfigFactory<Config: ConvGemmConfig> + ConvolutionLaunch
{
    type Convolution<MP: MatmulPrecision>: Convolution<MP, Config = Self::Config>;

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes;
}

#[cube]
pub trait Convolution<MP: MatmulPrecision>: 'static + Send + Sync {
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

/// Provides configuration for a matmul kernel at any level
pub trait ConvolutionConfigFactory: Send + Sync + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: ConvGemmConfig;

    fn setup<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError>;
}

/// Provides launch entry point to solve a matmul
pub trait ConvolutionLaunch: ConvolutionConfigFactory {
    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        bias: Option<TensorArg<'a, R>>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: &ConvolutionProblem,
        config: <Self as ConvolutionConfigFactory>::Config,
    );
}

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,

    pub kernel_size: Vec<u32>,
    pub stride: Vec<u32>,
    pub padding: Vec<i32>,
    pub dilation: Vec<u32>,

    pub batches: usize,
    pub channels: usize,
    pub shape: Vec<usize>,
    pub out_shape: Vec<usize>,

    pub dimensionality: Dimensionality,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: vec![],
            rhs_batches: vec![],
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
        }
    }
}

/// Spatial dimensionality of an operation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Dimensionality {
    Dim1,
    Dim2,
    Dim3,
}
