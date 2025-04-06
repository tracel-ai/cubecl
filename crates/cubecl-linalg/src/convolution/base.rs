use crate::matmul::{
    components::{
        InputRuntimeArg, InvalidConfigError, MatmulPrecision, MatmulProblem, MatmulSpec,
        MatrixLayout, OutputRuntimeArg,
        global::{AccumulatorLoader, OutputLoader},
    },
    kernels::MatmulAvailabilityError,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, FastDivmod,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use super::ConvGemmConfig;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub padded_channels: FastDivmod,
    pub out_h: u32,
    pub out_w: u32,
}

pub trait ConvolutionFamily:
    ConvolutionConfigFactory<Config: ConvGemmConfig> + ConvolutionLaunch
{
    type Convolution<MP: MatmulPrecision>: Convolution<MP, Config = Self::Config>;
}

#[cube]
pub trait Convolution<MP: MatmulPrecision>: 'static + Send + Sync {
    type LhsLoader: CubeType;
    type RhsLoader: CubeType;
    type Config: ConvGemmConfig;
    type AccumulatorLoader: AccumulatorLoader<MP>;

    type Out: OutputLoader<MP::EO>;
    type Accumulator: CubeType;

    /// Performs the convolution over data loaded by the
    /// LHS and RHS loaders, over the range given for K, and stores with
    /// using the output unloader.
    ///
    /// To compute the whole range of k values, use k_range=(0, K) where
    /// K is the K dimension of LHS and RHS.
    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        acc_loader: Self::AccumulatorLoader,
        unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        runtime_args: RuntimeArgs,
        #[comptime] config: Self::Config,
    );

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader;

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
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

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Out;

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator;
}

/// Provides configuration for a matmul kernel at any level
pub trait ConvolutionConfigFactory: Send + Sync + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: ConvGemmConfig;
    type Input;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError>;

    fn make_config(
        input: Self::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Self::Config;

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError>;
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
        config: <Self as ConvolutionConfigFactory>::Config,
    );
}

#[derive(Clone)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub lhs_line_size: u8,
    pub rhs_line_size: u8,
    pub out_line_size: u8,

    pub padded_channels: u32,
    pub kernel_size: (u32, u32),
    pub stride: (u32, u32),
    pub padding: (i32, i32),
    pub dilation: (u32, u32),

    pub batches: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,

    pub out_h: usize,
    pub out_w: usize,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            batches: (vec![], vec![]),
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            lhs_line_size: self.lhs_line_size,
            rhs_line_size: self.rhs_line_size,
            out_line_size: self.out_line_size,
        }
    }
}
