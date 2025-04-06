use config::HomogeneousConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};
use std::marker::PhantomData;

use crate::matmul::components::{
    EA, EI, EO, ES, InputRuntimeArg, MatmulSpec, OutputRuntimeArg, Rhs,
    global::{
        args::{self, MatmulArgs, TensorInput, TensorOutput},
        single_stage::{FullLoader, SyncFullLoader},
    },
    stage::multi_buffer::StageReaderFamily,
};
use crate::matmul::components::{
    Ident, InvalidConfigError, Lhs, MatrixLayout,
    global::{
        self, AccumulatorLoader, GlobalConfig,
        output_loader::Unloader,
        single_stage::{self, CyclicCoalescedLoading, SyncFullRhsLoader},
    },
    stage::{
        self, ContiguousTilingLayout, RowMajorTilingOrder, StageMatmulFamily, multi_buffer::Reader,
    },
};
use crate::{
    convolution::{
        base::{
            Convolution, ConvolutionConfigFactory, ConvolutionFamily, ConvolutionLaunch,
            ConvolutionProblem,
        },
        config::ConvGemmConfig,
        loader::{bias::BiasLoader, im2col::SimpleIm2colLoader},
    },
    matmul::components::MatmulPrecision,
};

pub struct ImplicitGemmConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

pub type ConvTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

impl<SMM> ConvolutionFamily<SMM> for ImplicitGemmConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<LhsReader = StageReaderFamily<Lhs>, RhsReader = StageReaderFamily<Rhs>>,
{
    type Convolution<MP: MatmulPrecision> =
        ImplicitGemmConvolution<MP, SMM::Matmul<MP, ConvTilingLayout, ConvTilingLayout>>;
}

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct ImplicitGemmConvolution<MP: MatmulPrecision, SMM: stage::StageMatmul<MP>> {
    _cs: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> Convolution<MP, SMM> for ImplicitGemmConvolution<MP, SMM>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = Reader<Lhs, MP::ES, ConvTilingLayout>,
            RhsReader = Reader<Rhs, MP::ES, ConvTilingLayout>,
        >,
{
    type LhsLoader = SimpleIm2colLoader<MP, Self::Config>;
    type Config = HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type RhsLoader =
        SyncFullRhsLoader<MP, SMM::Config, CyclicCoalescedLoading<RowMajorTilingOrder>>;
    type AccumulatorLoader = BiasLoader<MP>;

    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        Self::AccumulatorLoader::fill_stage::<SMM::Config>(&mut acc_loader, config.to_smm_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        sync_units();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.to_smm_config(),
        );

        for _ in 0..num_loops {
            sync_units();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config.to_matmul_config());

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                CubeOption::new_None(),
                config.to_smm_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        sync_units();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            CubeOption::new_None(),
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs,
            config.out_shape(0),
            config.out_shape(1),
            x_offset,
            y_offset,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(rhs, x_offset, y_offset, 0, config)
    }

    fn init_bias_loader(
        bias: CubeOption<VirtualTensor<MP::EO>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::AccumulatorLoader {
        Self::AccumulatorLoader::new::<SMM::Config>(bias, n_offset, config.to_smm_config())
    }

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }
}

impl<SMM> ConvolutionConfigFactory for ImplicitGemmConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily,
{
    type Config = config::HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type Input = SMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn make_config(
        input: Self::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Self::Config {
        let smm_config = SMM::make_config(
            input,
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            false,
        );
        let size = SMM::stage_shape(&smm_config);

        config::HomogeneousConfig::new(
            single_stage::Config::new(
                smm_config,
                // TODO: Find the correct condition to avoid check bounds.
                true,
                true,
                true,
                problem.lhs_layout,
                problem.rhs_layout,
                problem.lhs_line_size as u32,
                problem.rhs_line_size as u32,
                problem.out_line_size as u32,
                size.k,
            ),
            (problem.out_h as u32, problem.out_w as u32),
            problem.padded_channels,
            problem.kernel_size,
            problem.stride,
            problem.dilation,
            problem.padding,
        )
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), crate::matmul::kernels::MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
    }
}

impl<SMM: StageMatmulFamily<LhsReader = StageReaderFamily<Lhs>, RhsReader = StageReaderFamily<Rhs>>>
    ConvolutionLaunch for ImplicitGemmConvolutionFamily<SMM>
{
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        bias: Option<TensorArg<'a, R>>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: <Self as ConvolutionConfigFactory>::Config,
    ) {
        unsafe {
            implicit_conv::launch_unchecked::<MS::Args, EI<MS>, ES<MS>, EA<MS>, EO<MS>, Self, SMM, R>(
                client,
                cube_count,
                cube_dim,
                input,
                bias.into(),
                output,
                config,
            );
        }
    }
}

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

#[cube(launch_unchecked)]
pub(crate) fn implicit_conv<
    Args: MatmulArgs,
    EI: Numeric,
    ES: Numeric,
    EA: Numeric,
    EO: Numeric,
    GMM: ConvolutionFamily<SMM>,
    SMM: StageMatmulFamily,
>(
    inputs: &Input<Args, EI>,
    bias: &CubeOption<Tensor<Line<EO>>>,
    output: &mut Output<Args, EO>,
    #[comptime] config: GMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Lhs);
    let rhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Rhs);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let lhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    let x_offset = CUBE_POS_X * config.tiling_dimensions(Ident::Lhs).total_row();
    let y_offset = CUBE_POS_Y * config.tiling_dimensions(Ident::Rhs).total_col();
    let k_range = (0, rhs.shape(0));

    let bias = match bias {
        CubeOption::Some(bias) => {
            CubeOption::new_Some(VirtualTensor::<EO>::new::<Tensor<Line<EO>>>(bias))
        }
        CubeOption::None => CubeOption::new_None(),
    };

    GMM::Convolution::<(EI, ES, EA, EO)>::execute(
        GMM::Convolution::<(EI, ES, EA, EO)>::init_lhs_loader(lhs, x_offset, k_range.0, config),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_rhs_loader(rhs, k_range.0, y_offset, config),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_bias_loader(bias, y_offset, config),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_unloader(out, x_offset, y_offset),
        &mut GMM::Convolution::<(EI, ES, EA, EO)>::init_accumulator(config),
        k_range,
        config,
    );
}

pub mod config {
    use std::ops::Deref;

    use crate::{
        convolution::ConvGemmConfig,
        matmul::components::{MatmulConfig, TilingDimensions},
    };
    use global::GlobalConfig;

    use super::*;

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct HomogeneousConfig<M: GlobalConfig> {
        matmul: M,
        out_shape: (u32, u32),
        padded_channels: u32,
        kernel_size: (u32, u32),
        stride: (u32, u32),
        dilation: (u32, u32),
        padding: (i32, i32),
    }

    impl<M: GlobalConfig> Deref for HomogeneousConfig<M> {
        type Target = M;

        fn deref(&self) -> &Self::Target {
            &self.matmul
        }
    }

    impl<M: GlobalConfig> GlobalConfig for HomogeneousConfig<M> {
        type SmmConfig = M::SmmConfig;

        fn to_smm_config(&self) -> Self::SmmConfig {
            self.matmul.to_smm_config()
        }

        fn global_line_size(&self, ident: Ident) -> u32 {
            self.matmul.global_line_size(ident)
        }

        fn stage_line_size(&self, ident: Ident) -> u32 {
            self.matmul.stage_line_size(ident)
        }

        fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions {
            self.matmul.tiling_dimensions(ident)
        }

        fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
            self.matmul.matrix_layout(ident)
        }

        fn num_planes(&self) -> u32 {
            self.matmul.num_planes()
        }

        fn plane_dim(&self) -> u32 {
            self.matmul.plane_dim()
        }

        fn check_row_bounds(&self, ident: Ident) -> bool {
            self.matmul.check_row_bounds(ident)
        }

        fn check_col_bounds(&self, ident: Ident) -> bool {
            self.matmul.check_col_bounds(ident)
        }

        fn check_k_bounds(&self) -> bool {
            self.matmul.check_k_bounds()
        }
    }

    impl<M: GlobalConfig> ConvGemmConfig for HomogeneousConfig<M> {
        fn out_shape(&self, dim: u32) -> u32 {
            match dim {
                0 => self.out_shape.0,
                1 => self.out_shape.1,
                _ => unreachable!(),
            }
        }

        fn padded_channels(&self) -> u32 {
            self.padded_channels
        }

        fn kernel_size(&self, dim: u32) -> u32 {
            match dim {
                0 => self.kernel_size.0,
                1 => self.kernel_size.1,
                _ => unreachable!(),
            }
        }

        fn dilation(&self, dim: u32) -> u32 {
            match dim {
                0 => self.dilation.0,
                1 => self.dilation.1,
                _ => unreachable!(),
            }
        }

        fn stride(&self, dim: u32) -> u32 {
            match dim {
                0 => self.stride.0,
                1 => self.stride.1,
                _ => unreachable!(),
            }
        }

        fn padding(&self, dim: u32) -> i32 {
            match dim {
                0 => self.padding.0,
                1 => self.padding.1,
                _ => unreachable!(),
            }
        }
    }

    impl<M: GlobalConfig> MatmulConfig for HomogeneousConfig<M> {}

    impl<M: GlobalConfig> HomogeneousConfig<M> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            matmul: M,
            out_shape: (u32, u32),
            padded_channels: u32,
            kernel_size: (u32, u32),
            stride: (u32, u32),
            dilation: (u32, u32),
            padding: (i32, i32),
        ) -> Self {
            Self {
                matmul,
                out_shape,
                padded_channels,
                kernel_size,
                stride,
                dilation,
                padding,
            }
        }

        pub fn to_matmul_config(self) -> M {
            self.matmul
        }
    }
}
