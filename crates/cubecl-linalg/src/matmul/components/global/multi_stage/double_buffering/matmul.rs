use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::single_stage::{Loader, SyncLoader, SyncLoadingStrategy};
use crate::matmul::components::global::{self, CommonGlobalConfig};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::global::{GlobalMatmulFamily, IndexedQuantization};
use crate::matmul::components::stage::StageEvent;
use crate::matmul::components::stage::StageEventListener;
use crate::matmul::components::{
    Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem, stage,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncLoadingStrategy,
    RL: SyncLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncLoadingStrategy,
    RL: SyncLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        DoubleBufferingMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
}

impl<SMM, LL, RL> MatmulConfigFactory for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncLoadingStrategy,
    RL: SyncLoadingStrategy,
{
    type Input = SMM::Input;
    type Config = CommonGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;

        if config.tiling_dimensions(Ident::Lhs).tile_count_col() != 2 {
            return Err(Box::new("Double buffering matmul needs exactly 2 buffers."));
        }

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.smm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let smm_config = SMM::make_config(input, problem, cube_dim, cube_count, quantized);
        let stage_shape = SMM::stage_shape(&smm_config);

        CommonGlobalConfig::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % stage_shape.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
        )
    }
}

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP, LL::TilingLayout, RL::TilingLayout>,
    LL: SyncLoadingStrategy,
    RL: SyncLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> global::GlobalMatmul<MP>
    for DoubleBufferingMatmul<MP, SMM, LL, RL>
where
    SMM: stage::StageMatmul<MP, LL::TilingLayout, RL::TilingLayout>,
    LL: SyncLoadingStrategy,
    RL: SyncLoadingStrategy,
{
    type Config = CommonGlobalConfig<SMM::Config>;
    type LhsLoader = SyncLoader<MP, Self::Config, LL>;
    type RhsLoader = SyncLoader<MP, Self::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<IndexedQuantization<MP::EI, MP::EO>>,
        #[comptime] config: Self::Config,
    ) {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let num_buffers = 2;
        let buffer_step = config.tiling_dimensions(Ident::Lhs).tile_shape_col();
        let k_step = num_buffers * buffer_step;

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        // Buffer A
        let lhs_reader_a = Self::LhsLoader::reader(&lhs_loader);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader);
        // Buffer B
        let lhs_reader_b = Self::LhsLoader::reader(&lhs_loader);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader);

        // Buffer A
        Self::LhsLoader::fill_stage(&mut lhs_loader, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, config);

        sync_units();

        for _ in 1..num_loops {
            SMM::execute_with_listener::<
                DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
            >(
                &lhs_reader_a,
                &rhs_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                CubeOption::new_None(),
                config.to_smm_config(),
                DoubleBufferingEventListener::new(&lhs_loader, &rhs_loader, config),
            );

            sync_units();

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            SMM::execute_with_listener::<
                DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
            >(
                &lhs_reader_b,
                &rhs_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
                acc,
                CubeOption::new_None(),
                config.to_smm_config(),
                DoubleBufferingEventListener::new(&lhs_loader, &rhs_loader, config),
            );
            sync_units();
        }

        SMM::execute_with_listener::<
            DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
        >(
            &lhs_reader_a,
            &rhs_reader_a,
            &mut lhs_tile_a,
            &mut rhs_tile_a,
            acc,
            CubeOption::new_None(),
            config.to_smm_config(),
            DoubleBufferingEventListener::new(&lhs_loader, &rhs_loader, config),
        );

        sync_units();

        SMM::execute(
            &lhs_reader_b,
            &rhs_reader_b,
            &mut lhs_tile_b,
            &mut rhs_tile_b,
            acc,
            CubeOption::new_None(),
            config.to_smm_config(),
        );

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
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(lhs, x_offset, y_offset, batch_offset, Ident::Lhs, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(rhs, x_offset, y_offset, batch_offset, Ident::Rhs, config)
    }

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        SMM::zero_accumulator(acc, config.to_smm_config());
    }
}

#[derive(CubeType)]
struct DoubleBufferingEventListener<Lhs: CubeType, Rhs: CubeType, G: GlobalConfig> {
    loader_lhs: Lhs,
    loader_rhs: Rhs,
    #[cube(comptime)]
    config: G,
}

#[cube]
impl<Lhs: CubeType + Clone, Rhs: CubeType + Clone, G: GlobalConfig>
    DoubleBufferingEventListener<Lhs, Rhs, G>
{
    pub fn new(
        loader_lhs: &Lhs,
        loader_rhs: &Rhs,
        #[comptime] config: G,
    ) -> DoubleBufferingEventListener<Lhs, Rhs, G> {
        DoubleBufferingEventListener::<Lhs, Rhs, G> {
            loader_lhs: comptime![loader_lhs.clone()],
            loader_rhs: comptime![loader_rhs.clone()],
            config,
        }
    }
}

fn should_handle_event(expected_event: u32, current_event: u32, total: u32) -> bool {
    current_event == expected_event || (total < expected_event && current_event + 1 == total)
}

fn should_handle_event_ratio(ratio: f32, current_event: u32, total: u32) -> bool {
    should_handle_event(f32::ceil(ratio * total as f32) as u32, current_event, total)
}

#[cube]
impl<MP: MatmulPrecision, LL: SyncLoadingStrategy, RL: SyncLoadingStrategy, G: GlobalConfig>
    StageEventListener
    for DoubleBufferingEventListener<SyncLoader<MP, G, LL>, SyncLoader<MP, G, RL>, G>
{
    fn on_event(this: &mut Self, #[comptime] event: StageEvent) {
        if let StageEvent::TmmCompleted { current, total } = event {
            if comptime![should_handle_event_ratio(0.25, current, total)] {
                SyncLoader::fill_stage(&mut this.loader_lhs, this.config);
            }

            if comptime![should_handle_event_ratio(0.5, current, total)] {
                SyncLoader::fill_stage(&mut this.loader_rhs, this.config);
            }
        };
    }
}
