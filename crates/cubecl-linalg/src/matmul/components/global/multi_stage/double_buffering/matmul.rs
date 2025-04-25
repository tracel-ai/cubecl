use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::{
    BufferId, SyncBufferLoader, SyncBufferLoaderJob, SyncBufferLoadingStrategy,
};
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::{self, CommonGlobalConfig};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::BufferReader;
use crate::matmul::components::stage::StageEvent;
use crate::matmul::components::stage::StageEventListener;
use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem,
    stage,
};
use crate::matmul::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = BufferReaderFamily, RhsReader = BufferReaderFamily>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        DoubleBufferingMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
}

impl<SMM, LL, RL> MatmulConfigFactory for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
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
    SMM: stage::StageMatmul<MP>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
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
    SMM: stage::StageMatmul<
            MP,
            LhsReader = BufferReader<MP::ES, LL::TilingLayout>,
            RhsReader = BufferReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Config = CommonGlobalConfig<SMM::Config>;
    type LhsLoader = (
        SyncBufferLoader<MP, Self::Config, LL>,
        SyncBufferLoader<MP, Self::Config, LL>,
    );
    type RhsLoader = (
        SyncBufferLoader<MP, Self::Config, RL>,
        SyncBufferLoader<MP, Self::Config, RL>,
    );
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        lhs_loader: Self::LhsLoader,
        rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        // TODO
        // At this level, we consider that stage=buffer
        // Even though underneath, we will use 1 stage that contains 2 buffers
        // Should rethink the stage abstraction, it's more a SMEM manager
        // than a stage
        // Then we can merge names stage=buffer

        let buffer_step = config.tiling_dimensions(Ident::Lhs).total_col();
        let loop_step = buffer_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stages = div_ceil(range, buffer_step);

        // Algorithm assumes an even number of stages
        let num_stages = needed_stages + (needed_stages % 2);
        let num_loops = (num_stages - 2) / 2;

        SMM::zero_accumulator(acc, config.to_smm_config());
        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        let mut lhs_loader_a = lhs_loader.0;
        let mut lhs_loader_b = lhs_loader.1;
        let mut rhs_loader_a = rhs_loader.0;
        let mut rhs_loader_b = rhs_loader.1;

        let lhs_reader_a = SyncBufferLoader::<MP, Self::Config, LL>::reader(&lhs_loader_a);
        let lhs_reader_b = SyncBufferLoader::<MP, Self::Config, LL>::reader(&lhs_loader_b);
        let rhs_reader_a = SyncBufferLoader::<MP, Self::Config, RL>::reader(&rhs_loader_a);
        let rhs_reader_b = SyncBufferLoader::<MP, Self::Config, RL>::reader(&rhs_loader_b);

        SyncBufferLoader::<MP, Self::Config, LL>::fill_stage(&mut lhs_loader_a, config);
        SyncBufferLoader::<MP, Self::Config, RL>::fill_stage(&mut rhs_loader_a, config);

        SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_b, buffer_step);
        SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_b, buffer_step);

        sync_units();

        for _ in 0..num_loops {
            SMM::execute_with_listener::<
                DoubleBufferingEventListener<
                    SyncBufferLoader<MP, Self::Config, LL>,
                    SyncBufferLoader<MP, Self::Config, RL>,
                    Self::Config,
                >,
            >(
                &lhs_reader_a,
                &rhs_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                config.to_smm_config(),
                DoubleBufferingEventListener::new(&lhs_loader_b, &rhs_loader_b, config),
            );

            SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_a, loop_step);
            SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_a, loop_step);

            sync_units();

            SMM::execute_with_listener::<
                DoubleBufferingEventListener<
                    SyncBufferLoader<MP, Self::Config, LL>,
                    SyncBufferLoader<MP, Self::Config, RL>,
                    Self::Config,
                >,
            >(
                &lhs_reader_b,
                &rhs_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
                acc,
                config.to_smm_config(),
                DoubleBufferingEventListener::new(&lhs_loader_a, &rhs_loader_a, config),
            );

            SyncBufferLoader::<MP, Self::Config, LL>::advance_view(&mut lhs_loader_b, loop_step);
            SyncBufferLoader::<MP, Self::Config, RL>::advance_view(&mut rhs_loader_b, loop_step);

            sync_units();
        }

        SMM::execute_with_listener::<
            DoubleBufferingEventListener<
                SyncBufferLoader<MP, Self::Config, LL>,
                SyncBufferLoader<MP, Self::Config, RL>,
                Self::Config,
            >,
        >(
            &lhs_reader_a,
            &rhs_reader_a,
            &mut lhs_tile_a,
            &mut rhs_tile_a,
            acc,
            config.to_smm_config(),
            DoubleBufferingEventListener::new(&lhs_loader_b, &rhs_loader_b, config),
        );

        sync_units();

        SMM::execute(
            &lhs_reader_b,
            &rhs_reader_b,
            &mut lhs_tile_b,
            &mut rhs_tile_b,
            acc,
            config.to_smm_config(),
        );

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
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
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        (
            SyncBufferLoader::<MP, Self::Config, LL>::new(
                lhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::A,
                InputIdent::Lhs,
                config,
            ),
            SyncBufferLoader::<MP, Self::Config, LL>::new(
                lhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::B,
                InputIdent::Lhs,
                config,
            ),
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        (
            SyncBufferLoader::<MP, Self::Config, RL>::new(
                rhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::A,
                InputIdent::Rhs,
                config,
            ),
            SyncBufferLoader::<MP, Self::Config, RL>::new(
                rhs,
                x_offset,
                y_offset,
                batch_offset,
                quantization,
                BufferId::B,
                InputIdent::Rhs,
                config,
            ),
        )
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

#[cube]
pub trait LoaderEventListener: CubeType + Clone {
    type State: CubeType;
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> LoaderEventListener
    for SyncBufferLoader<MP, G, L>
{
    type State = SyncBufferLoaderJob<MP, L>;
}

#[derive(CubeType)]
struct DoubleBufferingEventListener<
    Lhs: LoaderEventListener,
    Rhs: LoaderEventListener,
    G: GlobalConfig,
> {
    loader_lhs: Lhs,
    loader_rhs: Rhs,
    #[cube(comptime)]
    config: G,
    state_lhs: Sequence<Lhs::State>,
    state_rhs: Sequence<Rhs::State>,
}

#[cube]
impl<Lhs: LoaderEventListener, Rhs: LoaderEventListener, G: GlobalConfig>
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
            state_lhs: Sequence::new(),
            state_rhs: Sequence::new(),
        }
    }
}

#[derive(Clone)]
/// How events are handled for double buffering.
///
/// The goal is to overlap computation instructions with memory instructions.
enum EventListenerMode {
    /// We execute memory instructions based on the given ratio (between 0 and 1) of the
    /// computation that is completed.
    Full {
        /// The ratio to be waited for with LHS.
        ratio_lhs: f32,
        /// The ratio to be waited for with RHS.
        ratio_rhs: f32,
    },
    /// We execute memory instructions for each [STEP] after [START] compute tasks are executed.
    Splitted {
        /// The event number to execute the next LHS task.
        event_lhs: u32,
        /// If no more tasks need to be executed for LHS.
        event_lhs_completed: bool,
        /// The event number to execute the next RHS task.
        event_rhs: u32,
        /// If no more tasks need to be executed for RHS.
        event_rhs_completed: bool,
    },
}

impl CubeDebug for EventListenerMode {}

const STEP: u32 = 2;
const START: u32 = 1;

fn should_handle_event(expected_event: u32, current_event: u32, total: u32) -> bool {
    current_event == expected_event || (total <= expected_event && current_event + 1 == total)
}

fn should_handle_event_ratio(ratio: f32, current_event: u32, total: u32) -> bool {
    should_handle_event(f32::ceil(ratio * total as f32) as u32, current_event, total)
}

#[cube]
impl<
    MP: MatmulPrecision,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    G: GlobalConfig,
> StageEventListener
    for DoubleBufferingEventListener<SyncBufferLoader<MP, G, LL>, SyncBufferLoader<MP, G, RL>, G>
{
    fn on_event(this: &mut Self, #[comptime] event: StageEvent) {
        if let StageEvent::TmmCompleted { current, total } = event {
            if comptime![current == 0] {
                this.init();
            }

            let mode = this.mode(total);

            match comptime!(mode) {
                EventListenerMode::Full {
                    ratio_lhs,
                    ratio_rhs,
                } => this.on_full_event(ratio_lhs, ratio_rhs, current, total),
                EventListenerMode::Splitted {
                    event_lhs,
                    event_lhs_completed,
                    event_rhs,
                    event_rhs_completed,
                } => this.on_splitted_event(
                    event_lhs,
                    event_lhs_completed,
                    event_rhs,
                    event_rhs_completed,
                    current,
                    total,
                ),
            }
        };
    }
}

#[cube]
impl<
    MP: MatmulPrecision,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    G: GlobalConfig,
> DoubleBufferingEventListener<SyncBufferLoader<MP, G, LL>, SyncBufferLoader<MP, G, RL>, G>
{
    fn init(&mut self) {
        let job_lhs = SyncBufferLoader::create_job(&self.loader_lhs, self.config);
        let job_rhs = SyncBufferLoader::create_job(&self.loader_rhs, self.config);

        self.state_lhs.push(job_lhs);
        self.state_rhs.push(job_rhs);
    }

    fn mode(&self, #[comptime] total: u32) -> comptime_type!(EventListenerMode) {
        let lhs_job = self.state_lhs.index(0);
        let rhs_job = self.state_rhs.index(0);
        let num_tasks_total = comptime!(lhs_job.num_tasks + rhs_job.num_tasks);

        if comptime!(num_tasks_total * STEP + (START) >= total) {
            comptime! {
                EventListenerMode::Full {
                    ratio_lhs: 0.1,
                    ratio_rhs: 0.3,
                }
            }
        } else {
            let lhs_num_task_executed = lhs_job.current.read().counter;
            let rhs_num_task_executed = rhs_job.current.read().counter;

            comptime! {
                EventListenerMode::Splitted {
                    event_lhs: lhs_num_task_executed  * STEP + START,
                    event_lhs_completed: lhs_num_task_executed >= lhs_job.num_tasks,
                    event_rhs: rhs_num_task_executed  * STEP + (lhs_job.num_tasks * STEP) + START,
                    event_rhs_completed: rhs_num_task_executed >= rhs_job.num_tasks,
                }
            }
        }
    }

    fn on_full_event(
        &mut self,
        #[comptime] ratio_lhs: f32,
        #[comptime] ratio_rhs: f32,
        #[comptime] current: u32,
        #[comptime] total: u32,
    ) {
        if comptime![should_handle_event_ratio(ratio_lhs, current, total)] {
            let lhs_job = self.state_lhs.index_mut(0);

            #[unroll]
            for _ in 0..lhs_job.num_tasks {
                SyncBufferLoader::execute_task(&mut self.loader_lhs, lhs_job, self.config);
            }
        }
        if comptime![should_handle_event_ratio(ratio_rhs, current, total)] {
            let rhs_job = self.state_rhs.index_mut(0);

            #[unroll]
            for _ in 0..rhs_job.num_tasks {
                SyncBufferLoader::execute_task(&mut self.loader_rhs, rhs_job, self.config);
            }
        }
    }

    fn on_splitted_event(
        &mut self,
        #[comptime] event_lhs: u32,
        #[comptime] event_lhs_completed: bool,
        #[comptime] event_rhs: u32,
        #[comptime] event_rhs_completed: bool,
        #[comptime] current: u32,
        #[comptime] total: u32,
    ) {
        if comptime![!event_lhs_completed && should_handle_event(event_lhs, current, total)] {
            let lhs_job = self.state_lhs.index_mut(0);

            SyncBufferLoader::execute_task(&mut self.loader_lhs, lhs_job, self.config);
        }

        if comptime![!event_rhs_completed && should_handle_event(event_rhs, current, total)] {
            let rhs_job = self.state_rhs.index_mut(0);

            SyncBufferLoader::execute_task(&mut self.loader_rhs, rhs_job, self.config);
        }
    }
}
