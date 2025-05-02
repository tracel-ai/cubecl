use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::{
    BufferId, SyncBufferLoader, SyncBufferLoaderJob, SyncBufferLoadingStrategy, SyncFullLoader,
    SyncFullLoaderJob, SyncFullLoadingStrategy, sync_full_tilewise,
};
use crate::matmul::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::{self, LoadingValidation};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::{BufferReader, ColMajorTilingOrder};
use crate::matmul::components::stage::{FullReader, StageEvent};
use crate::matmul::components::stage::{FullReaderFamily, StageEventListener};
use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem,
    stage,
};
use crate::matmul::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::kernels::matmul::LoadingPrecomputeStrategy;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
}

/// The ordered double buffering global matmul
/// needs tilewise loading on Lhs to guarantee that planes
/// only use data that they have loaded themselves.
/// Also, it must be with col major tiling order so that
/// the first loading tasks that are executed are for the first k iteration and so on
pub type LL = sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>;

impl<SMM, RL> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = BufferReaderFamily>,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP, <LL as SyncFullLoadingStrategy>::TilingLayout, RL::TilingLayout>,
        RL,
    >;
}

impl<SMM, RL> MatmulConfigFactory for OrderedDoubleBufferingMatmulFamily<SMM, RL>
where
    SMM: stage::StageMatmulFamily,
    RL: SyncBufferLoadingStrategy,
{
    type Input = (SMM::Input, LoadingPrecomputeStrategy);
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        <LL as LoadingValidation>::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;

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
        let smm_config = SMM::make_config(input.0, problem, cube_dim, cube_count, quantized);
        let stage_shape = SMM::stage_shape(&smm_config);

        DoubleBufferingGlobalConfig::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % (2 * stage_shape.k) != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
            input.1,
        )
    }
}

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    RL: SyncBufferLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, RL> global::GlobalMatmul<MP>
    for OrderedDoubleBufferingMatmul<MP, SMM, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = FullReader<MP::ES, <LL as SyncFullLoadingStrategy>::TilingLayout>,
            RhsReader = BufferReader<MP::ES, RL::TilingLayout>,
        >,
    RL: SyncBufferLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = SyncFullLoader<MP, Self::Config, LL>;
    type RhsLoader = SyncBufferLoader<MP, Self::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EO>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let buffer_step = config.tiling_dimensions(Ident::Lhs).total_col();
        let loop_step = buffer_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = div_ceil(range, buffer_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        let lhs_reader = Self::LhsLoader::reader(&lhs_loader);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        Self::LhsLoader::fill_stage(&mut lhs_loader, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, BufferId::A, config);

        Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);

        sync_units();

        for _ in 0..num_loops {
            SMM::execute_with_listener::<
                DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
            >(
                &lhs_reader,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
                DoubleBufferingEventListener::new(BufferId::B, &lhs_loader, &rhs_loader, config),
            );

            // Rhs is advanced by 2 * k because Buffer B shares the same global memory state as Buffer A,
            // but it is implicitly offset by one buffer's worth (k elements) when reading.
            // Lhs, on the other hand, is advanced by k twice — once per load/execute cycle —
            // because it does not use a second, offset buffer; it simply progresses linearly through global memory.
            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, loop_step);

            sync_units();

            SMM::execute_with_listener::<
                DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
            >(
                &lhs_reader,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
                DoubleBufferingEventListener::new(BufferId::A, &lhs_loader, &rhs_loader, config),
            );

            sync_units();

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
        }

        SMM::execute_with_listener::<
            DoubleBufferingEventListener<Self::LhsLoader, Self::RhsLoader, Self::Config>,
        >(
            &lhs_reader,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            config.to_smm_config(),
            DoubleBufferingEventListener::new(BufferId::B, &lhs_loader, &rhs_loader, config),
        );

        // Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
        sync_units();

        SMM::execute(
            &lhs_reader,
            &rhs_reader_b,
            &mut lhs_tile,
            &mut rhs_tile,
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
        SyncFullLoader::<MP, Self::Config, LL>::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            InputIdent::Lhs,
            config,
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
        SyncBufferLoader::<MP, Self::Config, RL>::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            InputIdent::Rhs,
            config,
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
pub trait LhsLoaderEventListener: CubeType + Clone {
    type State: CubeType;
}

#[cube]
pub trait RhsLoaderEventListener: CubeType + Clone {
    type State: CubeType;
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> LhsLoaderEventListener
    for SyncFullLoader<MP, G, L>
{
    type State = SyncFullLoaderJob<MP, L>;
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> RhsLoaderEventListener
    for SyncBufferLoader<MP, G, L>
{
    type State = SyncBufferLoaderJob<MP, L>;
}

#[derive(CubeType)]
struct DoubleBufferingEventListener<
    Lhs: LhsLoaderEventListener,
    Rhs: RhsLoaderEventListener,
    G: GlobalConfig,
> {
    #[cube(comptime)]
    buffer_id: BufferId,
    loader_lhs: Lhs,
    loader_rhs: Rhs,
    #[cube(comptime)]
    config: G,
    state_lhs: Sequence<Lhs::State>,
    state_rhs: Sequence<Rhs::State>,
}

#[cube]
impl<Lhs: LhsLoaderEventListener, Rhs: RhsLoaderEventListener, G: GlobalConfig>
    DoubleBufferingEventListener<Lhs, Rhs, G>
{
    pub fn new(
        #[comptime] buffer_id: BufferId,
        loader_lhs: &Lhs,
        loader_rhs: &Rhs,
        #[comptime] config: G,
    ) -> DoubleBufferingEventListener<Lhs, Rhs, G> {
        DoubleBufferingEventListener::<Lhs, Rhs, G> {
            buffer_id,
            loader_lhs: comptime![loader_lhs.clone()],
            loader_rhs: comptime![loader_rhs.clone()],
            config,
            state_lhs: Sequence::new(),
            state_rhs: Sequence::new(),
        }
    }
}

#[derive(Clone)]
/// Analysis of [StageEvent] that reports when lhs and rhs should execute a task.
struct EventAnalysis {
    /// The event count to execute the next lhs task.
    lhs: u32,
    /// If no more tasks need to be executed for lhs.
    lhs_completed: bool,
    /// The event count to execute the next rhs task.
    rhs: u32,
    /// If no more tasks need to be executed for rhs.
    rhs_completed: bool,
}

impl CubeDebug for EventAnalysis {}

#[cube]
impl<
    MP: MatmulPrecision,
    LL: SyncFullLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    G: GlobalConfig,
> StageEventListener
    for DoubleBufferingEventListener<SyncFullLoader<MP, G, LL>, SyncBufferLoader<MP, G, RL>, G>
{
    fn on_event(this: &mut Self, #[comptime] event: StageEvent) {
        if let StageEvent::Begin = event {
            this.init();
        }

        // if let StageEvent::TmmCompleted { current, total } = event {
        //     let analysis = this.analyse(total);

        //     if comptime![!analysis.lhs_completed && analysis.lhs == current] {
        //         let lhs_job = this.state_lhs.index_mut(0);

        //         SyncFullLoader::execute_task(&mut this.loader_lhs, lhs_job, this.config);
        //     }

        //     if comptime![!analysis.rhs_completed && analysis.rhs == current] {
        //         let rhs_job = this.state_rhs.index_mut(0);

        //         SyncBufferLoader::execute_task(&mut this.loader_rhs, rhs_job, this.config);
        //     }
        // }

        // Cleanup remaining tasks if any.
        if let StageEvent::Finish = event {
            let lhs_job = this.state_lhs.index_mut(0);
            let lhs_num_task_executed = lhs_job.current.read().counter;

            #[unroll]
            for _ in lhs_num_task_executed..lhs_job.num_tasks {
                SyncFullLoader::execute_task(&mut this.loader_lhs, lhs_job, this.config);
            }

            let rhs_job = this.state_rhs.index_mut(0);
            let rhs_num_task_executed = rhs_job.current.read().counter;

            #[unroll]
            for _ in rhs_num_task_executed..rhs_job.num_tasks {
                SyncBufferLoader::execute_task(&mut this.loader_rhs, rhs_job, this.config);
            }
        }
    }
}

#[cube]
impl<
    MP: MatmulPrecision,
    LL: SyncFullLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
    G: GlobalConfig,
> DoubleBufferingEventListener<SyncFullLoader<MP, G, LL>, SyncBufferLoader<MP, G, RL>, G>
{
    fn init(&mut self) {
        let job_lhs = SyncFullLoader::create_job(&self.loader_lhs, self.config);
        let job_rhs = SyncBufferLoader::create_job(&self.loader_rhs, self.buffer_id, self.config);

        self.state_lhs.push(job_lhs);
        self.state_rhs.push(job_rhs);
    }

    fn analyse(&self, #[comptime] event_count_total: u32) -> comptime_type!(EventAnalysis) {
        let lhs_job = self.state_lhs.index(0);
        let rhs_job = self.state_rhs.index(0);
        let num_tasks_total = comptime!(lhs_job.num_tasks + rhs_job.num_tasks);

        let lhs_num_task_executed = lhs_job.current.read().counter;
        let rhs_num_task_executed = rhs_job.current.read().counter;

        comptime! {
            let step = 1u32;
            let start = event_count_total - (step * num_tasks_total);

            EventAnalysis {
                lhs: lhs_num_task_executed * step + start,
                lhs_completed: lhs_num_task_executed >= lhs_job.num_tasks,
                rhs: rhs_num_task_executed * step + (lhs_job.num_tasks * step) + start,
                rhs_completed: rhs_num_task_executed >= rhs_job.num_tasks,
            }
        }
    }
}
