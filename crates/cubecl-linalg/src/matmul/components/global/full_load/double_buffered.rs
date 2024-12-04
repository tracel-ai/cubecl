use crate::matmul::components::global::unloader::Unloader;
use crate::matmul::components::global::{Config as _, Loader, LoadingStrategy};
use crate::matmul::components::stage;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{config::MatmulConfig, global::ZeroAccumulatorLoader};
use crate::matmul::components::{global, MatmulProblem};
use crate::matmul::components::{Ident, MatrixLayout};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::{LhsLoader, RhsLoader};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _acc: PhantomData<EA>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<EG, ES, EA, SMM, LL, RL> global::Matmul<EG, ES> for Matmul<EG, ES, EA, SMM, LL, RL>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA, LhsReader = LhsReader<ES>, RhsReader = RhsReader<ES>>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
{
    type LhsLoader = LhsLoader<EG, ES, LL>;
    type RhsLoader = RhsLoader<EG, ES, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());

        let mut lhs_buffer = Self::LhsLoader::init_buffer::<Self::Config>(config);
        let mut rhs_buffer = Self::RhsLoader::init_buffer::<Self::Config>(config);

        let mut lhs_curr = LoadBuffer::current_half(&mut lhs_buffer, 0);
        let mut rhs_curr = LoadBuffer::current_half(&mut rhs_buffer, 0);
        let mut lhs_next = LoadBuffer::next_half(&mut lhs_buffer, 0);
        let mut rhs_next = LoadBuffer::next_half(&mut rhs_buffer, 0);

        // Fetch current
        Self::LhsLoader::fetch_global::<Self::Config>(
            &mut lhs_loader,
            &mut LoadBuffer::as_slice_mut(&mut lhs_buffer, lhs_curr),
            config,
        );
        Self::RhsLoader::fetch_global::<Self::Config>(
            &mut rhs_loader,
            &mut LoadBuffer::as_slice_mut(&mut rhs_buffer, rhs_curr),
            config,
        );

        for i in 0..num_stages {
            sync_units();

            // Advance tensor views
            Self::LhsLoader::to_next_stage::<Self::Config>(&mut lhs_loader, config);
            Self::RhsLoader::to_next_stage::<Self::Config>(&mut rhs_loader, config);

            // Fetch next
            Self::LhsLoader::fetch_global::<Self::Config>(
                &mut lhs_loader,
                &mut LoadBuffer::as_slice_mut(&mut lhs_buffer, lhs_next),
                config,
            );
            Self::RhsLoader::fetch_global::<Self::Config>(
                &mut rhs_loader,
                &mut LoadBuffer::as_slice_mut(&mut rhs_buffer, rhs_next),
                config,
            );

            // Fill stage with current
            let lhs_stage_reader = &LhsLoader::fill_stage::<Self::Config>(
                &mut lhs_loader,
                &mut LoadBuffer::as_slice_mut(&mut lhs_buffer, lhs_curr),
                config,
            );
            let rhs_stage_reader = &RhsLoader::fill_stage::<Self::Config>(
                &mut rhs_loader,
                &mut LoadBuffer::as_slice_mut(&mut rhs_buffer, rhs_curr),
                config,
            );

            sync_units();

            // Switch buffers for next iteration
            lhs_curr = LoadBuffer::current_half(&mut lhs_buffer, i + 1);
            rhs_curr = LoadBuffer::current_half(&mut rhs_buffer, i + 1);
            lhs_next = LoadBuffer::next_half(&mut lhs_buffer, i + 1);
            rhs_next = LoadBuffer::next_half(&mut rhs_buffer, i + 1);

            // Execute with current stage
            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );
        }

        sync_units();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        LhsLoader::new::<Self::Config>(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        RhsLoader::new::<Self::Config>(rhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_unloader(
        out: &mut Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
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

impl<EG, ES, EA, SMM, LL, RL> MatmulKernel<EG, EG> for Matmul<EG, ES, EA, SMM, LL, RL>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA>,
    LL: LoadingStrategy<EG, ES>,
    RL: LoadingStrategy<EG, ES>,
{
    type Config = Config<SMM::Config>;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let smm_config = SMM::make_config(problem, cube_dim, cube_count, advanced_config);

        Config::new(
            smm_config,
            problem.m as u32 % SMM::M != 0,
            problem.n as u32 % SMM::N != 0,
            problem.k as u32 % SMM::K != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the full load matmul
pub struct Config<S: stage::Config> {
    smm_config: S,
    check_m_bounds: bool,
    check_n_bounds: bool,
    check_k_bounds: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl<S: stage::Config> global::Config for Config<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn global_line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }

    fn stage_line_size(&self, ident: Ident) -> u32 {
        self.smm_config.line_size(ident)
    }

    fn stage_dim(&self, ident: Ident) -> Box<dyn StageDim> {
        self.smm_config.stage_dim(ident)
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => self.smm_config.layout(Ident::Out),
        }
    }

    fn num_planes(&self) -> u32 {
        self.smm_config.num_planes()
    }

    fn plane_dim(&self) -> u32 {
        self.smm_config.plane_dim()
    }

    fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
        self.smm_config.tiling_order(ident)
    }

    fn check_m_bounds(&self) -> bool {
        self.check_m_bounds
    }

    fn check_n_bounds(&self) -> bool {
        self.check_n_bounds
    }

    fn check_k_bounds(&self) -> bool {
        self.check_k_bounds
    }

    fn transpose_load(&self, ident: Ident) -> bool {
        self.layout(ident) != self.smm_config.layout(ident)
    }

    fn num_buffers(&self) -> u32 {
        2
    }
}

impl<S: stage::Config> MatmulConfig for Config<S> {}

impl<S: stage::Config> Config<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        smm_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            smm_config,
            check_m_bounds,
            check_n_bounds,
            check_k_bounds,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
