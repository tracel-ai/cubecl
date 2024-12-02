use crate::matmul::components::global::unloader::Unloader;
use crate::matmul::components::global::{Config as _, Loader};
use crate::matmul::components::stage;
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
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

use super::loader::{LhsBufferLoader, RhsBufferLoader};

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct Matmul<EG: Numeric, ES: Numeric, EA: Numeric, SMM: stage::Matmul<ES, EG, EA>> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _acc: PhantomData<EA>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<EG, ES, EA, SMM> global::Matmul<EG, ES> for Matmul<EG, ES, EA, SMM>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM:
        stage::Matmul<ES, EG, EA, LhsReader = LhsBufferReader<ES>, RhsReader = RhsBufferReader<ES>>,
{
    type LhsLoader = (LhsBufferLoader<EG, ES>, LhsBufferLoader<EG, ES>);
    type RhsLoader = (RhsBufferLoader<EG, ES>, RhsBufferLoader<EG, ES>);
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
        let range = k_range.1 - k_range.0;
        let num_stages = (range + SMM::K - 1) / SMM::K;

        SMM::zero_accumulator(acc, config.to_smm_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        // TODO
        //

        // Fetch 0
        let mut lhs_register_0 =
            LhsBufferLoader::fetch_global::<Self::Config>(&lhs_loader.0, config);
        let mut rhs_register_0 =
            RhsBufferLoader::fetch_global::<Self::Config>(&rhs_loader.0, config);

        for _ in 0..num_stages {
            // Fetch 1
            let mut lhs_register_1 =
                LhsBufferLoader::fetch_global::<Self::Config>(&lhs_loader.1, config);
            let mut rhs_register_1 =
                RhsBufferLoader::fetch_global::<Self::Config>(&rhs_loader.1, config);

            // Fill, compute and advance 0
            let lhs_reader_0 = LhsBufferLoader::fill_stage::<Self::Config>(
                &mut lhs_loader.0,
                &mut lhs_register_0,
                config,
            );
            let rhs_reader_0 = RhsBufferLoader::fill_stage::<Self::Config>(
                &mut rhs_loader.0,
                &mut rhs_register_0,
                config,
            );
            SMM::execute(
                &lhs_reader_0,
                &rhs_reader_0,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );
            LhsBufferLoader::to_next_stage::<Self::Config>(&mut lhs_loader.0, config);
            RhsBufferLoader::to_next_stage::<Self::Config>(&mut rhs_loader.0, config);

            sync_units();

            // Fetch 0
            // Last time is actually useless. Should we branch or do it anyway?
            lhs_register_0 = LhsBufferLoader::fetch_global::<Self::Config>(&lhs_loader.0, config);
            rhs_register_0 = RhsBufferLoader::fetch_global::<Self::Config>(&rhs_loader.0, config);

            // Fill, compute and advance 1
            let lhs_reader_1 = LhsBufferLoader::fill_stage::<Self::Config>(
                &mut lhs_loader.1,
                &mut lhs_register_1,
                config,
            );
            let rhs_reader_1 = RhsBufferLoader::fill_stage::<Self::Config>(
                &mut rhs_loader.1,
                &mut rhs_register_1,
                config,
            );
            SMM::execute(
                &lhs_reader_1,
                &rhs_reader_1,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );
            LhsBufferLoader::to_next_stage::<Self::Config>(&mut lhs_loader.1, config);
            RhsBufferLoader::to_next_stage::<Self::Config>(&mut rhs_loader.1, config);

            sync_units();
        }

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
        (
            LhsBufferLoader::new(lhs, x_offset, y_offset, batch_offset, 0, config),
            LhsBufferLoader::new(lhs, x_offset, y_offset, batch_offset, 1, config),
        )
    }

    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        (
            RhsBufferLoader::new(rhs, x_offset, y_offset, batch_offset, 0, config),
            RhsBufferLoader::new(rhs, x_offset, y_offset, batch_offset, 1, config),
        )
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

impl<EG, ES, EA, SMM> MatmulKernel<EG, EG> for Matmul<EG, ES, EA, SMM>
where
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    SMM: stage::Matmul<ES, EG, EA>,
{
    type Config = Config<SMM::Config>;

    fn check_config(config: Self::Config) {
        assert!(
            config.stage_dim(Ident::Lhs).num_tiles_y_dim() == 2,
            "Pipelined matmul needs exactly 2 buffers."
        );
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
            cube_dim.y,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the pipelined global matmul
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
    num_planes: u32,
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
        self.num_planes
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
        num_planes: u32,
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
            num_planes,
        }
    }
}
