use crate::matmul::components::global::unloader::Unloader;
use crate::matmul::components::global::{Config as _, GlobalMatmulFamily, Loader};
use crate::matmul::components::stage::single_buffer::{
    LhsBufferReader, LhsBufferReaderFamily, RhsBufferReader, RhsBufferReaderFamily,
};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::StageDim;
use crate::matmul::components::{config::MatmulConfig, global::ZeroAccumulatorLoader};
use crate::matmul::components::{global, MatmulProblem};
use crate::matmul::components::{stage, MatmulSpec};
use crate::matmul::components::{Ident, MatrixLayout};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::tensor::{ReadWrite, VirtualTensor};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::{LhsBufferLoader, RhsBufferLoader};

pub struct PipelinedMatmulFamily<SMM: stage::MatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for PipelinedMatmulFamily<SMM>
where
    SMM: stage::MatmulFamily<LhsReader = LhsBufferReaderFamily, RhsReader = RhsBufferReaderFamily>,
{
    type Matmul<MS: MatmulSpec> = PipelinedMatmul<MS, SMM::Matmul<MS::ES, MS::EG, MS::EA>>;
}

impl<SMM> MatmulConfigFactory for PipelinedMatmulFamily<SMM>
where
    SMM: stage::MatmulFamily,
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

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct PipelinedMatmul<MS: MatmulSpec, SMM: stage::Matmul<MS::ES, MS::EG, MS::EA>> {
    _ms: PhantomData<MS>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MS: MatmulSpec, SMM> global::GlobalMatmul<MS> for PipelinedMatmul<MS, SMM>
where
    SMM: stage::Matmul<
        MS::ES,
        MS::EG,
        MS::EA,
        LhsReader = LhsBufferReader<MS::ES>,
        RhsReader = RhsBufferReader<MS::ES>,
    >,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = LhsBufferLoader<MS::EG, MS::ES, SMM::Config>;
    type RhsLoader = RhsBufferLoader<MS::EG, MS::ES, SMM::Config>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MS::EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let num_buffers = 2;
        let buffer_step = config.stage_dim(Ident::Lhs).tile_size_y_dim();
        let k_step = num_buffers * buffer_step; // equal to SMM::K

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        ///////////////
        // Load A
        Self::LhsLoader::fill_stage(&mut lhs_loader, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, config);

        let lhs_buffer_reader_a = Self::LhsLoader::as_stage_reader(&lhs_loader);
        let rhs_buffer_reader_a = Self::RhsLoader::as_stage_reader(&rhs_loader);

        ///////////////
        // Get B
        Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
        Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

        let lhs_buffer_reader_b = Self::LhsLoader::as_stage_reader(&lhs_loader);
        let rhs_buffer_reader_b = Self::RhsLoader::as_stage_reader(&rhs_loader);

        for _ in 0..num_loops {
            sync_units();

            ///////////////
            // Load B & Advance
            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

            ///////////////
            // Execute A
            SMM::execute(
                &lhs_buffer_reader_a,
                &rhs_buffer_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                config.to_smm_config(),
            );

            sync_units();

            ///////////////
            // Load Next A
            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

            ///////////////
            // Execute B
            SMM::execute(
                &lhs_buffer_reader_b,
                &rhs_buffer_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
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
        lhs: VirtualTensor<MS::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MS::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(rhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_unloader(
        out: VirtualTensor<MS::EG, ReadWrite>,
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
