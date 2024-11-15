use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::global::unloader::Unloader;
use crate::matmul::components::global::{Config as _, Loader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{global, MatmulProblem};
use crate::matmul::components::{stage, PlaneMapper};
use crate::matmul::components::{Ident, MatrixLayout};
use crate::matmul::kernels::matmul::AdvancedConfig;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::{LhsBufferLoader, RhsBufferLoader};

/// Performs matrix multiplication at the global level, with planes split between two roles:
/// - Some planes load data to the stage
/// - Some planes are used in the stage matmul computation
/// Both roles alternate the buffer (tile index in dimension k) they are working on
pub struct Matmul<EG: Numeric, ES: Numeric, SMM: stage::Matmul<ES, EG>> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, SMM: stage::Matmul<ES, EG>> PlaneMapper for Matmul<EG, ES, SMM> {
    fn plane_id() -> u32 {
        // The change is rather in the smm where it must consider there are less planes
        // Also it's in the config that the GMM can influence the SMM
        // By changing num_planes, mostly, but also the plane mapper should have config as parameter
        // otherwise cannot know the offset

        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<EG, ES, SMM> global::Matmul<EG, ES> for Matmul<EG, ES, SMM>
where
    EG: Numeric,
    ES: Numeric,
    SMM: stage::Matmul<ES, EG, Lhs = LhsBufferReader<ES>, Rhs = RhsBufferReader<ES>>,
{
    type Lhs = LhsBufferLoader<EG, ES>;
    type Rhs = RhsBufferLoader<EG, ES>;
    type Out = Unloader<EG>;

    fn execute(
        mut lhs_loader: Self::Lhs,
        mut rhs_loader: Self::Rhs,
        mut out_unloader: Self::Out,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let is_consumer = Self::is_consumer(config);

        let num_buffers = config.stage_dim(Ident::Lhs).num_tiles_y;
        let buffer_step = config.stage_dim(Ident::Lhs).tile_size_y;
        let k_step = num_buffers * buffer_step; // equal to SMM::K

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages * num_buffers;

        let mut acc = SMM::acc_init_zeros(config.to_smm_config());

        for _ in 0..num_loops {
            let lhs_stage_reader = Self::Lhs::fill_stage::<Self::Config>(&mut lhs_loader, config);
            let rhs_stage_reader = Self::Rhs::fill_stage::<Self::Config>(&mut rhs_loader, config);

            sync_units();

            if is_consumer {
                SMM::execute(
                    &lhs_stage_reader,
                    &rhs_stage_reader,
                    &mut acc,
                    config.to_smm_config(),
                );
            }

            Self::Lhs::advance_view(&mut lhs_loader, buffer_step);
            Self::Rhs::advance_view(&mut rhs_loader, buffer_step);
        }

        if is_consumer {
            SMM::acc_read::<Self::Out, Self::Config>(
                &acc,
                &mut out_unloader,
                config.to_smm_config(),
                config,
            );
        }
    }

    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Lhs {
        Self::Lhs::new::<Self::Config>(
            lhs,
            x_offset,
            y_offset,
            nth_batch,
            !Self::is_consumer(config),
            config,
        )
    }

    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Rhs {
        Self::Rhs::new::<Self::Config>(
            rhs,
            x_offset,
            y_offset,
            nth_batch,
            !Self::is_consumer(config),
            config,
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
}

#[cube]
impl<EG: Numeric, ES: Numeric, SMM: stage::Matmul<ES, EG>> Matmul<EG, ES, SMM> {
    fn is_consumer(#[comptime] config: <Self as MatmulKernel<EG, EG>>::Config) -> bool {
        // This is not stored but recomputed each time
        // Should be configurable
        Self::plane_id() * 2 / config.plane_dim() == 1
    }
}

impl<EG, ES, SMM> MatmulKernel<EG, EG> for Matmul<EG, ES, SMM>
where
    EG: Numeric,
    ES: Numeric,
    SMM: stage::Matmul<ES, EG>,
{
    type Config = Config<SMM::Config>;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
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
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the HomogeneousGlobalMatmul
pub struct Config<S: stage::Config> {
    smm_config: S,
    check_m_bounds: bool,
    check_n_bounds: bool,
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

    fn stage_dim(&self, ident: Ident) -> StageDim {
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

    fn tiling_order(&self) -> TilingOrderConfig {
        self.smm_config.tiling_order()
    }

    fn check_m_bounds(&self) -> bool {
        self.check_m_bounds
    }

    fn check_n_bounds(&self) -> bool {
        self.check_n_bounds
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
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
