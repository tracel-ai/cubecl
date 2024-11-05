use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::global;
use crate::matmul::components::global::Loader;
use crate::matmul::components::stage;
use crate::matmul::components::stage::TilingOrderConfig;
use crate::matmul::components::stage::{LhsReader, RhsReader};
use crate::matmul::components::MatmulKernel;
use crate::matmul::components::StageDim;
use crate::matmul::components::{Ident, MatrixLayout};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::tensor_view;

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    SMM: stage::Matmul<ES, EG, LhsReader<ES, G::SmmConfig>, RhsReader<ES, G::SmmConfig>, G::SmmConfig>,
    G: global::Config,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _stage_matmul: PhantomData<SMM>,
    _config: PhantomData<G>,
}

#[cube]
impl<EG, ES, SMM, G>
    global::Matmul<
        EG,
        ES,
        tensor_view::LhsLoader<EG, ES, G>,
        tensor_view::RhsLoader<EG, ES, G>,
        tensor_view::Unloader<EG, G>,
        G,
    > for Matmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: stage::Matmul<
        ES,
        EG,
        LhsReader<ES, G::SmmConfig>,
        RhsReader<ES, G::SmmConfig>,
        G::SmmConfig,
    >,
    G: global::Config,
{
    fn execute(
        mut lhs_loader: tensor_view::LhsLoader<EG, ES, G>,
        mut rhs_loader: tensor_view::RhsLoader<EG, ES, G>,
        mut out_unloader: tensor_view::Unloader<EG, G>,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = SMM::acc_init_zeros(config.to_smm_config());

        for _ in 0..num_loops {
            let lhs_stage_reader = &tensor_view::LhsLoader::fill_stage(&mut lhs_loader, config);
            let rhs_stage_reader = &tensor_view::RhsLoader::fill_stage(&mut rhs_loader, config);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut acc,
                config.to_smm_config(),
            );

            sync_units();

            tensor_view::LhsLoader::advance_view(&mut lhs_loader, k_step);
            tensor_view::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::acc_read::<tensor_view::Unloader<EG, G>, G>(
            &acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }
}

impl<EG, ES, SMM, G> MatmulKernel<EG, EG> for Matmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: stage::Matmul<
        ES,
        EG,
        LhsReader<ES, G::SmmConfig>,
        RhsReader<ES, G::SmmConfig>,
        G::SmmConfig,
    >,
    G: global::Config,
{
    type Config = G;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the HomogeneousGlobalMatmul
pub struct Config<S: stage::Config> {
    smm_config: S,
    out_smem_line_size: u32,
    check_m_bounds: bool,
    check_n_bounds: bool,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
}

impl<S: stage::Config> global::Config for Config<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
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

    fn out_smem_line_size(&self) -> u32 {
        self.out_smem_line_size
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
    pub fn new(
        smm_config: S,
        out_smem_line_size: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
    ) -> Self {
        Self {
            smm_config,
            out_smem_line_size,
            check_m_bounds,
            check_n_bounds,
            lhs_layout,
            rhs_layout,
        }
    }
}
