use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::global::GlobalMatmul;
use crate::matmul::components::global::GmmConfig;
use crate::matmul::components::global::Loader;
use crate::matmul::components::matrix::{Ident, MatrixLayout};
use crate::matmul::components::stage::{LhsReader, RhsReader, StageMatmul};
use crate::matmul::components::stage::{SmmConfig, TilingOrderConfig};
use crate::matmul::components::stage_dim::StageDim;
use crate::matmul::components::MatmulKernel;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::tensor_loader::LhsTensorLoader;
use super::tensor_loader::RhsTensorLoader;
use super::tensor_unloader::TensorUnloader;

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct Matmul<
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<ES, EG, LhsReader<ES, G::SmmConfig>, RhsReader<ES, G::SmmConfig>, G::SmmConfig>,
    G: GmmConfig,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _stage_matmul: PhantomData<SMM>,
    _config: PhantomData<G>,
}

#[cube]
impl<EG, ES, SMM, G>
    GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, G>,
        RhsTensorLoader<EG, ES, G>,
        TensorUnloader<EG, G>,
        G,
    > for Matmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM:
        StageMatmul<ES, EG, LhsReader<ES, G::SmmConfig>, RhsReader<ES, G::SmmConfig>, G::SmmConfig>,
    G: GmmConfig,
{
    fn execute(
        mut lhs_loader: LhsTensorLoader<EG, ES, G>,
        mut rhs_loader: RhsTensorLoader<EG, ES, G>,
        mut out_unloader: TensorUnloader<EG, G>,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = SMM::acc_init_zeros(config.to_smm_config());

        for _ in 0..num_loops {
            let lhs_stage_reader = &LhsTensorLoader::fill_stage(&mut lhs_loader, config);
            let rhs_stage_reader = &RhsTensorLoader::fill_stage(&mut rhs_loader, config);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut acc,
                config.to_smm_config(),
            );

            sync_units();

            LhsTensorLoader::advance_view(&mut lhs_loader, k_step);
            RhsTensorLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::acc_read::<TensorUnloader<EG, G>, G>(
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
    SMM:
        StageMatmul<ES, EG, LhsReader<ES, G::SmmConfig>, RhsReader<ES, G::SmmConfig>, G::SmmConfig>,
    G: GmmConfig,
{
    type Config = G;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the HomogeneousGlobalMatmul
pub struct Config<S: SmmConfig> {
    smm_config: S,
    out_smem_line_size: u32,
    check_m_bounds: bool,
    check_n_bounds: bool,
}

impl<S: SmmConfig> GmmConfig for Config<S> {
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
        self.smm_config.layout(ident)
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
}

impl<S: SmmConfig> MatmulConfig for Config<S> {}

impl<S: SmmConfig> Config<S> {
    pub fn new(
        smm_config: S,
        out_smem_line_size: u32,
        check_m_bounds: bool,
        check_n_bounds: bool,
    ) -> Self {
        Self {
            smm_config,
            out_smem_line_size,
            check_m_bounds,
            check_n_bounds,
        }
    }
}
