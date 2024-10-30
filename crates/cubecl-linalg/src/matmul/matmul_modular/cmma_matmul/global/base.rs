use crate::matmul::matmul_modular::cmma_matmul::stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matmul_modular::matmul_global::GmmConfig;
use crate::matmul::matmul_modular::matmul_global::{GlobalMatmul, Loader};
use crate::matmul::matmul_modular::matmul_stage::StageMatmul;
use crate::matmul::matmul_modular::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::{LhsTensorLoader, RhsTensorLoader, TensorUnloader};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct HomogeneousGlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        G::SmmConfig,
    >,
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
    > for HomogeneousGlobalMatmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        G::SmmConfig,
    >,
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

impl<EG, ES, SMM, G> Matmul<EG, EG> for HomogeneousGlobalMatmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        G::SmmConfig,
    >,
    G: GmmConfig,
{
    type Config = G;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }
}
