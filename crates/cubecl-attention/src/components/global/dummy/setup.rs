use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::stage::FullReaderFamily;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    global::{
        GlobalAttentionFamily,
        dummy::{DummyGlobalAttention, config::DummyGlobalConfig},
    },
    stage::{StageAttentionConfig as _, StageAttentionFamily},
};

pub struct DummyGlobalAttentionFamily<SA: StageAttentionFamily> {
    _phantom: PhantomData<SA>,
}

// impl<SMM, LL, RL> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL>
// where
//     SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
//     LL: SyncFullLoadingStrategy,
//     RL: SyncFullLoadingStrategy,
// {
//     type Matmul<MP: MatmulPrecision> =
//         SimpleMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
//     type Config = SimpleConfig<SMM::Config>;

impl<SA: StageAttentionFamily<KeyReader = FullReaderFamily, ValueReader = FullReaderFamily>>
    GlobalAttentionFamily for DummyGlobalAttentionFamily<SA>
{
    type Attention<AP: AttentionPrecision> = DummyGlobalAttention<AP, SA::Attention<AP>>;

    type Config = DummyGlobalConfig<SA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError> {
        let stage_config = SA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyGlobalConfig::new(stage_config, stage_config.num_planes())
    }
}
