use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;
use cubecl_matmul::components::{global::PartitionedStageFamily, stage::StridedStageFamily};

use crate::components::{
    AttentionElems, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError,
    global::{
        GlobalAttentionFamily,
        simple::{SimpleGlobalAttention, config::SimpleGlobalConfig},
    },
    stage::{StageAttentionConfig as _, StageAttentionFamily},
};

pub struct SimpleGlobalAttentionFamily<SA: StageAttentionFamily> {
    _phantom: PhantomData<SA>,
}

impl<
    SA: StageAttentionFamily<
            KeyStage = StridedStageFamily,
            ValueStage = StridedStageFamily,
            OutStage = PartitionedStageFamily,
        >,
> GlobalAttentionFamily for SimpleGlobalAttentionFamily<SA>
{
    type Attention<AP: AttentionPrecision> = SimpleGlobalAttention<AP, SA::Attention<AP>>;

    type Config = SimpleGlobalConfig<SA::Config>;

    fn setup<R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let stage_config = SA::setup::<R>(client, problem, selection, line_sizes, dtypes)?;

        SimpleGlobalConfig::new(stage_config, stage_config.num_planes(), problem.causal)
    }
}
