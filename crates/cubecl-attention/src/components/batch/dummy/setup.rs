use crate::components::{
    AttentionPrecision,
    batch::{
        BatchAttentionFamily,
        dummy::{DummyBatchAttention, config::DummyBatchConfig},
    },
};

pub struct DummyBatchAttentionFamily {}
impl BatchAttentionFamily for DummyBatchAttentionFamily {
    type Attention<AP: AttentionPrecision> = DummyBatchAttention;

    type Config = DummyBatchConfig;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R::Server, R::Channel>,
        problem: &crate::components::AttentionProblem,
        selection: &crate::components::AttentionSelection,
        line_sizes: &crate::components::AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        todo!()
    }

    unsafe fn launch_unchecked<
        'a,
        MS: crate::components::AttentionSpec,
        R: cubecl_core::Runtime,
    >(
        client: &cubecl_core::prelude::ComputeClient<
            <R as cubecl_core::Runtime>::Server,
            <R as cubecl_core::Runtime>::Channel,
        >,
        cube_dim: cubecl_core::CubeDim,
        cube_count: cubecl_core::CubeCount,
        input: crate::components::InputRuntimeArg<'a, MS, R>,
        output: crate::components::OutputRuntimeArg<'a, MS, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        config: Self::Config,
    ) {
        todo!()
    }
}
