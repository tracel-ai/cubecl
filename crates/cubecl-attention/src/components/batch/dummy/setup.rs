use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;

use crate::components::{
    Args, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection, EA, EI, EM,
    EO, ES,
    batch::{
        BatchAttentionFamily,
        dummy::{DummyBatchAttention, config::DummyBatchConfig},
        entry_point::attention,
    },
    global::GlobalAttentionFamily,
};

pub struct DummyBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for DummyBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = DummyBatchAttention<AP, GA::Attention<AP>>;
    type Config = DummyBatchConfig<GA::Config>;

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        let global_config = GA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        DummyBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<
        'a,
        AS: crate::components::AttentionSpec,
        R: cubecl_core::Runtime,
    >(
        client: &cubecl_core::prelude::ComputeClient<
            <R as cubecl_core::Runtime>::Server,
            <R as cubecl_core::Runtime>::Channel,
        >,
        cube_dim: cubecl_core::CubeDim,
        cube_count: cubecl_core::CubeCount,
        input: crate::components::InputRuntimeArg<'a, AS, R>,
        output: crate::components::OutputRuntimeArg<'a, AS, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        config: Self::Config,
    ) {
        unsafe {
            attention::launch_unchecked::<Args<AS>, EI<AS>, ES<AS>, EM<AS>, EA<AS>, EO<AS>, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
            );
        }
    }
}
