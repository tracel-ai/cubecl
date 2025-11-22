use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;

use crate::components::{
    AttentionElems, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    InputRuntimeArg, OutputRuntimeArg,
    args::AttentionArgs,
    batch::{
        BatchAttentionFamily,
        entry_point::attention,
        simple::{SimpleBatchAttention, config::SimpleBatchConfig},
    },
    global::GlobalAttentionFamily,
};

pub struct SimpleBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for SimpleBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = SimpleBatchAttention<AP, GA::Attention<AP>>;
    type Config = SimpleBatchConfig<GA::Config>;

    fn setup<R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        let global_config = GA::setup::<R>(client, problem, selection, line_sizes, dtypes)?;

        SimpleBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<<R as cubecl_core::Runtime>::Server>,
        cube_dim: cubecl_core::CubeDim,
        cube_count: cubecl_core::CubeCount,
        input: InputRuntimeArg<'a, AA, R>,
        output: OutputRuntimeArg<'a, AA, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        config: Self::Config,
        dtypes: &AttentionElems,
    ) {
        unsafe {
            attention::launch_unchecked::<AA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                config,
                dtypes.into(),
            );
        }
    }
}
