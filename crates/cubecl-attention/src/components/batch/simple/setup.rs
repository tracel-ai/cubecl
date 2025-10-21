use std::marker::PhantomData;

use cubecl_core::client::ComputeClient;

use crate::components::{
    Args, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    attention_types::*,
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

    fn setup<AP: crate::components::AttentionPrecision, R: cubecl_core::Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, crate::components::AttentionSetupError> {
        let global_config = GA::setup::<AP, R>(client, problem, selection, line_sizes)?;

        SimpleBatchConfig::new(
            global_config,
            selection
                .hypercube_selection
                .to_hypercube_config(problem, client.properties().hardware.max_cube_count.clone()),
            problem.seq_kv as u32,
        )
        .validate(problem)
    }

    unsafe fn launch_unchecked<
        'a,
        AS: crate::components::AttentionSpec,
        R: cubecl_core::Runtime,
    >(
        client: &cubecl_core::prelude::ComputeClient<<R as cubecl_core::Runtime>::Server>,
        cube_dim: cubecl_core::CubeDim,
        cube_count: cubecl_core::CubeCount,
        input: crate::components::InputRuntimeArg<'a, AS, R>,
        output: crate::components::OutputRuntimeArg<'a, AS, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        config: Self::Config,
    ) {
        unsafe {
            attention::launch_unchecked::<
                Args<AS>,
                QG<AS>,
                QT<AS>,
                KG<AS>,
                KS<AS>,
                VG<AS>,
                VS<AS>,
                KVT<AS>,
                SM<AS>,
                ACC<AS>,
                MSK<AS>,
                OG<AS>,
                OS<AS>,
                Self,
                R,
            >(
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
