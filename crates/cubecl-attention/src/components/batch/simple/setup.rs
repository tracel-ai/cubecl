use std::marker::PhantomData;

use cubecl_core::server::LaunchError;

use crate::components::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError, InputRuntimeArg,
    OutputRuntimeArg,
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

    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: cubecl_core::Runtime>(
        client: &cubecl_core::prelude::ComputeClient<R>,
        cube_dim: cubecl_core::CubeDim,
        cube_count: cubecl_core::CubeCount,
        input: InputRuntimeArg<'a, AA, R>,
        output: OutputRuntimeArg<'a, AA, R>,
        cube_count_input: crate::components::batch::CubeCountInputArgs<'a, R>,
        dtypes: &AttentionElems,
        blueprint: AttentionBlueprint,
    ) -> Result<(), LaunchError> {
        unsafe {
            attention::launch_unchecked::<AA, Self, R>(
                client,
                cube_count,
                cube_dim,
                input,
                output,
                cube_count_input,
                blueprint,
                dtypes.into(),
            )
        }
    }

    fn expand_blueprint(
        blueprint: AttentionBlueprint,
    ) -> Result<Self::Config, AttentionSetupError> {
        let global_config = GA::expand_blueprint(&blueprint)?;

        Ok(SimpleBatchConfig::new(
            global_config,
            blueprint.hypercube_blueprint.to_hypercube_config(),
        ))
    }
}
