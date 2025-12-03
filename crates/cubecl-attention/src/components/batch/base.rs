use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};

use crate::components::{
    AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError, InputRuntimeArg,
    OutputRuntimeArg,
    args::AttentionArgs,
    attention_types::*,
    batch::{CubeCountInput, CubeCountInputArgs, HypercubeConfig},
    global::GlobalAttentionConfig,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [BatchAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait BatchAttentionFamily: Send + Sync + 'static {
    /// The specific [BatchAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: BatchAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: BatchAttentionConfig;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, AA, R>,
        output: OutputRuntimeArg<'a, AA, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        dtypes: &AttentionElems,
        attention_blueprint: AttentionBlueprint,
    ) -> Result<(), LaunchError>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_blueprint(blueprint: AttentionBlueprint)
    -> Result<Self::Config, AttentionSetupError>;
}

#[cube]
pub trait BatchAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Attention.
    type Config: BatchAttentionConfig;

    fn execute(
        query: VirtualTensor<QG<AP>>,
        key: VirtualTensor<KG<AP>>,
        value: VirtualTensor<VG<AP>>,
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        cube_count_args: CubeCountInput,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Batch Attention level
pub trait BatchAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type GlobalConfig: GlobalAttentionConfig;

    fn global_config(&self) -> Self::GlobalConfig;

    fn hypercube_config(&self) -> HypercubeConfig;
    fn cube_dim(&self) -> CubeDim;
}
