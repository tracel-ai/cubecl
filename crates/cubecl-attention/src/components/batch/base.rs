use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionSpec, AvailableLineSizes, InputRuntimeArg, OutputRuntimeArg,
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
    unsafe fn launch_unchecked<'a, MS: AttentionSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        cube_count_input: CubeCountInputArgs<'a, R>,
        config: Self::Config,
    );

    /// Constructs the configuration based on the Attention problem, selection, and line sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &AttentionProblem,
        selection: &AttentionSelection,
        line_sizes: &AttentionLineSizes,
    ) -> Result<Self::Config, AttentionSetupError>;

    /// Filters out line sizes that are incompatible with this Attention family.
    ///
    /// By default, returns the input unchanged.
    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }
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
