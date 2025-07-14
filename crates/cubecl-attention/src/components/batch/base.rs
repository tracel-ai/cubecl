use cubecl_core::prelude::*;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionSpec, AvailableLineSizes, InputRuntimeArg, OutputRuntimeArg,
    batch::{CubeCountInputArgs, HypercubeConfig},
    global::GlobalConfig,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [BatchAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait BatchAttentionFamily: Send + Sync + 'static {
    /// The specific [BatchAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: BatchAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: BatchConfig;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<'a, MS: AttentionSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
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
        client: &ComputeClient<R::Server, R::Channel>,
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
    type Config: BatchConfig;
}

/// Configuration for the Batch Attention level
pub trait BatchConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type GlobalConfig: GlobalConfig;

    fn hypercube_config(&self) -> HypercubeConfig;
    fn cube_dim(&self) -> CubeDim;
}
