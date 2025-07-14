use cubecl_core::prelude::*;

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [StageAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific [StageAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: StageConfig;

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
pub trait StageAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Attention.
    type Config: StageConfig;
}

/// Configuration for the Stage Attention level
pub trait StageConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {}
