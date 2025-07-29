use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::stage::{ContiguousTilingLayout, ReaderFamily, RowMajorTilingOrder};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, global::dummy::GlobalToTileReader,
};
use std::{fmt::Debug, hash::Hash};

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

/// A family of [StageAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific [StageAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<
            AP,
            Config = Self::Config,
            KeyReader = <Self::KeyReader as ReaderFamily>::Reader<AP::ES, AttentionTilingLayout>,
            ValueReader = <Self::ValueReader as ReaderFamily>::Reader<
                AP::ES,
                AttentionTilingLayout,
            >,
        >;

    /// The configuration type associated with this Attention family.
    type Config: StageConfig;

    type KeyReader: ReaderFamily;
    type ValueReader: ReaderFamily;

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
    type KeyReader: CubeType;
    type ValueReader: CubeType;
    type Accumulator: CubeType;
    type Writer: CubeType;

    /// The configuration type associated with this Attention.
    type Config: StageConfig;

    type State: CubeType;

    fn init_state() -> Self::State;
    fn zero_accumulator(acc: &mut Self::Accumulator);

    fn execute(
        query_reader: &GlobalToTileReader,
        key_reader: &Self::KeyReader,
        value_reader: &Self::ValueReader,
        acc: &mut Self::Accumulator,
        prev_state: &Self::State,
        #[comptime] config: Self::Config,
    ) -> Self::State;

    fn last_update(acc: &mut Self::Accumulator, prev_state: Self::State);

    fn write(acc: &Self::Accumulator, writer: Self::Writer);
}

/// Configuration for the Stage Attention level
pub trait StageConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;
    fn rows_per_plane(&self) -> u32;
}
