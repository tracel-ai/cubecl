use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{TilingScheme, stage::StageMemoryConfig};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, FlashIdent, stage::StageAttentionConfig,
};
use std::{fmt::Debug, hash::Hash};

/// A family of [GlobalAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait GlobalAttentionFamily: Send + Sync + 'static {
    /// The specific [GlobalAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: GlobalAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: GlobalAttentionConfig;

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
pub trait GlobalAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// Simply loads to registers once before loop
    type QueryLoader: CubeType;
    /// Writes to Out at the same offset it loaded Query
    type Writer: CubeType;
    /// Holds out tmp accumulated
    type Accumulator: CubeType;

    /// Loads to SMEM transposed
    type KeyLoader: CubeType;
    /// Loads to SMEM as is
    type ValueLoader: CubeType;

    /// The configuration type associated with this Attention.
    type Config: GlobalAttentionConfig;

    fn execute(
        query_loader: Self::QueryLoader,
        key_loader: Self::KeyLoader,
        value_loader: Self::ValueLoader,
        writer: Self::Writer,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    );

    fn init_query_loader(query: VirtualTensor<AP::EI>) -> Self::QueryLoader;
    fn init_key_loader(
        key: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyLoader;
    fn init_value_loader(
        value: VirtualTensor<AP::EI>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueLoader;
    fn init_writer(out: VirtualTensor<AP::EO, ReadWrite>) -> Self::Writer;
    fn init_accumulator() -> Self::Accumulator;
}

/// Configuration for the Global Attention level
pub trait GlobalAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageAttentionConfig;
    type ScoreStageMemoryConfig: StageMemoryConfig;
    type ValueStageMemoryConfig: StageMemoryConfig;

    fn stage_config(&self) -> Self::StageConfig;
    fn score_stage_memory_config(&self) -> Self::ScoreStageMemoryConfig;
    fn value_stage_memory_config(&self) -> Self::ValueStageMemoryConfig;

    fn cube_dim(&self) -> CubeDim;
    fn plane_dim(&self) -> u32;
    fn tc(&self) -> u32;

    fn global_line_size(&self, ident: FlashIdent) -> u32;
}
