use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::global::dummy::AttentionReader;
use cubecl_matmul::components::{global::memory::GlobalMemoryConfig, stage::StageMemoryConfig};
use cubecl_std::{CubeOption, tensor::r#virtual::VirtualTensor};

use crate::components::{
    AttentionIdent, AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AttentionTilingScheme, AvailableLineSizes, attention_types::*,
    global::dummy::QueryReader, stage::StageAttentionConfig,
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
pub trait GlobalAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// Writes to Out at the same offset it loaded Query
    type Writer: CubeType;

    /// Loads to SMEM transposed
    type KeyReader: AttentionReader<KS<AP>, Self::Config>;
    /// Loads to SMEM as is
    type ValueReader: AttentionReader<VS<AP>, Self::Config>;
    /// Loads to SMEM as is
    type MaskReader: CubeType;

    /// The configuration type associated with this Attention.
    type Config: GlobalAttentionConfig;

    fn execute(
        query_reader: QueryReader<AP>,
        key_reader: Self::KeyReader,
        value_reader: Self::ValueReader,
        mask_reader: Self::MaskReader,
        writer: Self::Writer,
        seq_q: u32,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    );

    fn init_query_reader(
        q_offset: u32,
        query: VirtualTensor<QG<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<AP>;

    fn init_key_reader(
        key: VirtualTensor<KG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader;

    fn init_value_reader(
        value: VirtualTensor<VG<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader;

    fn init_mask_reader(
        q_offset: u32,
        mask: CubeOption<VirtualTensor<MSK<AP>>>,
        seq_kv_shape: u32,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader;

    fn init_writer(
        q_offset: u32,
        out: VirtualTensor<OG<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer;
}

/// Configuration for the Global Attention level
pub trait GlobalAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageAttentionConfig;

    fn stage_config(&self) -> Self::StageConfig;
    fn score_stage_memory_config(&self) -> StageMemoryConfig;
    fn value_stage_memory_config(&self) -> StageMemoryConfig;

    fn cube_dim(&self) -> CubeDim;
    fn plane_dim(&self) -> u32;
    fn global_memory_config(&self, ident: AttentionIdent) -> GlobalMemoryConfig;

    fn tiling_scheme(&self) -> AttentionTilingScheme;

    fn causal_mask(&self) -> bool;
}
