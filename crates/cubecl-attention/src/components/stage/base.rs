use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEventListener, WriteTiling},
    stage::{ContiguousTilingLayout, RowMajorTilingOrder, StageFamily},
};
use std::{fmt::Debug, hash::Hash};

use crate::components::{
    AttentionLineSizes, AttentionPrecision, AttentionProblem, AttentionSelection,
    AttentionSetupError, AvailableLineSizes, global::GlobalAttentionConfig, stage::RunningState,
};
use crate::components::{AttentionTilingScheme, global::simple::QueryReader};
use crate::components::{attention_types::*, tile::TileAttentionConfig};
use crate::components::{global::simple::MaskReader, stage::AttentionPartitioner};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

/// A family of [TileAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait StageAttentionFamily: Send + Sync + 'static {
    /// The specific [TileAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: StageAttention<
            AP,
            Config = Self::Config,
            KeyStage = <Self::KeyStage as StageFamily>::Stage<KS<AP>, AttentionTilingLayout>,
            ValueStage = <Self::ValueStage as StageFamily>::Stage<VS<AP>, AttentionTilingLayout>,
            OutStage = <Self::OutStage as StageFamily<ReadWrite>>::Stage<OS<AP>, WriteTiling>,
        >;

    /// The configuration type associated with this Attention family.
    type Config: StageAttentionConfig;

    type KeyStage: StageFamily;
    type ValueStage: StageFamily;
    type OutStage: StageFamily<ReadWrite>;

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
pub trait StageAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    type KeyStage: CubeType;
    type ValueStage: CubeType;
    type OutStage: CubeType;

    /// The configuration type associated with this Attention.
    type Config: StageAttentionConfig;
    type Partitioner: AttentionPartitioner;

    type QueryRegisters: CubeType;
    type KeyValueRegisters: CubeType;
    type SoftmaxRegisters: CubeType;
    type AccumulatorRegisters: CubeType;
    type MaskRegisters: CubeType;

    fn init_state(#[comptime] config: Self::Config) -> Sequence<RunningState<SM<AP>>>;

    fn execute(
        query: &Self::QueryRegisters,
        key_stage: &Self::KeyStage,
        value_stage: &Self::ValueStage,
        key_value: &mut Self::KeyValueRegisters,
        mask_reader: &MaskReader<AP>,
        mask_partition: &mut Self::MaskRegisters,
        score: &mut Self::SoftmaxRegisters,
        accumulator: &mut Self::AccumulatorRegisters,
        prev_state: &mut Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    );

    fn rescale(
        acc: &mut Self::AccumulatorRegisters,
        state: Sequence<RunningState<SM<AP>>>,
        #[comptime] config: Self::Config,
    );

    fn write<W: WriteEventListener, G: GlobalAttentionConfig>(
        acc: &Self::AccumulatorRegisters,
        stage: &mut Self::OutStage,
        writer: &mut W,
        #[comptime] tile_config: Self::Config,
    );

    fn init_query(#[comptime] config: Self::Config) -> Self::QueryRegisters;
    fn init_key_value(#[comptime] config: Self::Config) -> Self::KeyValueRegisters;
    fn init_mask(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: Self::Config,
    ) -> Self::MaskRegisters;
    fn init_softmax(#[comptime] config: Self::Config) -> Self::SoftmaxRegisters;
    fn init_accumulator(#[comptime] config: Self::Config) -> Self::AccumulatorRegisters;

    fn read_query(
        reader: &QueryReader<AP>,
        registers: &mut Self::QueryRegisters,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Tile Attention level
pub trait StageAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type TileAttentionConfig: TileAttentionConfig;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;

    fn tile_config(&self) -> Self::TileAttentionConfig;

    fn tiling_scheme(&self) -> AttentionTilingScheme;
    fn reuse_key_value(&self) -> bool;

    fn num_rows_per_unit(&self) -> u32;
}
