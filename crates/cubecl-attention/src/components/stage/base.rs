use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::{
    global::{WriteEventListener, WriteTiling, read::sync_full_cyclic::SyncFullCyclicLoading},
    stage::{ContiguousTilingLayout, RowMajorTilingOrder, StageFamily, StageMemoryConfig},
};
use std::{fmt::Debug, hash::Hash};

use crate::components::{
    AttentionBlueprint, AttentionPartitionSize, AttentionPrecision, AttentionSetupError,
    AttentionStageSize, global::GlobalAttentionConfig, stage::RunningState,
};
use crate::components::{attention_types::*, tile::TileAttentionConfig};
use crate::components::{global::simple::MaskReader, stage::AttentionPartitioner};
use crate::components::{
    global::simple::QueryReader,
    stage::{plane::PlanePartitionStageConfig, unit::UnitPartitionStageConfig},
};
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

pub type AttentionTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type AttentionLoadingStrategy = SyncFullCyclicLoading<RowMajorTilingOrder>;

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

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_blueprint(
        blueprint: &AttentionBlueprint,
    ) -> Result<Self::Config, AttentionSetupError>;
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
        #[comptime] config: Self::Config,
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

    fn tile_config(&self) -> Self::TileAttentionConfig;

    fn elements_in_tile_seq_q(&self) -> u32;
    fn elements_in_tile_seq_kv(&self) -> u32;

    fn elements_in_partition_seq_q(&self) -> u32;
    fn elements_in_partition_seq_kv(&self) -> u32;
    fn elements_in_stage_seq_q(&self) -> u32;

    fn plane_dim(&self) -> u32;
    fn num_planes(&self) -> u32;

    fn key_smem_config(&self) -> StageMemoryConfig;
    fn value_smem_config(&self) -> StageMemoryConfig;
    fn out_smem_config(&self) -> StageMemoryConfig;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PartitionAttentionConfig<TC: TileAttentionConfig> {
    Unit(UnitPartitionStageConfig<TC>),
    Plane(PlanePartitionStageConfig<TC>),
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SharedPartitionAttentionConfig<TC: TileAttentionConfig> {
    pub tile_config: TC,
    pub partition_size: AttentionPartitionSize,
    pub stage_size: AttentionStageSize,
    pub reuse_key_value: bool,
    pub num_planes: u32,
    pub key_smem_config: StageMemoryConfig,
    pub value_smem_config: StageMemoryConfig,
    pub out_smem_config: StageMemoryConfig,
}

impl<TC: TileAttentionConfig> PartitionAttentionConfig<TC> {
    pub fn shared(&self) -> SharedPartitionAttentionConfig<TC> {
        match self {
            PartitionAttentionConfig::Unit(unit_partition_stage_config) => {
                unit_partition_stage_config.shared
            }
            PartitionAttentionConfig::Plane(plane_partition_stage_config) => {
                plane_partition_stage_config.shared
            }
        }
    }
}

pub fn validate<TC: TileAttentionConfig>(
    config: PartitionAttentionConfig<TC>,
) -> Result<PartitionAttentionConfig<TC>, AttentionSetupError> {
    let tile_size = config.shared().tile_config.attention_tile_size();
    let partition_size = config.shared().partition_size;

    let head_val_different = tile_size.head_dim != tile_size.val_dim
        || partition_size.head_dim != partition_size.val_dim;

    if head_val_different {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Differing head dim and val dim is not yet supported".to_string(),
        )));
    }

    // This check is stricter than the previous one, but the other may be removed
    // eventually while this one will always remain true.
    if config.shared().reuse_key_value && head_val_different {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
        "When reusing key/value, head_dim must equal val_dim in both tile_size and partition_size."
            .to_string(),
    )));
    }

    Ok(config)
}

impl<TC: TileAttentionConfig> StageAttentionConfig for PartitionAttentionConfig<TC> {
    type TileAttentionConfig = TC;

    fn tile_config(&self) -> Self::TileAttentionConfig {
        self.shared().tile_config
    }

    fn num_planes(&self) -> u32 {
        self.shared().num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.shared().tile_config.plane_dim()
    }

    fn key_smem_config(&self) -> StageMemoryConfig {
        self.shared().key_smem_config
    }

    fn value_smem_config(&self) -> StageMemoryConfig {
        self.shared().value_smem_config
    }

    fn out_smem_config(&self) -> StageMemoryConfig {
        self.shared().out_smem_config
    }

    fn elements_in_tile_seq_q(&self) -> u32 {
        self.tile_config().attention_tile_size().seq_q
    }

    fn elements_in_tile_seq_kv(&self) -> u32 {
        self.tile_config().attention_tile_size().seq_kv
    }

    fn elements_in_partition_seq_q(&self) -> u32 {
        self.shared().partition_size.seq_q * self.elements_in_tile_seq_kv()
    }

    fn elements_in_partition_seq_kv(&self) -> u32 {
        self.shared().partition_size.seq_kv * self.elements_in_tile_seq_kv()
    }

    fn elements_in_stage_seq_q(&self) -> u32 {
        self.shared().stage_size.seq_q * self.elements_in_partition_seq_q()
    }
}
