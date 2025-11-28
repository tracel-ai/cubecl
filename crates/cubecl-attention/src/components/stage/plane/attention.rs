use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords1d;

use crate::components::{
    global::simple::PlaneAttentionWriter,
    stage::{
        BroadcastReducer, partition_attention::PartitionAttention,
        partitioner::AttentionPartitioner,
    },
};

use crate::components::{stage::SharedPartitionAttentionConfig, tile::TileAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanePartitionStageConfig<TC: TileAttentionConfig> {
    pub shared: SharedPartitionAttentionConfig<TC>,
}

pub type PlanePartitionAttention<AP, SK, SV, SO, TA> =
    PartitionAttention<AP, SK, SV, SO, TA, PlanePartitioner>;

pub struct PlanePartitioner {}

#[cube]
impl AttentionPartitioner for PlanePartitioner {
    type Reducer = BroadcastReducer;
    type Writer<ES: Float, EG: Float> = PlaneAttentionWriter<ES, EG>;

    fn seq_q_index() -> Coords1d {
        UNIT_POS_Y
    }
}
