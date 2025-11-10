use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords1d;

use crate::components::{
    global::simple::PlaneAttentionWriter,
    stage::{
        BroadcastReducer, partition_attention::PartitionAttention,
        partitioner::AttentionPartitioner, plane::PlanePartitionStageConfig,
    },
    tile::TileAttention,
};

pub type PlanePartitionAttention<AP, SK, SV, SO, FA> = PartitionAttention<
    AP,
    SK,
    SV,
    SO,
    FA,
    PlanePartitioner,
    PlanePartitionStageConfig<<FA as TileAttention<AP>>::Config>,
>;

pub struct PlanePartitioner {}

#[cube]
impl AttentionPartitioner for PlanePartitioner {
    type Reducer = BroadcastReducer;
    type Writer<ES: Float, EG: Float> = PlaneAttentionWriter<ES, EG>;

    fn seq_q_index() -> Coords1d {
        UNIT_POS_Y
    }
}
