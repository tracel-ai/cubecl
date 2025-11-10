use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::stage::{ReduceOp, Reducer};
use crate::components::tile::{RowVal, RowWise, TileAttentionConfig};
use crate::components::tile::{RowwiseFormat, RowwiseFormatExpand};

#[derive(CubeType)]
/// Applies reduction on rows, masking planes that do not participate in the row
pub struct BroadcastReducer {}

#[cube]
impl Reducer for BroadcastReducer {
    fn reduce<E: Float, F: RowwiseFormat<E>, RO: ReduceOp<E>, FC: TileAttentionConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] _config: FC,
    ) {
        let num_units_per_row = data.num_units_per_row();
        let num_shares_within_plane = comptime!((num_units_per_row as f32).log2().ceil() as u32);

        let unit_pos = UNIT_POS_X;
        let unit_pos_in_row = unit_pos % num_units_per_row;

        RO::reduce_local_accumulate::<F>(data, vals);

        for i in 0..num_shares_within_plane {
            let offset = num_units_per_row >> (i + 1);
            let source_unit = unit_pos + offset;

            let value_from_source = rowwise_plane_broadcast(vals, source_unit);

            // Mask if outside the row
            let mask = unit_pos_in_row + offset >= num_units_per_row;
            RO::reduce_step_rowwise(vals, &value_from_source, mask);
        }

        // Broadcast back to subgroup
        let result = &rowwise_plane_broadcast(vals, unit_pos - unit_pos_in_row);
        vals.copy_from(result);
    }
}

#[cube]
fn rowwise_plane_broadcast<E: Float>(val: &RowWise<E>, source_unit: u32) -> RowWise<E> {
    let mut result = Sequence::new();

    #[unroll]
    for row in 0..val.num_rows {
        result.push(RowVal::<E> {
            val: plane_broadcast(val.index(row), source_unit),
        });
    }

    RowWise::<E> {
        num_rows: val.num_rows,
        vals: result,
    }
}
