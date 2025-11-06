use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::tile::FragmentAttentionConfig;
use crate::components::tile::RowWise;
use crate::components::tile::RowwiseFormat;
use crate::components::stage::ReduceOp;
use crate::components::stage::Reducer;

#[derive(CubeType)]
/// Trivial reducer for one unit
pub struct UnitReducer {}

#[cube]
impl Reducer for UnitReducer {
    fn reduce<E: Float, F: RowwiseFormat<E>, RO: ReduceOp<E>, FC: FragmentAttentionConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] _config: FC,
    ) {
        RO::reduce_local_accumulate::<F>(data, vals);
    }
}
