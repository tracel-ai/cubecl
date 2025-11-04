use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::fragment::FragmentAttentionConfig;
use crate::components::fragment::FragmentOps;
use crate::components::tile::ReduceOp;
use crate::components::tile::Reducer;
use crate::components::tile::RowWise;

#[derive(CubeType)]
/// Trivial reducer for one unit
pub struct UnitReducer {}

#[cube]
impl Reducer for UnitReducer {
    fn reduce<E: Float, F: FragmentOps<E>, RO: ReduceOp<E>, FC: FragmentAttentionConfig>(
        vals: &mut RowWise<E>,
        data: &F,
        #[comptime] _config: FC,
    ) {
        RO::reduce_local_accumulate::<F>(data, vals);
    }
}
