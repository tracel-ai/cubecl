use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionPrecision;
use crate::components::tile::ValueMatmul;

#[derive(CubeType)]
pub struct ProbFragment<AP: AttentionPrecision, VM: ValueMatmul<AP>> {
    tmp_smem: SharedMemory<AP::EA>,
    pub fragment: VM::Lhs,
}

#[cube]
impl<AP: AttentionPrecision, VM: ValueMatmul<AP>> ProbFragment<AP, VM> {
    pub fn new(fragment: VM::Lhs, tmp_smem: SharedMemory<AP::EA>) -> Self {
        ProbFragment::<AP, VM> { tmp_smem, fragment }
    }

    pub fn row_sum(&self) -> AP::EA {
        let row = UNIT_POS_X / 4;
        let row_offset = row * 8;

        let mut rowsum = AP::EA::from_int(0);
        for i in 0..8 {
            rowsum += self.tmp_smem[row_offset + i];
        }

        rowsum
    }
}
