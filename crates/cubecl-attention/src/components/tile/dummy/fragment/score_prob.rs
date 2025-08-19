use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::AttentionPrecision;
use crate::components::tile::{ScoreMatmul, ValueMatmul};

#[derive(CubeType)]
pub struct ScoreFragment<AP: AttentionPrecision, SM: ScoreMatmul<AP>> {
    tmp_smem: SharedMemory<AP::EA>,
    pub fragment: SM::Accumulator,
    #[cube(comptime)]
    config: SM::Config,
}

#[cube]
impl<AP: AttentionPrecision, SM: ScoreMatmul<AP>> ScoreFragment<AP, SM> {
    pub fn new(#[comptime] config: SM::Config) -> Self {
        let mut fragment = SM::allocate_accumulator(config);
        SM::zero_accumulator(&mut fragment, config);
        ScoreFragment::<AP, SM> {
            tmp_smem: SharedMemory::<AP::EA>::new(64),
            fragment,
            config,
        }
    }

    pub fn multiply_score(&mut self, factor: AP::EA) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        SM::write_results::<AP::EA>(
            &self.fragment,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.config,
        );
        self.tmp_smem[index_0] *= factor;
        self.tmp_smem[index_1] *= factor;
        sync_plane();
    }

    pub fn row_max(&mut self, base: AP::EA) -> AP::EA {
        let row = UNIT_POS_X / 4;
        let row_offset = row * 8;
        let mut rowmax = base;

        for i in 0..8 {
            let ts = self.tmp_smem[row_offset + i];
            if ts > rowmax {
                rowmax = ts;
            }
        }

        rowmax
    }

    pub fn to_prob<VM: ValueMatmul<AP>>(
        &mut self,
        m: AP::EA,
        #[comptime] value_config: VM::Config,
    ) -> ProbFragment<AP, VM> {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        self.tmp_smem[index_0] = Exp::exp(self.tmp_smem[index_0] - m);
        self.tmp_smem[index_1] = Exp::exp(self.tmp_smem[index_1] - m);
        sync_plane();

        let tile = Tile::<AP::EA> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };

        // TODO this VM::Lhs is unfortunate, technically could be the same as SM::Accumulator but it's hard to make types match
        let mut fragment = VM::allocate_lhs(value_config);
        VM::fill_lhs(&tile, &mut fragment, value_config);

        ProbFragment::new(fragment, self.tmp_smem)
    }
}

#[derive(CubeType)]
pub struct ProbFragment<AP: AttentionPrecision, VM: ValueMatmul<AP>> {
    tmp_smem: SharedMemory<AP::EA>,
    pub fragment: VM::Lhs,
}

#[cube]
impl<AP: AttentionPrecision, VM: ValueMatmul<AP>> ProbFragment<AP, VM> {
    fn new(fragment: VM::Lhs, tmp_smem: SharedMemory<AP::EA>) -> Self {
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
