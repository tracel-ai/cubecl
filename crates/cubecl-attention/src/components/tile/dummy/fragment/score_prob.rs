use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::tile::dummy::{FlashMatmul, FlashPrecision};

#[derive(CubeType)]
pub struct ScoreFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::SP>,
    pub fragment: FM::ScoreProb,
    #[cube(comptime)]
    config: FM::Config,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> ScoreFragment<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> Self {
        let mut fragment = FM::allocate_score_prob(config);
        FM::zero_score_prob(&mut fragment);
        ScoreFragment::<FP, FM> {
            tmp_smem: SharedMemory::<FP::SP>::new(64),
            fragment,
            config,
        }
    }

    pub fn multiply_score(&mut self, factor: FP::SP) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        FM::tmp_write_score_prob::<FP::SP>(
            &self.fragment,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.config,
        );
        self.tmp_smem[index_0] *= factor;
        self.tmp_smem[index_1] *= factor;
        sync_plane();
    }

    pub fn row_max(&mut self, base: FP::SP) -> FP::SP {
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

    pub fn to_prob(&mut self, m: FP::SP) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        self.tmp_smem[index_0] = Exp::exp(self.tmp_smem[index_0] - m);
        self.tmp_smem[index_1] = Exp::exp(self.tmp_smem[index_1] - m);
        sync_plane();

        // Should be directly in registers
        let tile = Tile::<FP::SP> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };
        FM::tmp_fill_prob(&tile, &mut self.fragment, self.config);
    }

    pub fn row_sum(&self) -> FP::SP {
        let row = UNIT_POS_X / 4;
        let row_offset = row * 8;

        let mut rowsum = FP::SP::from_int(0);
        for i in 0..8 {
            rowsum += self.tmp_smem[row_offset + i];
        }

        rowsum
    }
}
