use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::tile::dummy::FlashMatmul;
use crate::components::tile::dummy::FlashPrecision;

#[derive(CubeType)]
pub struct AccumulatorFragment<FP: FlashPrecision, FM: FlashMatmul<FP>> {
    tmp_smem: SharedMemory<FP::A>,
    pub fragment: FM::Accumulator,
    #[cube(comptime)]
    config: FM::Config,
}

#[cube]
impl<FP: FlashPrecision, FM: FlashMatmul<FP>> AccumulatorFragment<FP, FM> {
    pub fn new(#[comptime] config: FM::Config) -> AccumulatorFragment<FP, FM> {
        comment!("Allocating accumulator");
        let mut fragment = FM::allocate_accumulator(config);
        FM::zero_accumulator(&mut fragment, config);
        AccumulatorFragment::<FP, FM> {
            tmp_smem: SharedMemory::new(64),
            fragment,
            config,
        }
    }

    pub fn scale(&mut self, factor: FP::A) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        FM::write_results::<FP::A>(
            &self.fragment,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.config,
        );
        self.tmp_smem[index_0] *= factor;
        self.tmp_smem[index_1] *= factor;

        let tile = Tile::<FP::A> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };

        FM::tmp_fill_accumulator(&tile, &mut self.fragment, self.config);
    }
}
