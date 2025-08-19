use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::Tile;

use crate::components::AttentionPrecision;
use crate::components::tile::ValueMatmul;

#[derive(CubeType)]
pub struct AccumulatorFragment<AP: AttentionPrecision, VM: ValueMatmul<AP>> {
    tmp_smem: SharedMemory<AP::EA>,
    pub fragment: VM::Accumulator,
    #[cube(comptime)]
    config: VM::Config,
}

#[cube]
impl<AP: AttentionPrecision, VM: ValueMatmul<AP>> AccumulatorFragment<AP, VM> {
    pub fn new(#[comptime] config: VM::Config) -> AccumulatorFragment<AP, VM> {
        let mut fragment = VM::allocate_accumulator(config);
        VM::zero_accumulator(&mut fragment, config);
        AccumulatorFragment::<AP, VM> {
            tmp_smem: SharedMemory::new(64),
            fragment,
            config,
        }
    }

    pub fn scale(&mut self, factor: AP::EA) {
        let index_0 = 2 * UNIT_POS_X;
        let index_1 = index_0 + 1;

        VM::write_results::<AP::EA>(
            &self.fragment,
            &mut self.tmp_smem.to_slice_mut().try_cast_unchecked(),
            self.config,
        );
        self.tmp_smem[index_0] *= factor;
        self.tmp_smem[index_1] *= factor;

        let tile = Tile::<AP::EA> {
            slice: self.tmp_smem.to_slice().try_cast_unchecked(),
            stride: 8,
            layout: MatrixLayout::RowMajor,
        };

        VM::fill_accumulator(&tile, &mut self.fragment, self.config);
    }
}
