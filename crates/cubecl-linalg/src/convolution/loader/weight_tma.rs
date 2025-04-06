use core::marker::PhantomData;

use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};
use cubecl_std::{CubeOption, FastDivmod};

use crate::matmul::components::stage::RowMajorTilingOrder;
use crate::matmul::components::{
    Ident, InputIdent, MatmulPrecision, global::Quantization, stage::multi_buffer::FullReader,
};
use crate::matmul::components::{
    global::{self, tensor_view::MappedTensorReader},
    stage::{ContiguousTilingLayout, Stage, StageConfig},
};

pub type TmaWeightTiling = ContiguousTilingLayout<RowMajorTilingOrder>;
pub type TmaWeightReader<MP> = FullReader<<MP as MatmulPrecision>::ES, TmaWeightTiling>;

#[derive(CubeType)]
pub struct TmaWeightLoader<MP: MatmulPrecision, S: StageConfig> {
    pub tensor_view: MappedTensorReader<MP::EI>,
    pub stage: Stage<MP::ES, TmaWeightTiling>,
    #[cube(comptime)]
    _config: PhantomData<S>,
}

#[cube]
impl<MP: MatmulPrecision, S: StageConfig> TmaWeightLoader<MP, S> {
    pub fn new<G: global::GlobalConfig>(
        tensor: TensorMap<MP::EI>,
        x: u32,
        y: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let stage = Stage::new_aligned::<G::SmmConfig>(Ident::Rhs, 128u32, config.to_smm_config());

        let tensor_view = MappedTensorReader::new(tensor, x, y, 0);

        TmaWeightLoader::<MP, S> {
            tensor_view,
            stage,
            _config: PhantomData::<S>,
        }
    }

    pub fn fill_stage(
        this: &mut Self,
        barrier: &Barrier<MP::ES>,
        padded_channels: FastDivmod,
        #[comptime] config: S,
    ) {
        if UNIT_POS == 0 {
            let k = this.tensor_view.tile_x;
            let out_c = this.tensor_view.tile_y;
            let tiling_dims = config.tiling_dimensions(Ident::Rhs);

            let tensor = this.tensor_view.tensor.try_cast_unchecked();

            #[unroll]
            for tile in 0..tiling_dims.tile_count() {
                let (x, y) = TmaWeightTiling::to_x_y::<S>(tile, Ident::Rhs, config);
                let mut tile = this.stage.get_tile::<S>(x, y, Ident::Rhs, config);
                let mut slice = tile.slice.to_slice_mut();

                let k = x * tiling_dims.tile_shape_row() + k;
                let out_c = y * tiling_dims.tile_shape_col() + out_c;
                let (in_c, k_idx) = padded_channels.div_mod(k);

                barrier.tma_load_3d(&tensor, &mut slice, k_idx as i32, in_c as i32, out_c as i32);
            }
        }
    }

    pub fn reader(this: &Self) -> TmaWeightReader<MP> {
        TmaWeightReader::<MP>::new(this.stage, InputIdent::Rhs)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}
