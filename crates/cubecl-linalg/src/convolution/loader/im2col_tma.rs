use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use cubecl_std::{FastDivmod, tensor::r#virtual::VirtualTensor};
use std::marker::PhantomData;

use crate::{
    convolution::reader::tma::Im2colTmaReader,
    matmul::components::{
        Ident, MatmulPrecision,
        stage::{
            ColMajorTilingOrder, ContiguousTilingLayout, StageConfig, multi_buffer::FullReader,
        },
    },
};
use crate::{
    convolution::{ConvGemmConfig, base::RuntimeArgs},
    matmul::components::{InputIdent, stage::Stage},
};

pub type TmaIm2colTiling = ContiguousTilingLayout<ColMajorTilingOrder>;
pub type TmaIm2colReader<MP> = FullReader<<MP as MatmulPrecision>::ES, TmaIm2colTiling>;

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct TmaIm2colLoader<MP: MatmulPrecision, G: ConvGemmConfig> {
    pub map: Im2colTmaReader<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<ColMajorTilingOrder>>,
    padded_channels: FastDivmod,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: ConvGemmConfig> TmaIm2colLoader<MP, G> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new_aligned::<G::SmmConfig>(Ident::Lhs, 128u32, config.to_smm_config());

        let (nh_offset, w_offset) = runtime_args.out_w.div_mod(x_offset);
        let (n_offset, h_offset) = runtime_args.out_h.div_mod(nh_offset);

        let map = Im2colTmaReader::<MP::EI>::new(tensor, n_offset, h_offset, w_offset, y_offset);

        TmaIm2colLoader::<MP, G> {
            map,
            stage,
            padded_channels: runtime_args.padded_channels,
            _config: PhantomData::<G>,
        }
    }

    pub fn fill_stage(this: &mut Self, bar: &Barrier<MP::ES>, #[comptime] config: G) {
        let tmm = config.to_smm_config();
        let tiling_dims = tmm.tiling_dimensions(Ident::Lhs);
        if UNIT_POS == 0 {
            let m_size = tiling_dims.total_row();
            let k_size = tiling_dims.tile_shape_col();
            let slice_size = m_size * k_size;
            let mut full_stage = this.stage.as_slice_mut();
            let tensor = this.map.tensor.try_cast_unchecked();

            let in_h = (this.map.h_offset * config.stride(0)) as i32 - config.padding(0);
            let in_w = (this.map.w_offset * config.stride(1)) as i32 - config.padding(1);

            #[unroll]
            for tile_k in 0..tiling_dims.tile_count_col() {
                let k = this.map.k_offset + tile_k * k_size;
                let (k_idx, channel_start) = this.padded_channels.div_mod(k);
                let (k_x, k_y) = (k_idx % config.kernel_size(1), k_idx / config.kernel_size(1));
                let slice_start = tile_k * slice_size;
                let mut stage = full_stage.slice_mut(slice_start, slice_start + slice_size);

                let offset_y = k_y * config.dilation(0);
                let offset_x = k_x * config.dilation(1);

                bar.tma_load_im2col_4d(
                    &tensor,
                    &mut stage,
                    this.map.n_offset as i32,
                    in_h,
                    in_w,
                    channel_start as i32,
                    offset_y as u16,
                    offset_x as u16,
                );
            }
        }
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.map.update_view(k_offset);
    }

    pub fn reader(this: &Self) -> TmaIm2colReader<MP> {
        TmaIm2colReader::<MP>::new(this.stage, InputIdent::Lhs)
    }
}
