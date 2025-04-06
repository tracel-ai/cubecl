use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::{
    convolution::ConvGemmConfig,
    matmul::components::{InputIdent, stage::Stage},
};
use crate::{
    convolution::reader::tma::Im2colTmaReader,
    matmul::components::{
        Ident, MatmulPrecision,
        global::CopyMechanism,
        stage::{
            ColMajorTilingOrder, ContiguousTilingLayout, StageConfig, multi_buffer::FullReader,
        },
    },
};

pub type TmaIm2colTiling = ContiguousTilingLayout<ColMajorTilingOrder>;
pub type TmaIm2colReader<MP> = FullReader<<MP as MatmulPrecision>::ES, TmaIm2colTiling>;

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct Im2colTmaLoader<MP: MatmulPrecision, G: ConvGemmConfig> {
    pub map: Im2colTmaReader<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<ColMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: ConvGemmConfig> Im2colTmaLoader<MP, G> {
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());

        let height = tensor.shape(1);
        let width = tensor.shape(2);

        let (w_offset, nh_offset) = (x_offset % width, x_offset / width);
        let (h_offset, n_offset) = (nh_offset % height, nh_offset / height);

        let map = Im2colTmaReader::<MP::EI>::new(tensor, n_offset, h_offset, w_offset, y_offset);

        Im2colTmaLoader::<MP, G> {
            map,
            stage,
            _config: PhantomData::<G>,
        }
    }

    fn fill_stage<CM: CopyMechanism<MP::ES>>(this: &mut Self, bar: &CM, #[comptime] config: G) {
        let tmm = config.to_smm_config();
        let tiling_dims = tmm.tiling_dimensions(Ident::Lhs);
        if UNIT_POS == 0 {
            let m_size = tiling_dims.total_row();
            let k_size = tiling_dims.tile_shape_col();
            let slice_size = m_size * k_size;
            let mut full_stage = this.stage.as_slice_mut();
            let tensor = this.map.tensor.try_cast_unchecked();
            let channels = config.padded_channels();

            #[unroll]
            for tile_k in 0..tiling_dims.tile_count_col() {
                let k = this.map.k_offset + tile_k * k_size;
                let (channel_start, k_idx) = (k % channels, k / channels);
                let (k_x, k_y) = (k_idx % config.kernel_size(1), k_idx / config.kernel_size(0));
                let slice_start = tile_k * slice_size;
                let mut stage = full_stage.slice_mut(slice_start, slice_start + slice_size);
                let coords = (
                    this.map.n_offset as i32,
                    this.map.h_offset as i32 - config.padding(0),
                    this.map.w_offset as i32 - config.padding(1),
                    channel_start as i32,
                );
                let offsets = (k_y as u16, k_x as u16);
                CM::tma_load_im2col_4d(bar, &tensor, &mut stage, coords, offsets);
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
