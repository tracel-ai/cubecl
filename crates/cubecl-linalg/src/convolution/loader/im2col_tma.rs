use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::{
    convolution::reader::tma::Im2colTmaReader,
    matmul::components::{
        Ident, Lhs, MatmulPrecision,
        global::{
            CopyMechanism,
            single_stage::{AsyncFullLoader, FullLoader},
        },
        stage::{
            ColMajorTilingOrder, ContiguousTilingLayout, RowMajorTilingOrder, StageConfig,
            multi_buffer::Reader,
        },
    },
};
use crate::{
    convolution::{ConvGemmConfig, reader::im2col::Im2colReader},
    matmul::components::stage::Stage,
};

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct Im2colTmaLoader<MP: MatmulPrecision, G: ConvGemmConfig> {
    pub map: Im2colTmaReader<MP::EI>,
    pub stage: Stage<MP::ES, ContiguousTilingLayout<ColMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: ConvGemmConfig> FullLoader<MP, G> for Im2colTmaLoader<MP, G> {
    type StageReader = Reader<Lhs, MP::ES, ContiguousTilingLayout<ColMajorTilingOrder>>;

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.map.update_view(k_offset);
    }

    fn reader(this: &Self) -> Self::StageReader {
        Reader::new(this.stage)
    }
}

#[cube]
impl<MP: MatmulPrecision, G: ConvGemmConfig> AsyncFullLoader<MP, G> for Im2colTmaLoader<MP, G> {
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

    fn clear_stage(this: &mut Self, #[comptime] config: G) {
        this.stage
            .clear::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
    }
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
}

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct SimpleIm2col;

#[cube]
impl SimpleIm2col {
    pub fn load_to_slice<MP: MatmulPrecision, G: ConvGemmConfig>(
        read_view: &Im2colReader<MP::EI>,
        slice: &mut SliceMut<Line<MP::ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_tiling.total_size();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_tiling.tile_size();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = ContiguousTilingLayout::<RowMajorTilingOrder>::to_x_y::<
                G::SmmConfig,
            >(nth_tile, ident, config.to_smm_config());

            let line_read =
                read_view.load_simple::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            slice[unit_position / line_size] = Line::cast_from(line_read);
        }
    }
}

pub fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
    Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
