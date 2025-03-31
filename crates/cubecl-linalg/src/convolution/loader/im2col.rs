use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::matmul::components::{
    Ident, MatmulPrecision,
    global::single_stage::{FullLoader, SyncFullLoader},
    stage::{ContiguousTilingLayout, RowMajorTilingOrder, multi_buffer::LhsReader},
};
use crate::{
    convolution::{ConvGemmConfig, reader::im2col::Im2colReader},
    matmul::components::stage::MonoStage,
};

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct SimpleIm2colLoader<CS: MatmulPrecision, G: ConvGemmConfig> {
    pub tensor_view: Im2colReader<CS::EG>,
    pub stage: MonoStage<CS::ES, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<CS: MatmulPrecision, G: ConvGemmConfig> FullLoader<CS::EG, CS::ES, G>
    for SimpleIm2colLoader<CS, G>
{
    type StageReader = LhsReader<CS::ES, ContiguousTilingLayout<RowMajorTilingOrder>>;

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset);
    }

    fn reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }
}

#[cube]
impl<CS: MatmulPrecision, G: ConvGemmConfig> SyncFullLoader<CS::EG, CS::ES, G>
    for SimpleIm2colLoader<CS, G>
{
    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        SimpleIm2col::load_to_slice::<CS, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
    }
}

#[cube]
impl<CS: MatmulPrecision, G: ConvGemmConfig> SimpleIm2colLoader<CS, G> {
    pub fn new(
        tensor: VirtualTensor<CS::EG>,
        shape_out_y: u32,
        shape_out_x: u32,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = MonoStage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let shape_batch = tensor.shape(0);
        let shape_channel = tensor.shape(3);

        let shape_m = shape_batch * shape_out_y * shape_out_x;
        let shape_k = shape_channel * config.kernel_size(0) * config.kernel_size(1);

        let tensor_view = Im2colReader::<CS::EG>::new(
            tensor,
            shape_out_y,
            shape_out_x,
            x_offset,
            y_offset,
            shape_k,
            shape_channel,
            shape_m,
        );

        SimpleIm2colLoader::<CS, G> {
            tensor_view,
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
    pub fn load_to_slice<CS: MatmulPrecision, G: ConvGemmConfig>(
        read_view: &Im2colReader<CS::EG>,
        slice: &mut SliceMut<Line<CS::ES>>,
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

            let (tile_x, tile_y) =
                ContiguousTilingLayout::<RowMajorTilingOrder>::to_x_y::<G::SmmConfig>(
                    nth_tile,
                    stage_tiling.tile_count_row(),
                    stage_tiling.tile_count_col(),
                );

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
