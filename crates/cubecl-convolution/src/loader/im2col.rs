use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use cubecl_matmul::components::global::load::LoaderMode;
use cubecl_matmul::components::{InputPrecision, MatmulIdent, StageIdent};
use cubecl_std::div_ceil;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use crate::base::RuntimeArgs;
use crate::{ConvGemmConfig, reader::im2col::Im2colReader};
use cubecl_matmul::components::stage::{
    ContiguousTilingLayout, FullStageToTileReader, RowMajorTilingOrder, StageMemory,
};

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct SimpleIm2colLoader<IP: InputPrecision, G: ConvGemmConfig> {
    pub tensor_view: Im2colReader<IP::Global>,
    pub stage: StageMemory<IP::Stage, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<IP: InputPrecision, G: ConvGemmConfig> SimpleIm2colLoader<IP, G> {
    pub fn new(
        tensor: VirtualTensor<IP::Global>,
        x_offset: u32,
        y_offset: u32,
        runtime_args: &RuntimeArgs,
        #[comptime] config: G,
    ) -> Self {
        let stage = StageMemory::new::<G::StageMemoryConfig>(
            1u32,
            StageIdent::Lhs,
            config.stage_memory_config(),
        );

        let shape_m = runtime_args.size_m;
        let shape_k = runtime_args.size_k;

        let tensor_view = Im2colReader::<IP::Global>::new(
            tensor,
            comptime![runtime_args.out_shape.clone()],
            x_offset,
            y_offset,
            shape_k,
            shape_m,
        );

        SimpleIm2colLoader::<IP, G> {
            tensor_view,
            stage,
            _config: PhantomData::<G>,
        }
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset);
    }

    pub fn reader(
        this: &Self,
    ) -> FullStageToTileReader<IP::Stage, ContiguousTilingLayout<RowMajorTilingOrder>> {
        FullStageToTileReader::new(this.stage, StageIdent::Lhs)
    }

    pub fn fill_stage(this: &mut Self, #[comptime] config: G) {
        let line_size = config.global_line_size(MatmulIdent::Lhs);
        SimpleIm2col::load_to_slice::<IP, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(line_size),
            MatmulIdent::Lhs,
            config,
        );
    }
}

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct SimpleIm2col;

#[cube]
impl SimpleIm2col {
    pub fn load_to_slice<IP: InputPrecision, G: ConvGemmConfig>(
        tensor_reader: &Im2colReader<IP::Global>,
        slice: &mut SliceMut<Line<IP::Stage>>,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) {
        let line_size = config.global_line_size(ident);

        let num_stage_elements = config.tiling_scheme().elements_in_stage(ident);
        let total_units = comptime!(config.num_loading_planes(ident) * config.plane_dim());

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        if let LoaderMode::Strict = config.loader_mode() {
            let jump_length = comptime!(total_units * line_size);

            comptime! {
                            assert!(
                num_stage_elements % jump_length == 0,
                "Too many data will be loaded, resulting in out of bounds.
            Try setting line size and number of planes so that jump_length divides num_stage_elements."
            );
                    }

            let num_loads_per_unit = num_stage_elements / jump_length;

            for i in 0..num_loads_per_unit {
                let unit_position = unit_position_base + i * jump_length;

                load_at_position::<IP, G>(tensor_reader, slice, unit_position, ident, config);
            }
        } else {
            let jump_length = comptime!(total_units * line_size);
            let num_loads_per_unit = div_ceil(num_stage_elements, jump_length);

            for i in 0..num_loads_per_unit {
                let unit_position = unit_position_base + i * jump_length;

                if unit_position < num_stage_elements {
                    load_at_position::<IP, G>(tensor_reader, slice, unit_position, ident, config);
                }
            }
        }
    }
}

#[cube]
fn load_at_position<IP: InputPrecision, G: ConvGemmConfig>(
    tensor_reader: &Im2colReader<IP::Global>,
    slice: &mut SliceMut<Line<IP::Stage>>,
    unit_position: u32,
    #[comptime] ident: MatmulIdent,
    #[comptime] config: G,
) {
    let line_size = config.global_line_size(ident);
    let tile_num_elements = config.tiling_scheme().elements_in_tile(ident);
    let nth_tile = unit_position / tile_num_elements;
    let pos_within_tile = unit_position % tile_num_elements;

    let (tile_x, tile_y) = ContiguousTilingLayout::<RowMajorTilingOrder>::to_x_y::<
        G::StageMemoryConfig,
    >(nth_tile, ident.into_stage(), config.stage_memory_config());

    let line_read = tensor_reader.load_simple::<G>(tile_x, tile_y, pos_within_tile, ident, config);

    slice[unit_position / line_size] = Line::cast_from(line_read);
}
