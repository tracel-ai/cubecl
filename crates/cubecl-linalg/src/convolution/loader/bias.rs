use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor};

use crate::{
    convolution::homogeneous::simple::ConvTilingLayout,
    matmul::components::{
        Ident,
        global::{AccumulatorLoader, GlobalConfig},
        stage::{Skew, Stage, StageConfig},
        tile::{Tile, TileConfig, TileMatmul},
    },
};
use crate::{convolution::reader::bias::BiasReader, matmul::components::MatmulPrecision};

/// Special loader to broadcast the 1D bias to the 2D accumulator matrix
#[derive(CubeType)]
pub enum BiasLoader<MP: MatmulPrecision> {
    Some {
        tensor_view: BiasReader<MP::EO>,
        stage: Stage<MP::EA, ConvTilingLayout>,
    },
    None,
}

#[cube]
impl<MP: MatmulPrecision> AccumulatorLoader<MP> for BiasLoader<MP> {
    fn fill_stage<G: GlobalConfig>(this: &mut Self, #[comptime] config: G) {
        match this {
            BiasLoader::Some { tensor_view, stage } => {
                let stage_tiling = config.tiling_dimensions(Ident::Rhs);
                let line_size = config.global_line_size(Ident::Out);

                let num_stage_elements = stage_tiling.total_col();

                let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
                let unit_position_base = unit_id * line_size;

                let mut slice = stage.as_slice_mut(line_size);

                if unit_position_base < num_stage_elements {
                    let read_line = tensor_view.load_simple::<G>(unit_position_base, line_size);
                    slice[unit_id] = Line::cast_from(read_line);
                }
            }
            BiasLoader::None => {}
        }
    }

    /// Load accumulator
    fn load<TMM: TileMatmul<MP>>(
        this: &mut Self,
        acc: &mut TMM::Accumulator,
        tile_n: u32,
        #[comptime] config: TMM::Config,
    ) {
        match this {
            BiasLoader::Some { stage, .. } => {
                let line_size = config.stage_line_size(Ident::Out);
                let tile_elems = config.tile_shape().n / line_size;
                let start = tile_n * tile_elems;
                let slice = stage
                    .as_slice_mut(line_size)
                    .slice(start, start + tile_elems);
                let tile =
                    Tile::new_strided::<TMM::Config>(slice, 0u32, Skew::None, Ident::Out, config);
                TMM::fill_accumulator(&tile, acc, config);
            }
            BiasLoader::None => {
                TMM::zero_accumulator(acc, config);
            }
        }
    }
}

#[cube]
impl<MP: MatmulPrecision> BiasLoader<MP> {
    pub fn new<G: GlobalConfig>(
        tensor: CubeOption<VirtualTensor<MP::EO>>,
        n_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        match tensor {
            CubeOption::Some(tensor) => {
                let stage = init_stage::<MP::EA, G>(config);
                let shape_n = tensor.shape(0);
                let tensor_view = BiasReader::<MP::EO>::new(tensor, n_offset, shape_n);

                BiasLoader::<MP>::new_Some(tensor_view, stage)
            }
            CubeOption::None => BiasLoader::new_None(),
        }
    }
}

#[cube]
fn init_stage<ES: Numeric, G: GlobalConfig>(#[comptime] config: G) -> Stage<ES, ConvTilingLayout> {
    let line_size = config.to_smm_config().stage_line_size(Ident::Out);

    let smem = SharedMemory::new_lined(
        comptime!(config.tiling_dimensions(Ident::Out).total_col() / line_size),
        line_size,
    );

    Stage::<ES, ConvTilingLayout>::new_with_smem(smem, Skew::None)
}

#[cube]
fn init_empty_stage<ES: Numeric>() -> Stage<ES, ConvTilingLayout> {
    Stage::<ES, ConvTilingLayout>::new_with_smem(SharedMemory::new(1), Skew::None)
}
