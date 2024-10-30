use cubecl_core::{self as cubecl};
use cubecl_core::{prelude::*, CubeType};

use super::{
    base::{Coordinates, Dimensions},
    config::CubeTiling2dConfig,
    tile::block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
};

#[derive(CubeType)]
pub(crate) struct WriteTileInfo {
    pub coordinates: Coordinates,
    pub offset_output: u32,
    pub out_stride: u32,
}

#[cube]
pub(crate) trait OutputWriter<F: Float>: Sync + Send + 'static {
    fn write_output<B: BlockWriter<F>>(
        out: &mut Tensor<Line<F>>,
        results: &Array<F>,
        write_tile_info: WriteTileInfo,
        dims: Dimensions,
        #[comptime] config: CubeTiling2dConfig,
    );
}

#[cube]
pub(crate) fn write_to_output<F: Float, W: OutputWriter<F>>(
    out: &mut Tensor<Line<F>>,
    results: &Array<F>,
    coordinates: Coordinates,
    offset_output: u32,
    dims: Dimensions,
    #[comptime] config: CubeTiling2dConfig,
) {
    let check_m_bounds = config.check_m_bounds;
    let check_n_bounds = config.check_n_bounds;

    let write_info = WriteTileInfo {
        coordinates,
        offset_output,
        out_stride: dims.n,
    };

    if check_m_bounds {
        if check_n_bounds {
            W::write_output::<WholeCheckBlockIO>(out, results, write_info, dims, config);
        } else {
            W::write_output::<VerticalCheckBlockIO>(out, results, write_info, dims, config);
        }
    } else if check_n_bounds {
        W::write_output::<HorizontalCheckBlockIO>(out, results, write_info, dims, config);
    } else {
        W::write_output::<UncheckedBlockIO>(out, results, write_info, dims, config);
    }
}
