use cubecl_core as cubecl;
use cubecl_core::prelude::*;

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
    pub offset_output: UInt,
    pub out_stride: UInt,
}

#[cube]
pub(crate) trait OutputWriter<F: Float>: Sync + Send + 'static {
    fn write_output<B: BlockWriter<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        write_tile_info: WriteTileInfo,
        dims: Dimensions,
        config: Comptime<CubeTiling2dConfig>,
    );
}

#[cube]
pub(crate) fn write_to_output<F: Float, W: OutputWriter<F>>(
    out: &mut Tensor<F>,
    results: &Array<F>,
    coordinates: Coordinates,
    offset_output: UInt,
    dims: Dimensions,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    let write_info = WriteTileInfo {
        coordinates,
        offset_output,
        out_stride: dims.n,
    };

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
            W::write_output::<WholeCheckBlockIO>(out, results, write_info, dims, config);
        } else {
            W::write_output::<VerticalCheckBlockIO>(out, results, write_info, dims, config);
        }
    } else if Comptime::get(check_n_bounds) {
        W::write_output::<HorizontalCheckBlockIO>(out, results, write_info, dims, config);
    } else {
        W::write_output::<UncheckedBlockIO>(out, results, write_info, dims, config);
    }
}
