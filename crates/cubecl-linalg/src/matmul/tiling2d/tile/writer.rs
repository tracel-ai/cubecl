use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use std::marker::PhantomData;

use crate::matmul::tiling2d::{
    base::Dimensions,
    config::CubeTiling2dConfig,
    write_output::{OutputWriter, WriteTileInfo},
};

use super::{
    block_io::base::BlockWriter,
    loader::CheckBounds,
    memory_access::{MatchingVectorization, UnmatchingVectorization},
};

pub(crate) struct TileWriter<F: Float> {
    _f: PhantomData<F>,
}

#[cube]
impl<F: Float> OutputWriter<F> for TileWriter<F> {
    fn write_output<B: BlockWriter<F>>(
        out: &mut Tensor<Line<F>>,
        results: &Array<F>,
        write_info: WriteTileInfo,
        dims: Dimensions,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let vectorization = vectorization_of(out);
        let tile_size = config.tile_size;
        let coordinates = write_info.coordinates;

        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.n,
            skip_row: coordinates.skip_row,
            skip_col: coordinates.skip_col,
        };

        if vectorization == tile_size {
            B::write_output::<MatchingVectorization>(
                out,
                results,
                write_info,
                config,
                check_bounds,
            );
        } else {
            B::write_output::<UnmatchingVectorization>(
                out,
                results,
                write_info,
                config,
                check_bounds,
            );
        }
    }
}
