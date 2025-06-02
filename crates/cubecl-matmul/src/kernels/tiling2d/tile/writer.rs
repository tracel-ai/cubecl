use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use std::marker::PhantomData;

use crate::kernels::tiling2d::{
    base::Dimensions,
    config::CubeTiling2dConfig,
    write_output::{OutputWriter, WriteTileInfo},
};

use super::{
    block_io::base::BlockWriter,
    loader::CheckBounds,
    memory_access::{MatchingVectorization, UnmatchingVectorization},
};

pub(crate) struct TileWriter<N: Numeric> {
    _f: PhantomData<N>,
}

#[cube]
impl<N: Numeric> OutputWriter<N> for TileWriter<N> {
    fn write_output<B: BlockWriter<N>>(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        write_info: WriteTileInfo,
        dims: Dimensions,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let line_size = out.line_size();
        let tile_size = config.tile_size;
        let coordinates = write_info.coordinates;

        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.n,
            skip_row: coordinates.skip_row,
            skip_col: coordinates.skip_col,
        };

        if comptime![line_size == tile_size] {
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
