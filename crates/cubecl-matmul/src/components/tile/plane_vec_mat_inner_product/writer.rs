use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::tile::{
    StridedTile,
    plane_vec_mat_inner_product::{LineContainer, config::PlaneVecMatInnerProductConfig},
};

/// Writer for the output of the VecMat operation.
#[derive(CubeType)]
pub struct MatrixStageWriter {}

#[cube]
impl MatrixStageWriter {
    pub fn store_fragment<A: Numeric, S: Numeric>(
        tile: &mut StridedTile<S, ReadWrite>,
        acc: &Sequence<LineContainer<A>>,
        #[comptime] config: PlaneVecMatInnerProductConfig,
    ) {
        if UNIT_POS_X == 0 {
            let out_line_size = tile.slice.line_size();
            let total_out_lines = comptime![config.n() / out_line_size];
            let mut out_line_iter = comptime![0];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..total_out_lines {
                let mut out_line = Line::<S>::empty(out_line_size);
                let mut within_line = comptime![0];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..out_line_size {
                    let n = comptime!(out_line_iter * out_line_size + within_line);

                    let line_container = acc.index(n);
                    let mut sum = A::from_int(0);
                    for i in 0..config.reduce_line_size() {
                        sum += line_container.line[i];
                    }

                    out_line[within_line] = S::cast_from(sum);
                    comptime![within_line += 1];
                }

                tile.slice[out_line_iter] = out_line;
                comptime![out_line_iter += 1];
            }
        }
    }
}
