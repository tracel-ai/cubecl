use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Accumulators, Dimensions, Offsets},
    block_io::{
        base::{BlockWriter, BlockWriterExpand},
        horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO,
        vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::CmmaConfig,
};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: &mut Tensor<F>,
    accumulators: Accumulators<F>,
    offsets: Offsets,
    dims: Dimensions,
    #[comptime] config: CmmaConfig,
) {
    let accumulator_sm = fragment_to_shared_memory(accumulators);
    shared_memory_to_output(out, offsets, accumulator_sm, dims, config);
}

#[cube]
fn fragment_to_shared_memory<F: Float>(accumulators: Accumulators<F>) -> SharedMemory<F> {
    let mut acc_sm = SharedMemory::<F>::new(4096);

    let coop_id = UNIT_POS_Y;
    let slice_offset_0 = coop_id * 512;
    let slice_offset_1 = slice_offset_0 + 256;
    let slice_offset_2 = slice_offset_1 + 256;

    let slice = &mut acc_sm[slice_offset_0..slice_offset_1];
    cmma::store(slice, &accumulators.first, 16, cmma::MatrixLayout::RowMajor);

    let slice = &mut acc_sm[slice_offset_1..slice_offset_2];
    cmma::store(
        slice,
        &accumulators.second,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    acc_sm
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    #[comptime] config: CmmaConfig,
) {
    let check_m_bounds = config.check_m_bounds;
    let check_n_bounds = config.check_n_bounds;

    if check_m_bounds {
        if check_n_bounds {
            write_tile::<F, WholeCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        } else {
            write_tile::<F, VerticalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        }
    } else if check_n_bounds {
        write_tile::<F, HorizontalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
    } else {
        write_tile::<F, UncheckedBlockIO>(out, offsets, accumulator_sm, dims, config);
    }
}

// #[cube]
// fn write_tile<F: Float, W: BlockWriter<F>>(
//     out: &mut Tensor<F>,
//     offsets: Offsets,
//     accumulator_sm: SharedMemory<F>,
//     dims: Dimensions,
//     #[comptime] config: CmmaConfig,
// ) {
//     // Other values not supported
//     let n_tiles = 2;

//     let tile_size = config.tile_size;
//     let out_vec = vectorization(out);
//     let n_units_per_tile_row = tile_size / out_vec;
//     let num_tile_elems = tile_size * tile_size;

//     let coop_dim = 32;
//     let coop_id = UNIT_POS_Y;
//     let lane_id = UNIT_POS_X;

//     let tile_row = coop_id / n_tiles;
//     let tile_col = (coop_id % n_tiles) * n_tiles;

//     let read_offset = n_tiles * coop_id * num_tile_elems;
//     let read_0 = read_offset + lane_id * out_vec;
//     let read_1 = read_0 + coop_dim * out_vec;

//     let unit_write_row_0 = lane_id / n_units_per_tile_row;
//     let unit_write_row_1 = unit_write_row_0 + coop_dim / out_vec;
//     let unit_write_col = (lane_id % n_units_per_tile_row) * n_units_per_tile_row;

//     let row_offset = offsets.cube_row + tile_row * tile_size;
//     let write_row_0 = row_offset + unit_write_row_0;
//     let write_row_1 = row_offset + unit_write_row_1;
//     let write_col = offsets.cube_col + tile_col * tile_size + unit_write_col;

//     W::write_output(
//         out,
//         accumulator_sm,
//         0,
//         offsets.batch_out,
//         read_0,
//         write_row_0,
//         write_col,
//         dims,
//         config,
//     );
//     W::write_output(
//         out,
//         accumulator_sm,
//         0,
//         offsets.batch_out,
//         read_1,
//         write_row_1,
//         write_col,
//         dims,
//         config,
//     );
//     W::write_output(
//         out,
//         accumulator_sm,
//         1,
//         offsets.batch_out,
//         read_0,
//         write_row_0,
//         write_col,
//         dims,
//         config,
//     );
//     W::write_output(
//         out,
//         accumulator_sm,
//         1,
//         offsets.batch_out,
//         read_1,
//         write_row_1,
//         write_col,
//         dims,
//         config,
//     );
// }

// Recursive expansion of cube macro
// ==================================

#[allow(dead_code)]
fn write_tile<F: Float, W: BlockWriter<F>>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    config: CmmaConfig,
) {
    let n_tiles = 2;
    let tile_size = config.tile_size;
    let out_vec = vectorization(out);
    let n_units_per_tile_row = tile_size / out_vec;
    let num_tile_elems = tile_size * tile_size;
    let coop_dim = 32;
    let coop_id = UNIT_POS_Y;
    let lane_id = UNIT_POS_X;
    let tile_row = coop_id / n_tiles;
    let tile_col = (coop_id % n_tiles) * n_tiles;
    let read_offset = n_tiles * coop_id * num_tile_elems;
    let read_0 = read_offset + lane_id * out_vec;
    let read_1 = read_0 + coop_dim * out_vec;
    let unit_write_row_0 = lane_id / n_units_per_tile_row;
    let unit_write_row_1 = unit_write_row_0 + coop_dim / out_vec;
    let unit_write_col = (lane_id % n_units_per_tile_row) * n_units_per_tile_row;
    let row_offset = offsets.cube_row + tile_row * tile_size;
    let write_row_0 = row_offset + unit_write_row_0;
    let write_row_1 = row_offset + unit_write_row_1;
    let write_col = offsets.cube_col + tile_col * tile_size + unit_write_col;
    W::write_output(
        out,
        accumulator_sm,
        0,
        offsets.batch_out,
        read_0,
        write_row_0,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        0,
        offsets.batch_out,
        read_1,
        write_row_1,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        1,
        offsets.batch_out,
        read_0,
        write_row_0,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        1,
        offsets.batch_out,
        read_1,
        write_row_1,
        write_col,
        dims,
        config,
    );
}
mod write_tile {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<F: Float, W: BlockWriter<F>>(
        out: impl cubecl::new_ir::Expr<Output = Tensor<F>> + 'static + Clone,
        offsets: impl cubecl::new_ir::Expr<Output = Offsets> + 'static + Clone,
        accumulator_sm: impl cubecl::new_ir::Expr<Output = SharedMemory<F>> + 'static + Clone,
        dims: impl cubecl::new_ir::Expr<Output = Dimensions> + 'static + Clone,
        config: CmmaConfig,
    ) -> impl cubecl::new_ir::Expr<Output = ()> {
        use cubecl::new_ir::{ExpandExpr as _, PartialExpand as _};
        {
            {
                let mut __statements = Vec::new();
                let n_tiles = 2;
                let tile_size = config.tile_size;
                let __init = vectorization::expand(cubecl::new_ir::OnceExpr::new(out.clone()));
                let out_vec = cubecl::new_ir::Variable::new(
                    "out_vec",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: out_vec,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::DivExpr::new(tile_size, out_vec.clone());
                let n_units_per_tile_row = cubecl::new_ir::Variable::new(
                    "n_units_per_tile_row",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: n_units_per_tile_row,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let num_tile_elems = tile_size * tile_size;
                let coop_dim = 32;
                let coop_id = UNIT_POS_Y;
                let lane_id = UNIT_POS_X;
                let tile_row = coop_id / n_tiles;
                let tile_col = (coop_id % n_tiles) * n_tiles;
                let read_offset = n_tiles * coop_id * num_tile_elems;
                let __init = cubecl::new_ir::AddExpr::new(
                    read_offset,
                    cubecl::new_ir::MulExpr::new(lane_id, out_vec.clone()),
                );
                let read_0 = cubecl::new_ir::Variable::new(
                    "read_0",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: read_0,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::AddExpr::new(
                    read_0.clone(),
                    cubecl::new_ir::MulExpr::new(coop_dim, out_vec.clone()),
                );
                let read_1 = cubecl::new_ir::Variable::new(
                    "read_1",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: read_1,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::DivExpr::new(lane_id, n_units_per_tile_row.clone());
                let unit_write_row_0 = cubecl::new_ir::Variable::new(
                    "unit_write_row_0",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: unit_write_row_0,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::AddExpr::new(
                    unit_write_row_0.clone(),
                    cubecl::new_ir::DivExpr::new(coop_dim, out_vec.clone()),
                );
                let unit_write_row_1 = cubecl::new_ir::Variable::new(
                    "unit_write_row_1",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: unit_write_row_1,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::MulExpr::new(
                    cubecl::new_ir::RemExpr::new(lane_id, n_units_per_tile_row.clone()),
                    n_units_per_tile_row.clone(),
                );
                let unit_write_col = cubecl::new_ir::Variable::new(
                    "unit_write_col",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: unit_write_col,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::AddExpr::new(
                    offsets.clone().expand().__cube_row(),
                    tile_row * tile_size,
                );
                let row_offset = cubecl::new_ir::Variable::new(
                    "row_offset",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: row_offset,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init =
                    cubecl::new_ir::AddExpr::new(row_offset.clone(), unit_write_row_0.clone());
                let write_row_0 = cubecl::new_ir::Variable::new(
                    "write_row_0",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: write_row_0,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init =
                    cubecl::new_ir::AddExpr::new(row_offset.clone(), unit_write_row_1.clone());
                let write_row_1 = cubecl::new_ir::Variable::new(
                    "write_row_1",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: write_row_1,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                let __init = cubecl::new_ir::AddExpr::new(
                    cubecl::new_ir::AddExpr::new(
                        offsets.clone().expand().__cube_col(),
                        tile_col * tile_size,
                    ),
                    unit_write_col.clone(),
                );
                let write_col = cubecl::new_ir::Variable::new(
                    "write_col",
                    cubecl::new_ir::Expr::vectorization(&__init),
                );
                __statements.push({
                    cubecl::new_ir::Statement::Local {
                        variable: cubecl::new_ir::Expr::expression_untyped(
                            &(cubecl::new_ir::Initializer {
                                left: write_col,
                                right: __init,
                            }),
                        ),
                        mutable: false,
                        ty: None,
                    }
                });
                __statements.push(cubecl::new_ir::Statement::Expression(
                    cubecl::new_ir::Expr::expression_untyped(
                        &(<W as cubecl::new_ir::StaticExpand>::Expanded::write_output(
                            cubecl::new_ir::OnceExpr::new(out.clone()),
                            cubecl::new_ir::OnceExpr::new(accumulator_sm.clone()),
                            0,
                            cubecl::new_ir::OnceExpr::new(offsets.clone().expand().__batch_out()),
                            cubecl::new_ir::OnceExpr::new(read_0.clone()),
                            cubecl::new_ir::OnceExpr::new(write_row_0.clone()),
                            cubecl::new_ir::OnceExpr::new(write_col.clone()),
                            cubecl::new_ir::OnceExpr::new(dims.clone()),
                            config,
                        )),
                    ),
                ));
                __statements.push(cubecl::new_ir::Statement::Expression(
                    cubecl::new_ir::Expr::expression_untyped(
                        &(<W as cubecl::new_ir::StaticExpand>::Expanded::write_output(
                            cubecl::new_ir::OnceExpr::new(out.clone()),
                            cubecl::new_ir::OnceExpr::new(accumulator_sm.clone()),
                            0,
                            cubecl::new_ir::OnceExpr::new(offsets.clone().expand().__batch_out()),
                            cubecl::new_ir::OnceExpr::new(read_1.clone()),
                            cubecl::new_ir::OnceExpr::new(write_row_1.clone()),
                            cubecl::new_ir::OnceExpr::new(write_col.clone()),
                            cubecl::new_ir::OnceExpr::new(dims.clone()),
                            config,
                        )),
                    ),
                ));
                __statements.push(cubecl::new_ir::Statement::Expression(
                    cubecl::new_ir::Expr::expression_untyped(
                        &(<W as cubecl::new_ir::StaticExpand>::Expanded::write_output(
                            cubecl::new_ir::OnceExpr::new(out.clone()),
                            cubecl::new_ir::OnceExpr::new(accumulator_sm.clone()),
                            1,
                            cubecl::new_ir::OnceExpr::new(offsets.clone().expand().__batch_out()),
                            cubecl::new_ir::OnceExpr::new(read_0.clone()),
                            cubecl::new_ir::OnceExpr::new(write_row_0.clone()),
                            cubecl::new_ir::OnceExpr::new(write_col.clone()),
                            cubecl::new_ir::OnceExpr::new(dims.clone()),
                            config,
                        )),
                    ),
                ));
                __statements.push(cubecl::new_ir::Statement::Expression(
                    cubecl::new_ir::Expr::expression_untyped(
                        &(<W as cubecl::new_ir::StaticExpand>::Expanded::write_output(
                            cubecl::new_ir::OnceExpr::new(out.clone()),
                            cubecl::new_ir::OnceExpr::new(accumulator_sm.clone()),
                            1,
                            cubecl::new_ir::OnceExpr::new(offsets.clone().expand().__batch_out()),
                            cubecl::new_ir::OnceExpr::new(read_1.clone()),
                            cubecl::new_ir::OnceExpr::new(write_row_1.clone()),
                            cubecl::new_ir::OnceExpr::new(write_col.clone()),
                            cubecl::new_ir::OnceExpr::new(dims.clone()),
                            config,
                        )),
                    ),
                ));
                cubecl::new_ir::BlockExpr::new(__statements, ())
            }
        }
    }
}
