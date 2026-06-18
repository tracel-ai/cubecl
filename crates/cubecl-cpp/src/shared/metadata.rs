use cubecl_core::ir::{
    dialect::general::{BufferLenOp, ShapeOp, StrideOp},
    prelude::*,
};

use crate::shared::{CompilationState, shared_op_with_out};

shared_op_with_out!(StrideOp, |op, ctx| {
    let buffer_idx = op.buffer_idx(ctx).0;
    let dim = op.dim(ctx).name(ctx);
    let state = ctx.aux_ty::<CompilationState>();
    let dyn_meta = state.dynamic_meta.unwrap().name(ctx);
    let info_st = state.info_st.unwrap().name(ctx);
    let position = state.ext_meta_positions[&buffer_idx];
    let offset = state.info.metadata.stride_offset_index(position);
    format!("{dyn_meta}[{info_st}.static_meta[{offset}] + {dim}]")
});

shared_op_with_out!(ShapeOp, |op, ctx| {
    let buffer_idx = op.buffer_idx(ctx).0;
    let dim = op.dim(ctx).name(ctx);
    let state = ctx.aux_ty::<CompilationState>();
    let dyn_meta = state.dynamic_meta.unwrap().name(ctx);
    let info_st = state.info_st.unwrap().name(ctx);
    let position = state.ext_meta_positions[&buffer_idx];
    let offset = state.info.metadata.shape_offset_index(position);
    format!("{dyn_meta}[{info_st}.static_meta[{offset}] + {dim}]")
});

shared_op_with_out!(BufferLenOp, |op, ctx| {
    let buffer_idx = op.buffer_idx(ctx).0;
    let state = ctx.aux_ty::<CompilationState>();
    let info_st = state.info_st.unwrap().name(ctx);
    let offset = state.info.metadata.buffer_len_index(buffer_idx);
    format!("{info_st}.static_meta[{offset}]")
});
