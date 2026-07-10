use cubecl_core::ir::{
    dialect::math::{SaturatingSAddOp, SaturatingSSubOp},
    interfaces::TypedExt,
    prelude::*,
};

use crate::cuda::ptx_with_out;

ptx_with_out!(
    SaturatingSAddOp,
    |_, _| "add.sat.s32 $0, $1, $2;".into(),
    |op, ctx| op.get_type(ctx).is_int_of_width(ctx, 32)
);
ptx_with_out!(
    SaturatingSSubOp,
    |_, _| "sub.sat.s32 $0, $1, $2;".into(),
    |op, ctx| op.get_type(ctx).is_int_of_width(ctx, 32)
);
