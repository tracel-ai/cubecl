use cubecl_core::{
    frontend::cast_value,
    ir::{
        ContextExt,
        dialect::plane::*,
        types::scalar::{BoolType, UIntType},
    },
};
use pliron::{context::Context, derive::op_interface_impl, value::Value};

use crate::{
    cuda::{cuda_op_with_out, ptx::InlinePtxOp},
    ptx_block,
    shared::{CompilationOptions, lowering::LowerOp},
    target::Cuda,
};

cuda_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("__shfl_sync(__activemask(), {val}, {lane})")
});

cuda_op_with_out!(ShuffleOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).name(ctx);
    format!("__shfl_sync(__activemask(), {val}, {lane})")
});

cuda_op_with_out!(ShuffleXorOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let mask = op.mask(ctx).name(ctx);
    format!("__shfl_xor_sync(__activemask(), {val}, {mask})")
});

cuda_op_with_out!(ShuffleUpOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_up_sync(__activemask(), {val}, {delta})")
});

cuda_op_with_out!(ShuffleDownOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_down_sync(__activemask(), {val}, {delta})")
});

cuda_op_with_out!(AllOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("__all_sync(__activemask(), {val})")
});

cuda_op_with_out!(AnyOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("__any_sync(__activemask(), {val})")
});

cuda_op_with_out!(BallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("{{__ballot_sync(__activemask(), {val}), 0, 0, 0}}")
});

#[op_interface_impl]
impl LowerOp<Cuda> for ElectOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        let opts = ctx.aux_ty::<CompilationOptions>();
        opts.supports_features.elect_sync
    }

    fn lower(&self, scope: &cubecl_core::ir::Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let u32 = UIntType::get(ctx, 32).to_handle();
        let ptx = ptx_block! {
            ".reg .pred %%px;"
            "elect.sync _|%%px, 0xffffffff;"
            "selp.b32 $0, 1, 0, %%px;"
        };
        let op = InlinePtxOp::new_volatile(ctx, Some(u32), ptx, vec![]);
        scope.register(&op);
        let cast = cast_value(scope, op.result(ctx).unwrap(), BoolType::get(ctx).into());
        vec![cast]
    }
}
