use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{dialect::plane::*, interfaces::TypedExt, prelude::*};

use crate::compiler::wgsl::{lower::LowerOp, to_wgsl::wgsl_op_with_out};

wgsl_op_with_out!(ElectOp; |_, _| "subgroupElect()".into());

wgsl_op_with_out!(AllOp; |op, ctx| {
    format!("subgroupAll({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(AnyOp; |op, ctx| {
    format!("subgroupAny({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(ISumOp, FSumOp; |op, ctx| {
    format!("subgroupAdd({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(InclusiveISumOp, InclusiveFSumOp; |op, ctx| {
    format!("subgroupInclusiveAdd({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ExclusiveISumOp, ExclusiveFSumOp; |op, ctx| {
    format!("subgroupExclusiveAdd({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(IProdOp, FProdOp; |op, ctx| {
    format!("subgroupMul({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(InclusiveIProdOp, InclusiveFProdOp; |op, ctx| {
    format!("subgroupInclusiveMul({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ExclusiveIProdOp, ExclusiveFProdOp; |op, ctx| {
    format!("subgroupExclusiveMul({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SMinOp, UMinOp, FMinOp; |op, ctx| {
    format!("subgroupMin({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(SMaxOp, UMaxOp, FMaxOp; |op, ctx| {
    format!("subgroupMax({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(BallotOp; |op, ctx| {
    format!("subgroupBallot({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(BroadcastOp; |op, ctx| {
    let lane = op.lane(ctx).0;
    format!("subgroupBroadcast({}, {lane}u)", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ShuffleOp; |op, ctx| {
    let lane = op.lane(ctx).name(ctx);
    format!("subgroupShuffle({}, {lane})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ShuffleXorOp; |op, ctx| {
    let mask = op.mask(ctx).name(ctx);
    format!("subgroupShuffleXor({}, {mask})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ShuffleUpOp; |op, ctx| {
    let delta = op.delta(ctx).name(ctx);
    format!("subgroupShuffleUp({}, {delta})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ShuffleDownOp; |op, ctx| {
    let delta = op.delta(ctx).name(ctx);
    format!("subgroupShuffleDown({}, {delta})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(UniformLoadOp, AtomicUniformLoadOp; |op, ctx| {
    format!("workgroupUniformLoad({})", op.ptr(ctx).name(ctx))
});

define_scalar!(T);
define_size!(N);

macro_rules! unroll_plane_unop {
    ($ty: ty, $s_ty: ident, $op: expr) => {
        const _: () = {
            #[cube]
            fn unroll(input: Vector<$s_ty, N>) -> Vector<$s_ty, N> {
                let mut out = Vector::default();
                #[unroll]
                for i in 0..input.vector_size() {
                    out.insert(i, $op(input.extract(i)));
                }
                out
            }

            #[op_interface_impl]
            impl LowerOp for $ty {
                fn should_lower(&self, ctx: &Context) -> bool {
                    self.get_operand(ctx).vector_size(ctx) > 1
                }

                fn lower(&self, scope: &cubecl_ir::Scope) -> Vec<Value> {
                    let inp = self.get_operand(scope.ctx());
                    scope.register_value_type::<T, N>(inp);
                    vec![unroll::expand(scope, inp.into()).read_value(scope)]
                }
            }
        };
    };
}

unroll_plane_unop!(AllOp, bool, plane_all);
unroll_plane_unop!(AnyOp, bool, plane_any);
