use std::collections::{HashMap, HashSet};

use cubecl_core::{
    frontend::HasValue,
    ir::{
        Builtin, Scope,
        attributes::{FuncInterface, IndexAttr},
        dialect::general::ReadBuiltinOp,
        ident,
        prelude::*,
    },
};
use pliron::{
    builtin::{
        ops::FuncOp,
        types::{IntegerType, Signedness},
    },
    debug_info::set_block_arg_name,
    irbuild::{listener::DummyListener, match_rewrite::MatchRewrite},
    value::Value,
};

use crate::{
    metal::{BuiltInAttr, metal_op_with_out, signature::ATTR_BUILTIN_ATTRIBUTE},
    shared::{
        CompilationState,
        builtin::{LowerBuiltins, absolute_pos, constant, cube_count, cube_pos},
    },
    target::Metal,
};

#[cube_op(name = "msl.read_dim3_builtin")]
#[result_ty(argument)]
pub struct ReadDim3BuiltinOp {
    pub builtin: Value,
    pub dim: IndexAttr,
}

metal_op_with_out!(ReadDim3BuiltinOp, |op, ctx| {
    format!("{}[{}]", op.builtin(ctx).name(ctx), op.dim(ctx).0)
});

impl MatchRewrite for LowerBuiltins<Metal> {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op.as_op::<ReadBuiltinOp>(ctx).unwrap().builtin(ctx).0;
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        if let Some(new_value) = builtin.maybe_lower_metal(&scope) {
            rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        }
        Ok(())
    }
}

trait MetalBuiltin {
    fn maybe_lower_metal(&self, scope: &Scope) -> Option<Value>;
}

impl MetalBuiltin for Builtin {
    fn maybe_lower_metal(&self, scope: &Scope) -> Option<Value> {
        let cube_dim = scope.ctx().aux_ty::<CompilationState>().cube_dim;
        match self {
            Builtin::UnitPos => None,
            // This is common enough to be worth replacing. Z is almost always 1, and Y is often 1.
            // Replacing it with a constant allows simplifying the positional math
            Builtin::UnitPosX if cube_dim.x == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosY if cube_dim.y == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosZ if cube_dim.z == 1 => Some(constant::expand(scope, 0).value(scope)),
            Builtin::UnitPosX | Builtin::UnitPosY | Builtin::UnitPosZ => None,
            Builtin::CubePosCluster => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterX => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterY => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePosClusterZ => Some(constant::expand(scope, 0).value(scope)),
            Builtin::CubePos => Some(cube_pos::expand(scope).value(scope)),
            Builtin::CubePosX | Builtin::CubePosY | Builtin::CubePosZ => None,
            Builtin::CubeDim => Some(constant::expand(scope, cube_dim.num_elems()).value(scope)),
            Builtin::CubeDimX => Some(constant::expand(scope, cube_dim.x).value(scope)),
            Builtin::CubeDimY => Some(constant::expand(scope, cube_dim.y).value(scope)),
            Builtin::CubeDimZ => Some(constant::expand(scope, cube_dim.z).value(scope)),
            Builtin::CubeClusterDim => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimX => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimY => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeClusterDimZ => Some(constant::expand(scope, 1).value(scope)),
            Builtin::CubeCount => Some(cube_count::expand(scope).value(scope)),
            Builtin::CubeCountX | Builtin::CubeCountY | Builtin::CubeCountZ => None,
            Builtin::PlaneDim => None,
            Builtin::PlanePos => None,
            Builtin::UnitPosPlane => None,
            Builtin::AbsolutePos => Some(absolute_pos::expand(scope).value(scope)),
            Builtin::AbsolutePosX | Builtin::AbsolutePosY | Builtin::AbsolutePosZ => None,
        }
    }
}

pub fn append_msl_builtins(ctx: &mut Context, entry_func: FuncOp) {
    let op = entry_func.get_operation();
    let mut used = HashSet::new();
    let mut read_ops = HashSet::new();
    let state = &mut (&mut used, &mut read_ops);
    visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, state, op, |ctx, (used, ops), op| {
        ops.insert(op);
        used.insert(built_in_attr(op.builtin(ctx).0));
    });

    let values = used
        .into_iter()
        .map(|attr| {
            let entry = entry_func.get_entry_block(ctx);
            let i = entry_func.push_argument(ctx, attr.ty(ctx));
            entry_func.set_arg_attr(ctx, i, &ATTR_BUILTIN_ATTRIBUTE, Box::new(attr));
            set_block_arg_name(ctx, entry, i, Some(ident(attr.to_string())));
            let value = entry.deref(ctx).get_argument(i);
            (attr, value)
        })
        .collect::<HashMap<_, _>>();

    let mut rewriter = IRRewriter::<DummyListener>::default();

    for op in read_ops {
        rewriter.set_insertion_point_before_operation(op.get_operation());

        let builtin = op.builtin(ctx).0;
        let attr = built_in_attr(builtin);
        let value = values[&attr];
        match builtin {
            Builtin::UnitPos | Builtin::PlaneDim | Builtin::PlanePos | Builtin::UnitPosPlane => {
                rewriter.replace_operation_with_values(ctx, op.get_operation(), vec![value]);
            }
            Builtin::UnitPosX | Builtin::CubePosX | Builtin::CubeCountX | Builtin::AbsolutePosX => {
                read_dim3(ctx, &mut rewriter, op, value, 0);
            }
            Builtin::UnitPosY | Builtin::CubePosY | Builtin::CubeCountY | Builtin::AbsolutePosY => {
                read_dim3(ctx, &mut rewriter, op, value, 1);
            }
            Builtin::UnitPosZ | Builtin::CubePosZ | Builtin::CubeCountZ | Builtin::AbsolutePosZ => {
                read_dim3(ctx, &mut rewriter, op, value, 2);
            }
            other => unreachable!("{other:?} should be lowered"),
        }
    }
}

fn read_dim3(
    ctx: &mut Context,
    rewriter: &mut impl Rewriter,
    op: ReadBuiltinOp,
    value: Value,
    dim: usize,
) {
    let u32 = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
    let new_op = ReadDim3BuiltinOp::new(ctx, u32, value, dim);
    rewriter.append_op(ctx, &new_op);
    rewriter.replace_operation(ctx, op.get_operation(), new_op.get_operation());
}

fn built_in_attr(builtin: Builtin) -> BuiltInAttr {
    match builtin {
        Builtin::UnitPos => BuiltInAttr::ThreadIndexInThreadgroup,
        Builtin::UnitPosX | Builtin::UnitPosY | Builtin::UnitPosZ => {
            BuiltInAttr::ThreadPositionInThreadgroup
        }
        Builtin::CubePosX | Builtin::CubePosY | Builtin::CubePosZ => {
            BuiltInAttr::ThreadgroupPositionInGrid
        }
        Builtin::CubeCountX | Builtin::CubeCountY | Builtin::CubeCountZ => {
            BuiltInAttr::ThreadgroupsPerGrid
        }
        Builtin::PlaneDim => BuiltInAttr::ThreadsPerSIMDgroup,
        Builtin::PlanePos => BuiltInAttr::SIMDgroupIndexInThreadgroup,
        Builtin::UnitPosPlane => BuiltInAttr::ThreadIndexInSIMDgroup,
        Builtin::AbsolutePosX | Builtin::AbsolutePosY | Builtin::AbsolutePosZ => {
            BuiltInAttr::ThreadPositionInGrid
        }
        other => unreachable!("{other:?} should be lowered"),
    }
}
