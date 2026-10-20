use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    Builtin, ContextExt,
    dialect::general::ReadBuiltinOp,
    ident,
    prelude::{
        AnalysisManager, Context, MatchRewrite, MatchRewriter, OneResultInterface, Operation,
        OperationPtrExt, PassResult, Ptr, Result, Rewriter, SingleBlockRegionInterface,
        op_interface_impl, pass_name,
    },
    rewrite::visit_all_ops_of_type_mut,
};
use cubecl_opt::passes::uniformity::{DEVICE_UNIFORM, UniformAttr, op_dyn_uniformity};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::{ConstantOp, FuncOp, ModuleOp},
        types::{IntegerType, Signedness},
    },
    identifier::Identifier,
    irbuild::match_rewrite::{RewriterOrder, apply_match_rewrite},
    op::Op,
    pass::Pass,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};
use pliron_spirv::{
    attrs::BuiltInAttr,
    decorations::DecoratableOp,
    ops::{AddressOfOp, GlobalVariableOp, InBoundsAccessChainOp, LoadOp},
    types::{PointerType, VectorType},
};
use rspirv::spirv::{BuiltIn, Capability, MemoryAccess, StorageClass};

use crate::{CustomCapabilitiesOp, KernelInfo, decorate_uniform};

/// The name of the `Input` variable that holds `builtin`.
fn builtin_var_name(builtin: BuiltIn) -> Identifier {
    ident(format!("_spirv_builtin_{builtin:?}"))
}

/// Gives each builtin that a kernel reads its own `Input` variable with a `BuiltIn` decoration.
/// Vulkan does not allow compute builtins as members of an `Input` block: spirv-val rejects
/// `BuiltIn SubgroupLocalInvocationId cannot be used as a member decoration`.
pub struct LowerBuiltinsPass;

#[pass_name]
impl Pass for LowerBuiltinsPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let module = op.as_op::<ModuleOp>(ctx).expect("Should run on module");
        let module = module.get_body(ctx, 0);

        let mut funcs = vec![];
        visit_all_ops_of_type_mut::<FuncOp, _>(ctx, &mut funcs, op, |_, funcs, func| {
            funcs.push(func);
        });

        let mut rewriter = RewriteBuiltins { used: vec![] };
        for func in funcs {
            let op = func.get_operation();
            res.ir_changed |=
                apply_match_rewrite(ctx, &mut rewriter, RewriterOrder::default(), op)?;
        }

        for builtin in rewriter.used.into_iter().rev() {
            let ty = builtin_ty(ctx, builtin);
            let var = GlobalVariableOp::new(
                ctx,
                ty,
                StorageClass::Input,
                builtin_var_name(builtin),
                None,
            );
            var.set_decoration_built_in(ctx, BuiltInAttr::new(builtin));
            var.get_operation().insert_at_front(module, ctx);
        }

        Ok(res)
    }
}

// The capability collection reads ops and types, not decorations, so the variable declares the
// capability its builtin needs.
#[op_interface_impl]
impl CustomCapabilitiesOp for GlobalVariableOp {
    fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
        match self.get_decoration_built_in(ctx).map(|builtin| builtin.0) {
            Some(
                BuiltIn::SubgroupSize | BuiltIn::SubgroupId | BuiltIn::SubgroupLocalInvocationId,
            ) => vec![Capability::GroupNonUniform],
            _ => vec![],
        }
    }
}

fn builtin_ty(ctx: &Context, builtin: BuiltIn) -> TypeHandle {
    let u32 = IntegerType::get(ctx, 32, Signedness::Signless).to_handle();
    let dim3 = VectorType::get(ctx, 3, u32).to_handle();
    match builtin {
        BuiltIn::NumWorkgroups => dim3,
        BuiltIn::WorkgroupId => dim3,
        BuiltIn::LocalInvocationId => dim3,
        BuiltIn::GlobalInvocationId => dim3,
        BuiltIn::LocalInvocationIndex => u32,
        BuiltIn::SubgroupSize => u32,
        BuiltIn::SubgroupId => u32,
        BuiltIn::SubgroupLocalInvocationId => u32,
        _ => unreachable!("Unsupported builtin"),
    }
}

struct RewriteBuiltins {
    /// The builtins read so far, in the order of the first read.
    used: Vec<BuiltIn>,
}

impl RewriteBuiltins {
    /// A pointer to the variable of `builtin`.
    fn builtin_ptr(&mut self, scope: &Scope, builtin: BuiltIn) -> Value {
        if !self.used.contains(&builtin) {
            self.used.push(builtin);
        }
        let ctx = scope.ctx_mut();
        let ptr_ty = PointerType::get(ctx, builtin_ty(ctx, builtin), StorageClass::Input).into();
        scope.register_with_result(&AddressOfOp::new(ctx, ptr_ty, builtin_var_name(builtin)))
    }
}

impl MatchRewrite for RewriteBuiltins {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<ReadBuiltinOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let builtin = op.as_op::<ReadBuiltinOp>(ctx).unwrap();
        let uniformity = op_dyn_uniformity(ctx, op);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let ty = builtin.get_result(ctx).get_type(ctx);
        let builtin = builtin.builtin(ctx).0;
        let value = self.lower_builtin(&scope, ty, builtin, uniformity);
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

#[cube]
fn absolute_pos() -> usize {
    cube_pos() * CUBE_DIM as usize + UNIT_POS as usize
}

#[cube]
fn cube_count() -> usize {
    CUBE_COUNT_X as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_Z as usize
}

#[cube]
fn cube_pos() -> usize {
    CUBE_POS_Z as usize * CUBE_COUNT_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_Y as usize * CUBE_COUNT_X as usize
        + CUBE_POS_X as usize
}

#[cube]
fn constant(#[comptime] value: u32) -> u32 {
    value
}

impl RewriteBuiltins {
    fn lower_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        builtin: Builtin,
        uniformity: Option<UniformAttr>,
    ) -> Value {
        let cube_dim = scope.ctx().aux_ty::<KernelInfo>().cube_dim;
        match builtin {
            Builtin::UnitPos if cube_dim.num_elems() == 1 => {
                constant::expand(scope, 0).value(scope)
            }
            Builtin::UnitPosX if cube_dim.x == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosY if cube_dim.y == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosZ if cube_dim.z == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPos => {
                self.read_scalar_builtin(scope, ty, BuiltIn::LocalInvocationIndex, uniformity)
            }
            Builtin::UnitPosX => {
                self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 0, uniformity)
            }
            Builtin::UnitPosY => {
                self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 1, uniformity)
            }
            Builtin::UnitPosZ => {
                self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 2, uniformity)
            }
            Builtin::CubePosCluster => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterX => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterY => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterZ => constant::expand(scope, 0).value(scope),
            Builtin::CubePos => cube_pos::expand(scope).value(scope),
            Builtin::CubePosX => {
                self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 0, uniformity)
            }
            Builtin::CubePosY => {
                self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 1, uniformity)
            }
            Builtin::CubePosZ => {
                self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 2, uniformity)
            }
            Builtin::CubeDim => constant::expand(scope, cube_dim.num_elems()).value(scope),
            Builtin::CubeDimX => constant::expand(scope, cube_dim.x).value(scope),
            Builtin::CubeDimY => constant::expand(scope, cube_dim.y).value(scope),
            Builtin::CubeDimZ => constant::expand(scope, cube_dim.z).value(scope),
            Builtin::CubeClusterDim => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimX => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimY => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimZ => constant::expand(scope, 1).value(scope),
            Builtin::CubeCount => cube_count::expand(scope).value(scope),
            Builtin::CubeCountX => {
                self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 0, uniformity)
            }
            Builtin::CubeCountY => {
                self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 1, uniformity)
            }
            Builtin::CubeCountZ => {
                self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 2, uniformity)
            }
            Builtin::PlaneDim => {
                self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupSize, uniformity)
            }
            Builtin::PlanePos => {
                self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupId, uniformity)
            }
            Builtin::UnitPosPlane => {
                self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupLocalInvocationId, uniformity)
            }
            Builtin::AbsolutePos => absolute_pos::expand(scope).value(scope),
            Builtin::AbsolutePosX => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 0, uniformity)
            }
            Builtin::AbsolutePosY => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 1, uniformity)
            }
            Builtin::AbsolutePosZ => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 2, uniformity)
            }
        }
    }

    fn read_scalar_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        builtin: BuiltIn,
        uniformity: Option<UniformAttr>,
    ) -> Value {
        let ptr = self.builtin_ptr(scope, builtin);
        let ctx = scope.ctx_mut();
        let load = LoadOp::new(ctx, ty, ptr, MemoryAccess::NONE, None);
        decorate_uniform(ctx, load.get_operation(), uniformity);
        scope.register_with_result(&load)
    }

    fn read_dim3_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        builtin: BuiltIn,
        dim: u32,
        uniformity: Option<UniformAttr>,
    ) -> Value {
        let ptr = self.builtin_ptr(scope, builtin);
        let dim_const = const_int32(scope, dim);

        let ctx = scope.ctx_mut();
        let ptr_ty = PointerType::get(ctx, ty, StorageClass::Input).to_handle();
        let access_chain = InBoundsAccessChainOp::new(ctx, ptr_ty, ptr, vec![dim_const]);
        decorate_uniform(ctx, access_chain.get_operation(), Some(DEVICE_UNIFORM));
        let offset_ptr = scope.register_with_result(&access_chain);
        let load = LoadOp::new(ctx, ty, offset_ptr, MemoryAccess::NONE, None);
        decorate_uniform(ctx, load.get_operation(), uniformity);
        scope.register_with_result(&load)
    }
}

pub(crate) fn const_int32(scope: &Scope, value: u32) -> Value {
    let constant = const_op_int32(scope.ctx_mut(), value);
    scope.register_with_result(&constant)
}

pub(crate) fn const_op_int32(ctx: &mut Context, value: u32) -> ConstantOp {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let value = IntegerAttr::new(ty, APInt::from_u32(value, bw(32)));
    ConstantOp::new(ctx, Box::new(value))
}
