use core::{any::type_name, num::NonZeroUsize};

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    Builtin, ContextExt,
    dialect::general::ReadBuiltinOp,
    prelude::{
        AnalysisManager, Context, MatchRewrite, MatchRewriter, OneResultInterface, Operation,
        OperationPtrExt, PassResult, Ptr, Result, Rewriter, SingleBlockRegionInterface,
    },
    rewrite::visit_all_ops_of_type_mut,
    types::scalar::PoisonType,
};
use hashbrown::HashMap;
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::{ConstantOp, FuncOp, ModuleOp},
        types::{IntegerType, Signedness},
    },
    dict_key,
    irbuild::match_rewrite::apply_match_rewrite,
    op::Op,
    pass::Pass,
    r#type::{TypeHandle, Typed},
    utils::apint::APInt,
    value::Value,
};
use pliron_spirv::{
    attrs::BuiltInAttr,
    decorations::DecorationInfo,
    ops::{AccessChainOp, AddressOfOp, GlobalVariableOp, LoadOp},
    types::{MemberDecorationInfo, PointerType, StructType, VectorType},
};
use rspirv::spirv::{BuiltIn, Decoration, MemoryAccess, StorageClass};

use crate::KernelInfo;

dict_key!(BUILTINS_NAME, "_spirv_builtins");

pub struct LowerBuiltinsPass;

impl Pass for LowerBuiltinsPass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let module = op.as_op::<ModuleOp>(ctx).expect("Should run on module");
        let module = module.get_body(ctx, 0);

        let mut offsets = HashMap::new();

        let mut funcs = vec![];

        visit_all_ops_of_type_mut::<FuncOp, _>(ctx, &mut funcs, op, |ctx, funcs, func| {
            let entry = func.get_entry_block(ctx);
            let dummy_ty = PoisonType::get(ctx).to_handle();
            let tmp_builtins = AddressOfOp::new(ctx, dummy_ty, BUILTINS_NAME.clone());
            tmp_builtins.get_operation().insert_at_front(entry, ctx);
            funcs.push((func, tmp_builtins.get_result(ctx)))
        });

        for (func, struct_val) in funcs.iter().copied() {
            let op = func.get_operation();
            let mut rewriter = RewriteBuiltins {
                struct_val,
                offsets: &mut offsets,
            };
            res.ir_changed |= apply_match_rewrite(ctx, &mut rewriter, Default::default(), op)?;
        }

        if offsets.is_empty() {
            // Just keep builtins struct around with a dummy builtin to make entrypoint logic simpler
            offsets.insert(BuiltIn::LocalInvocationIndex, 0);
        }

        let mut used_builtins = offsets.into_iter().collect::<Vec<_>>();
        used_builtins.sort_by_key(|(_, offset)| *offset);

        let mut field_types = vec![];
        let mut member_dec = vec![];
        let type_dec = vec![DecorationInfo::unit(Decoration::Block)];

        for (builtin, id) in used_builtins {
            let decoration =
                DecorationInfo::new(Decoration::BuiltIn, Box::new(BuiltInAttr::new(builtin)));
            field_types.push(builtin_ty(ctx, builtin));
            member_dec.push(MemberDecorationInfo::new(id, decoration));
        }

        let builtins_struct =
            StructType::get(ctx, field_types, vec![], member_dec, type_dec).into();
        let builtins_struct_ptr =
            PointerType::get(ctx, builtins_struct, StorageClass::Input).into();
        let builtins_var = GlobalVariableOp::new(
            ctx,
            builtins_struct,
            StorageClass::Input,
            BUILTINS_NAME.clone(),
            None,
        );
        builtins_var.get_operation().insert_at_front(module, ctx);

        for (_, struct_val) in funcs.iter() {
            struct_val.set_type(ctx, builtins_struct_ptr);
        }

        Ok(res)
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

struct RewriteBuiltins<'a> {
    struct_val: Value,
    offsets: &'a mut HashMap<BuiltIn, u32>,
}

impl RewriteBuiltins<'_> {
    fn get_offset(&mut self, scope: &Scope, builtin: BuiltIn) -> Value {
        let value = if let Some(&existing) = self.offsets.get(&builtin) {
            existing
        } else {
            let offset = self.offsets.len() as u32;
            self.offsets.insert(builtin, offset);
            offset
        };
        const_int32(scope, value)
    }
}

impl MatchRewrite for RewriteBuiltins<'_> {
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
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let ty = builtin.get_result(ctx).get_type(ctx);
        let builtin = builtin.builtin(ctx).0;
        let value = self.lower_builtin(&scope, ty, builtin);
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

#[cube]
fn absolute_pos() -> usize {
    let cubes_x = CUBE_COUNT_X as usize;
    let cubes_y = CUBE_COUNT_Y as usize;
    let cube_dim_x = CUBE_DIM_X as usize;
    let cube_dim_y = CUBE_DIM_Y as usize;
    let z = ABSOLUTE_POS_Z as usize * cubes_x * cube_dim_x * cubes_y * cube_dim_y;
    let y = ABSOLUTE_POS_Y as usize * cubes_x * cube_dim_x;
    z + y + ABSOLUTE_POS_X as usize
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

impl RewriteBuiltins<'_> {
    fn lower_builtin(&mut self, scope: &Scope, ty: TypeHandle, builtin: Builtin) -> Value {
        let cube_dim = scope.ctx().aux_ty::<KernelInfo>().cube_dim;
        match builtin {
            Builtin::UnitPos if cube_dim.num_elems() == 1 => {
                constant::expand(scope, 0).value(scope)
            }
            Builtin::UnitPosX if cube_dim.x == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosY if cube_dim.y == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPosZ if cube_dim.z == 1 => constant::expand(scope, 0).value(scope),
            Builtin::UnitPos => self.read_scalar_builtin(scope, ty, BuiltIn::LocalInvocationIndex),
            Builtin::UnitPosX => self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 0),
            Builtin::UnitPosY => self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 1),
            Builtin::UnitPosZ => self.read_dim3_builtin(scope, ty, BuiltIn::LocalInvocationId, 2),
            Builtin::CubePosCluster => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterX => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterY => constant::expand(scope, 0).value(scope),
            Builtin::CubePosClusterZ => constant::expand(scope, 0).value(scope),
            Builtin::CubePos => cube_pos::expand(scope).value(scope),
            Builtin::CubePosX => self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 0),
            Builtin::CubePosY => self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 1),
            Builtin::CubePosZ => self.read_dim3_builtin(scope, ty, BuiltIn::WorkgroupId, 2),
            Builtin::CubeDim => constant::expand(scope, cube_dim.num_elems()).value(scope),
            Builtin::CubeDimX => constant::expand(scope, cube_dim.x).value(scope),
            Builtin::CubeDimY => constant::expand(scope, cube_dim.y).value(scope),
            Builtin::CubeDimZ => constant::expand(scope, cube_dim.z).value(scope),
            Builtin::CubeClusterDim => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimX => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimY => constant::expand(scope, 1).value(scope),
            Builtin::CubeClusterDimZ => constant::expand(scope, 1).value(scope),
            Builtin::CubeCount => cube_count::expand(scope).value(scope),
            Builtin::CubeCountX => self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 0),
            Builtin::CubeCountY => self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 1),
            Builtin::CubeCountZ => self.read_dim3_builtin(scope, ty, BuiltIn::NumWorkgroups, 2),
            Builtin::PlaneDim => self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupSize),
            Builtin::PlanePos => self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupId),
            Builtin::UnitPosPlane => {
                self.read_scalar_builtin(scope, ty, BuiltIn::SubgroupLocalInvocationId)
            }
            Builtin::AbsolutePos => absolute_pos::expand(scope).value(scope),
            Builtin::AbsolutePosX if cube_dim.x == 1 => constant::expand(scope, 0).value(scope),
            Builtin::AbsolutePosY if cube_dim.y == 1 => constant::expand(scope, 0).value(scope),
            Builtin::AbsolutePosZ if cube_dim.z == 1 => constant::expand(scope, 0).value(scope),
            Builtin::AbsolutePosX => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 0)
            }
            Builtin::AbsolutePosY => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 1)
            }
            Builtin::AbsolutePosZ => {
                self.read_dim3_builtin(scope, ty, BuiltIn::GlobalInvocationId, 2)
            }
        }
    }

    fn read_scalar_builtin(&mut self, scope: &Scope, ty: TypeHandle, builtin: BuiltIn) -> Value {
        let ctx = scope.ctx_mut();
        let offset = self.get_offset(scope, builtin);
        let ptr_ty = PointerType::get(ctx, ty, StorageClass::Input).to_handle();
        let access_chain = AccessChainOp::new(ctx, ptr_ty, self.struct_val, vec![offset]);
        let offset_ptr = scope.register_with_result(&access_chain);
        let load = LoadOp::new(ctx, ty, offset_ptr, MemoryAccess::NONE, None);
        scope.register_with_result(&load)
    }

    fn read_dim3_builtin(
        &mut self,
        scope: &Scope,
        ty: TypeHandle,
        builtin: BuiltIn,
        dim: u32,
    ) -> Value {
        let ctx = scope.ctx_mut();
        let offset = self.get_offset(scope, builtin);
        let dim_const = const_int32(scope, dim);

        let ptr_ty = PointerType::get(ctx, ty, StorageClass::Input).to_handle();
        let access_chain =
            AccessChainOp::new(ctx, ptr_ty, self.struct_val, vec![offset, dim_const]);
        let offset_ptr = scope.register_with_result(&access_chain);
        let load = LoadOp::new(ctx, ty, offset_ptr, MemoryAccess::NONE, None);
        scope.register_with_result(&load)
    }
}

fn const_int32(scope: &Scope, value: u32) -> Value {
    let ty = IntegerType::get(scope.ctx(), 32, Signedness::Signless);
    let value = IntegerAttr::new(ty, APInt::from_u32(value, NonZeroUsize::new(32).unwrap()));
    let constant = ConstantOp::new(scope.ctx_mut(), Box::new(value));
    scope.register_with_result(&constant)
}
