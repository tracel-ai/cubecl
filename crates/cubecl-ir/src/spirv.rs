use alloc::{vec, vec::Vec};

use crate::{
    NoMemoryEffect,
    interfaces::{
        AlignedType, MaybeVectorizedType, MemoryEffect, MemoryEffects, ScalarizableType, TypedExt,
        memory_slot::PromotableRegionOpInterface,
    },
    scalar,
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    opts::mem2reg::AllocInfo,
    region::Region,
    r#type::{TypeHandle, Typed, TypedHandle},
    utils::table::{HMap, SmallMap},
    value::Value,
};
use pliron_spirv::{
    ops::{AccessChainOp, InBoundsAccessChainOp, LoadOp, LoopOp, SelectionOp},
    spirv::StorageClass,
    types::{ArrayType, FloatType, PointerType, VectorType, khr::CooperativeMatrixType},
};

NoMemoryEffect!(InBoundsAccessChainOp);
NoMemoryEffect!(AccessChainOp);

#[op_interface_impl]
impl MemoryEffects for LoadOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let ptr = self.get_operand_pointer(ctx);
        let Ok(ptr_ty) = TypedHandle::<PointerType>::from_handle(ptr.get_type(ctx), ctx) else {
            return vec![MemoryEffect::Read(self.get_operand_pointer(ctx))];
        };
        let storage_class = ptr_ty.deref(ctx).storage_class;
        match storage_class {
            // Inherently readonly memory that does not observe writes, memory effects are not
            // relevant to value. Until we add a better way to represent that, treat it as no memory
            // effect.
            StorageClass::UniformConstant
            | StorageClass::Input
            | StorageClass::ShaderRecordBufferKHR
            | StorageClass::IncomingCallableDataKHR
            | StorageClass::IncomingRayPayloadKHR => vec![],
            _ => vec![MemoryEffect::Read(self.get_operand_pointer(ctx))],
        }
    }
}

scalar!(FloatType);

#[type_interface_impl]
impl AlignedType for FloatType {
    fn align(&self, _ctx: &Context) -> usize {
        self.width.div_ceil(8) as usize
    }
}

#[type_interface_impl]
impl AlignedType for VectorType {
    fn align(&self, ctx: &Context) -> usize {
        self.count as usize * self.element_type.align(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for ArrayType {
    fn align(&self, ctx: &Context) -> usize {
        self.element_type.align(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for CooperativeMatrixType {
    fn align(&self, ctx: &Context) -> usize {
        self.component_type.align(ctx)
    }
}

#[type_interface_impl]
impl MaybeVectorizedType for VectorType {
    fn vector_size(&self, _ctx: &Context) -> usize {
        self.count as usize
    }
}

#[type_interface_impl]
impl ScalarizableType for VectorType {
    fn scalar_type(&self, ctx: &Context) -> TypeHandle {
        self.element_type.scalar_ty(ctx)
    }
}

fn update_merge(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    default_reaching_def: Value,
    reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
) {
    let term = block.deref(ctx).get_terminator(ctx);
    let term = term.expect("Should have terminator");
    let block_reaching_def = reaching_at_block_end.get(&block).copied();
    let block_reaching_def = block_reaching_def.unwrap_or(default_reaching_def);
    Operation::push_operand(term, ctx, block_reaching_def);
}

#[op_interface_impl]
impl PromotableRegionOpInterface for SelectionOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        _: &AllocInfo,
        reaching_def: Value,
        _: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        regions_to_process.insert(self.region(ctx), reaching_def);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return entry_reaching_def;
        }

        let merge_block = self.region(ctx).deref(ctx).get_tail().unwrap();
        update_merge(ctx, merge_block, entry_reaching_def, reaching_at_block_end);

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(res_idx)
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for LoopOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        _: &AllocInfo,
        reaching_def: Value,
        _: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        regions_to_process.insert(self.region(ctx), reaching_def);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return entry_reaching_def;
        }

        let merge_block = self.region(ctx).deref(ctx).get_tail().unwrap();
        update_merge(ctx, merge_block, entry_reaching_def, reaching_at_block_end);

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(res_idx)
    }
}
