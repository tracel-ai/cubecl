use cubecl_ir::{
    dialect::{
        branch::{ConditionOp, YieldOp},
        scf,
    },
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock,
    opts::mem2reg::AllocInfo,
    region::Region,
    utils::table::{HMap, SmallMap},
};

use crate::passes::mem2reg::PromotableRegionOpInterface;

fn update_terminator<T: Op>(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    default_reaching_def: Value,
    reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
) {
    let Some(term) = block.deref(ctx).get_terminator(ctx) else {
        return;
    };
    if !term.is_op::<T>(ctx) {
        return;
    }
    let block_reaching_def = reaching_at_block_end.get(&block).copied();
    let block_reaching_def = block_reaching_def.unwrap_or(default_reaching_def);
    Operation::push_operand(term, ctx, block_reaching_def);
}

#[op_interface_impl]
impl PromotableRegionOpInterface for scf::IfOp {
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
        regions_to_process.insert(self.then_region(ctx), reaching_def);
        regions_to_process.insert(self.else_region(ctx), reaching_def);
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

        update_terminator::<YieldOp>(
            ctx,
            self.then_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );
        update_terminator::<YieldOp>(
            ctx,
            self.else_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_result(ctx, res_idx)
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for scf::SwitchOp {
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
        regions_to_process.insert(self.default_region(ctx), reaching_def);
        for case_region in self.case_regions(ctx) {
            regions_to_process.insert(case_region, reaching_def);
        }
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return reaching_def;
        }

        update_terminator::<YieldOp>(
            ctx,
            self.default_block(ctx),
            reaching_def,
            reaching_at_block_end,
        );
        for case_block in self.case_blocks(ctx) {
            update_terminator::<YieldOp>(ctx, case_block, reaching_def, reaching_at_block_end);
        }

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(res_idx)
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for scf::RangeLoopOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        let body_region = self.loop_region(ctx);
        if !has_value_stores {
            regions_to_process.insert(body_region, reaching_def);
            return;
        }

        self.push_initial_carried_value(ctx, reaching_def);
        let idx = BasicBlock::push_argument(self.loop_body(ctx), ctx, alloc.ty);
        let new_arg = self.loop_body(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(body_region, new_arg);
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

        update_terminator::<YieldOp>(
            ctx,
            self.loop_body(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        let idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(idx)
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for scf::WhileOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        let before_region = self.before_region(ctx);
        let after_region = self.after_region(ctx);
        if !has_value_stores {
            regions_to_process.insert(before_region, reaching_def);
            regions_to_process.insert(after_region, reaching_def);
            return;
        }

        self.push_initial_carried_value(ctx, reaching_def);

        let idx = BasicBlock::push_argument(self.before_block(ctx), ctx, alloc.ty);
        let new_arg = self.before_block(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(before_region, new_arg);

        let idx = BasicBlock::push_argument(self.after_block(ctx), ctx, alloc.ty);
        let new_arg = self.after_block(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(after_region, new_arg);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return reaching_def;
        }

        let before = self.before_block(ctx);
        let last_arg = before.deref(ctx).arguments().last();
        update_terminator::<ConditionOp>(ctx, before, last_arg.unwrap(), reaching_at_block_end);

        let after = self.after_block(ctx);
        let last_arg = after.deref(ctx).arguments().last();
        update_terminator::<YieldOp>(ctx, after, last_arg.unwrap(), reaching_at_block_end);

        let idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(idx)
    }
}
