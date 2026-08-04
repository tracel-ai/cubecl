use cubecl_ir::{
    dialect::{branch::YieldOp, scf},
    prelude::*,
};
use pliron::{
    basic_block::BasicBlock, linked_list::ContainsLinkedList, opts::mem2reg::AllocInfo,
    region::Region, utils::table::IMap,
};

use crate::passes::mem2reg::{
    RegionDefOpInterface, RegionEdge, RegionGraph,
    RegionGraphNode::{self, AfterOp, BeforeOp},
    mem2reg_2::PromotableRegionOpInterface,
};

#[op_interface_impl]
impl RegionDefOpInterface for scf::IfOp {
    fn get_node_argument(&self, ctx: &Context, node: RegionGraphNode, arg_idx: usize) -> Value {
        match node {
            AfterOp => self.get_operation().deref(ctx).get_result(arg_idx),
            _ => panic!("node argument doesn't exist"),
        }
    }

    fn can_add_node_argument(&self, _ctx: &Context, node: RegionGraphNode) -> bool {
        matches!(node, AfterOp)
    }

    fn add_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, ty: TypeHandle) -> usize {
        match node {
            AfterOp => Operation::push_result(self.get_operation(), ctx, ty),
            _ => panic!("node argument doesn't exist"),
        }
    }

    fn remove_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, arg_idx: usize) {
        if node == AfterOp {
            Operation::remove_result(self.get_operation(), ctx, arg_idx);
        }
    }

    fn region_graph(&self, ctx: &Context) -> RegionGraph {
        let mut graph = RegionGraph::new();

        let then_node = RegionGraphNode::Region(self.then_region(ctx));
        let else_node = RegionGraphNode::Region(self.else_region(ctx));

        graph.add_successor(BeforeOp, then_node);
        graph.add_successor(BeforeOp, else_node);
        graph.add_successor(then_node, AfterOp);
        graph.add_successor(else_node, AfterOp);

        graph
    }

    fn add_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, operand: Value) {
        if let Some(yield_) = node_yield(ctx, edge) {
            Operation::push_operand(yield_, ctx, operand);
        }
    }

    fn remove_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, opd_idx: usize) {
        if let Some(yield_) = node_yield(ctx, edge) {
            Operation::remove_operand(yield_, ctx, opd_idx);
        }
    }
}

#[op_interface_impl]
impl RegionDefOpInterface for scf::SwitchOp {
    fn get_node_argument(&self, ctx: &Context, node: RegionGraphNode, arg_idx: usize) -> Value {
        match node {
            AfterOp => self.get_operation().deref(ctx).get_result(arg_idx),
            _ => panic!("node argument doesn't exist"),
        }
    }

    fn can_add_node_argument(&self, _ctx: &Context, node: RegionGraphNode) -> bool {
        matches!(node, AfterOp)
    }

    fn add_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, ty: TypeHandle) -> usize {
        match node {
            AfterOp => Operation::push_result(self.get_operation(), ctx, ty),
            _ => panic!("node argument doesn't exist"),
        }
    }

    fn remove_node_argument(&self, ctx: &mut Context, node: RegionGraphNode, arg_idx: usize) {
        if node == AfterOp {
            Operation::remove_result(self.get_operation(), ctx, arg_idx);
        }
    }

    fn region_graph(&self, ctx: &Context) -> RegionGraph {
        let mut graph = RegionGraph::new();

        let cases = self.get_operation().regions(ctx);

        for case in cases {
            let node = RegionGraphNode::Region(case);
            graph.add_successor(BeforeOp, node);
            graph.add_successor(node, AfterOp);
        }

        graph
    }

    fn add_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, operand: Value) {
        if let Some(yield_) = node_yield(ctx, edge) {
            Operation::push_operand(yield_, ctx, operand);
        }
    }

    fn remove_edge_operand(&self, ctx: &mut Context, edge: RegionEdge, opd_idx: usize) {
        if let Some(yield_) = node_yield(ctx, edge) {
            Operation::remove_operand(yield_, ctx, opd_idx);
        }
    }
}

fn region_yield(ctx: &Context, region: Ptr<Region>) -> Option<Ptr<Operation>> {
    let body = region.deref(ctx).get_head()?;
    let term = body.deref(ctx).get_terminator(ctx)?;
    if term.is_op::<YieldOp>(ctx) {
        Some(term)
    } else {
        None
    }
}

fn node_yield(ctx: &Context, edge: RegionEdge) -> Option<Ptr<Operation>> {
    match edge.pred {
        RegionGraphNode::Region(region) => region_yield(ctx, region),
        _ => None,
    }
}

fn update_terminator<T: Op>(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    default_reaching_def: Value,
    reaching_at_block_end: &IMap<Ptr<BasicBlock>, Value>,
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
        ctx: &Context,
        _: &AllocInfo,
        reaching_def: Value,
        _: bool,
        regions_to_process: &mut IMap<Ptr<Region>, Value>,
    ) {
        regions_to_process.insert(self.then_region(ctx), reaching_def);
        regions_to_process.insert(self.else_region(ctx), reaching_def);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        _: bool,
        reaching_at_block_end: &IMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
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

// impl PromotableRegionOpInterface for scf::WhileOp {
//     fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
//         true
//     }

//     fn setup_promotion(
//         &self,
//         ctx: &Context,
//         alloc: &AllocInfo,
//         reaching_def: Value,
//         has_value_stores: bool,
//         regions_to_process: &mut IMap<Ptr<Region>, Value>,
//     ) {
//         todo!()
//     }

//     fn finalize_promotion(
//         &self,
//         ctx: &mut Context,
//         alloc: &AllocInfo,
//         entry_reaching_def: Value,
//         has_value_stores: bool,
//         reaching_at_block_end: &IMap<Ptr<BasicBlock>, Value>,
//     ) -> Value {
//         todo!()
//     }
// }
