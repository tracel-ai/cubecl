use alloc::vec::Vec;

use cubecl_ir::{
    dialect::{
        base::OperationPtrExt,
        general::{AggregateConstructOp, AggregateExtractOp},
    },
    prelude::*,
};
use hashbrown::HashMap;
use pliron::{
    graph::walkers::{WALKCONFIG_PREORDER_FORWARD, uninterruptible::mutable::walk_op},
    opts::dce::dce,
};

type Aggregates = HashMap<Value, Vec<Value>>;

#[derive(Default)]
pub struct DisaggregatePass;

impl Pass for DisaggregatePass {
    fn name(&self) -> &str {
        "Disaggregate"
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut state = (Aggregates::new(), PassResult::default());
        walk_op(
            ctx,
            &mut state,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, (aggregates, res), node| {
                if let IRNode::Operation(op) = node {
                    if let Some(construct) = op.as_op::<AggregateConstructOp>(ctx) {
                        aggregates.insert(construct.get_result(ctx), construct.values(ctx));
                    }
                    if let Some(extract) = op.as_op::<AggregateExtractOp>(ctx) {
                        let field = extract.field(ctx).0;
                        let value = aggregates[&extract.aggregate(ctx)][field];
                        extract.get_result(ctx).replace_all_uses_with(ctx, &value);
                        res.ir_changed |= IRStatus::Changed;
                    }
                }
            },
        );
        let mut res = state.1;
        res.ir_changed |= dce(op, ctx)?;
        Ok(res)
    }
}
