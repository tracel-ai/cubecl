use alloc::vec::Vec;

use cubecl_ir::{
    dialect::{
        base::OperationPtrExt,
        general::{AggregateConstructOp, AggregateExtractOp},
    },
    pliron::{
        builtin::op_interfaces::OneResultInterface,
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op,
        },
        pass_manager::{AnalysisManager, Pass, PassResult},
        prelude::{Context, Operation, Ptr, Result},
        value::Value,
    },
};
use hashbrown::HashMap;
use pliron::irbuild::IRStatus;

type Aggregates = HashMap<Value, Vec<Value>>;

pub struct DisaggregateTransform;

impl Pass for DisaggregateTransform {
    fn name(&self) -> &str {
        "Disaggregate"
    }

    fn run(
        &self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut result = PassResult::default();
        result.ir_changed = IRStatus::Changed;

        let mut aggregates = Aggregates::new();
        walk_op(
            ctx,
            &mut aggregates,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, aggregates, node| {
                if let IRNode::Operation(op) = node {
                    if let Some(construct) = op.as_op::<AggregateConstructOp>(ctx) {
                        aggregates.insert(construct.get_result(ctx), construct.values(ctx));
                    }
                    if let Some(extract) = op.as_op::<AggregateExtractOp>(ctx) {
                        let field = extract.get_attr_field(ctx).unwrap().0;
                        let value = aggregates[&extract.aggregate(ctx)][field];
                        extract.get_result(ctx).replace_all_uses_with(ctx, &value);
                    }
                }
            },
        );
        Ok(result)
    }
}
