//! Control-flow unaware common subexpression elimination. Useful for applying to structured
//! control flow representation that can't lower (i.e. for C++). May also be useful as a quick
//! "first-pass" to reduce the graph complexity before lowering to a CFG.
//!
//! Uses a scoped model where expressions are never hoisted, but fully redundant expressions are
//! eliminated. Only works for single-block regions, and will return an error if there are any
//! multi-block regions. The logic isn't complex enough to handle them properly.

use alloc::string::{String, ToString};
use core::fmt::{self, Formatter};
use itertools::Itertools;

use cubecl_ir::{
    interfaces::MemoryEffects,
    prelude::{Rewriter as _, *},
};
use pliron::{
    attribute::AttributeDict,
    graph::ControlFlowGraph,
    irbuild::listener::DummyListener,
    linked_list::ContainsLinkedList,
    op::OpId,
    opts::dce::SideEffects,
    printable::{Printable, State},
    verify_err,
};
use thiserror::Error;

use crate::scoped_map::ScopedMap;

#[derive(Clone, PartialEq, Eq)]
struct ExpressionKey {
    op_id: OpId,
    operands: Vec<Value>,
    attributes: AttributeDict,
    result_types: Vec<TypeHandle>,
}

impl ExpressionKey {
    pub fn new(ctx: &Context, op: Ptr<Operation>) -> Self {
        let op_id = op.dyn_op(ctx).get_opid();
        let operands = op.operands(ctx);
        let attributes = op.deref(ctx).attributes.clone();
        let result_types = op.deref(ctx).result_types().collect();
        Self {
            op_id,
            operands,
            attributes,
            result_types,
        }
    }
}

impl Printable for ExpressionKey {
    fn fmt(&self, ctx: &Context, _state: &State, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expression<{}, operands: [{}], attributes: [{}]>",
            self.op_id.disp(ctx),
            self.operands.iter().map(|it| it.disp(ctx)).join(", "),
            self.attributes.disp(ctx)
        )
    }
}

type Expressions = ScopedMap<ExpressionKey, Ptr<Operation>>;
type Rewriter = IRRewriter<DummyListener>;

#[derive(Error, Debug)]
pub enum SimpleCSEError {
    #[error("Encountered multi-block region op {_0}. This is not supported for simple CSE.")]
    MultiBlockRegion(String),
}

#[derive(Default)]
pub struct SimpleCSEPass;

#[pass_name]
impl Pass for SimpleCSEPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let mut rewriter = Rewriter::default();
        let mut expressions = ScopedMap::new();

        res.ir_changed |= cse_op(ctx, op, &mut rewriter, &mut expressions)?;
        Ok(res)
    }
}

fn cse_op(
    ctx: &mut Context,
    op: Ptr<Operation>,
    rewriter: &mut Rewriter,
    expressions: &mut Expressions,
) -> Result<IRStatus> {
    if can_eliminate(ctx, op) {
        let key = ExpressionKey::new(ctx, op);
        if let Some(existing) = expressions.get(&key) {
            rewriter.replace_operation(ctx, op, *existing);
            return Ok(IRStatus::Changed);
        } else {
            expressions.insert(key, op);
            return Ok(IRStatus::Unchanged);
        }
    }

    let mut status = IRStatus::Unchanged;
    let regions = op.deref(ctx).regions().collect::<Vec<_>>();
    for region in regions {
        if region.deref(ctx).iter(ctx).count() > 1 {
            let loc = op.deref(ctx).loc();
            verify_err!(
                loc,
                SimpleCSEError::MultiBlockRegion(op.disp(ctx).to_string())
            )?;
        }
        expressions.push_scope();

        let block = region.entry_node(ctx).unwrap();
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for op in ops {
            status |= cse_op(ctx, op, rewriter, expressions)?;
        }

        expressions.pop_scope();
    }
    Ok(status)
}

fn can_eliminate(ctx: &Context, op: Ptr<Operation>) -> bool {
    let dyn_op = op.dyn_op(ctx);
    let no_side_effects =
        op_cast::<dyn SideEffects>(&*dyn_op).is_some_and(|effects| !effects.has_side_effects(ctx));
    let no_memory_effects = op_cast::<dyn MemoryEffects>(&*dyn_op)
        .is_some_and(|effects| effects.memory_effects(ctx).is_empty());
    no_side_effects && no_memory_effects
}
