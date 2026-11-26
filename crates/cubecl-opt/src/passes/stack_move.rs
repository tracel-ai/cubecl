use alloc::vec::Vec;
use cubecl_ir::{
    AddressSpace,
    dialect::{
        OperationPtrExt,
        memory::{CopyOp, LoadOp, StoreOp},
    },
    interfaces::{
        TypeExt,
        aliasing::{AliasingOp, PointerExt},
        side_effects::Extent,
    },
    prelude::{AnalysisManager, Context, Operation, PassResult, Ptr, Result, pass_name},
    rewrite::WALKCONFIG_ANY,
};
use pliron::{
    graph::{
        dominance::DomInfo,
        walkers::{IRNode, uninterruptible::immutable::walk_op},
    },
    irbuild::IRStatus,
    opts::mem2reg::PromotableAllocationInterface,
    pass::Pass,
    r#type::Typed,
    value::Value,
};

use crate::analyses::{
    alias_analysis::{self, AliasAnalysisStack, AliasResult},
    liveness::PointerLiveness,
    slices::get_value_transitive_users,
};

/// Eliminates moves between stack variables. A move requires the following:
/// * The source is dead after the read
/// * The store dominates all accesses of the destination root
/// * The entire allocation is accessed
pub struct RemoveStackMoves;

#[pass_name]
impl Pass for RemoveStackMoves {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut liveness = analyses.get_analysis_mut::<PointerLiveness>(op, ctx)?;
        let aa = alias_analysis::default_stack();
        let mut dom_info = DomInfo::default();

        let mut to_erase = Vec::new();
        walk_op(
            ctx,
            &mut (&aa, &mut liveness, &mut dom_info, &mut to_erase),
            &WALKCONFIG_ANY,
            op,
            |ctx, (aa, liveness, dom_info, to_erase), node| {
                let IRNode::Operation(op) = node else {
                    return;
                };
                if op.deref(ctx).num_regions() > 0 {
                    return;
                }
                if let Some((read, from, to)) = op_as_copy(ctx, aa, op) {
                    if liveness.is_live_after(ctx, read, from) {
                        return;
                    }
                    let mut uses_to_check = to.uses(ctx);
                    while let Some(r#use) = uses_to_check.pop() {
                        let user = r#use.user_op();
                        if user == op {
                            continue;
                        }
                        let root_doms = dom_info.value_strictly_dominates_op(ctx, from, user);
                        if let Some(aliasing) = user.cast::<dyn AliasingOp>(ctx)
                            && root_doms
                        {
                            let transitive = aliasing
                                .dest_ptr(ctx)
                                .into_iter()
                                .flat_map(|ptr| ptr.uses(ctx));
                            uses_to_check.extend(transitive);
                            continue;
                        }
                        let store_doms = dom_info.op_strictly_dominates_op(ctx, op, user);
                        if !root_doms || !store_doms {
                            return;
                        }
                    }
                    let affected = get_value_transitive_users(ctx, to);
                    to.replace_all_uses_with(ctx, &from);
                    liveness.update_in_place(ctx, affected);
                    to_erase.push(op);
                }
            },
        );

        let mut res = PassResult::default();
        for op in to_erase {
            Operation::erase(op, ctx);
            res.ir_changed |= IRStatus::Changed;
        }

        Ok(res)
    }
}

fn op_as_copy(
    ctx: &Context,
    aa: &AliasAnalysisStack,
    op: Ptr<Operation>,
) -> Option<(Ptr<Operation>, Value, Value)> {
    let (read, from, to) = if let Some(copy) = op.as_op::<CopyOp>(ctx)
        && copy.len(ctx).0 == 1
    {
        (op, copy.source(ctx), copy.destination(ctx))
    } else if let Some(store) = op.as_op::<StoreOp>(ctx)
        && let Some(value_def) = store.value(ctx).defining_op()
        && let Some(load) = value_def.as_op::<LoadOp>(ctx)
        && store.ptr(ctx).get_type(ctx).as_ptr(ctx).address_space == AddressSpace::Local
        && load.ptr(ctx).get_type(ctx).as_ptr(ctx).address_space == AddressSpace::Local
    {
        (value_def, load.ptr(ctx), store.ptr(ctx))
    } else {
        return None;
    };
    let from_root = from.get_root_ptr(ctx);
    let to_root = to.get_root_ptr(ctx);

    if !is_from_promotable_alloc(ctx, from_root) || !is_from_promotable_alloc(ctx, to_root) {
        return None;
    }

    let must_alias = |ptr, root| {
        aa.alias(
            ctx,
            (ptr, Extent::ExactPointer),
            (root, Extent::ExactPointer),
        ) == AliasResult::MustAlias
    };

    if must_alias(from, from_root) && must_alias(to, to_root) {
        Some((read, from_root, to_root))
    } else {
        None
    }
}

fn is_from_promotable_alloc(ctx: &Context, ptr: Value) -> bool {
    ptr.defining_op()
        .is_some_and(|op| op.impls::<dyn PromotableAllocationInterface>(ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        irfmt::parsers::spaced, linked_list::ContainsLinkedList, operation::verify_operation,
        parsable::parse_from_str, pass::Analysis, result::ExpectOk,
    };

    #[test]
    fn refreshing_a_replaced_pointer_updates_uses_of_its_alias() -> Result<()> {
        let input = r#"
          builtin.func @f: builtin.function <() -> (cube.index)> [] {
            ^entry():
              a = memory.declare_variable cube.index Local, align = 4, init = cube.index 7;
              b = memory.declare_variable cube.index Local, align = 4, init = cube.index 0;
              alias = cube.copy (b) [] []: <(cube.ptr <cube.index, Local>) -> (cube.ptr <cube.index, Local>)>;
              value = memory.load (alias) [] []: <(cube.ptr <cube.index, Local>) -> (cube.index)>;
              branch.return value
          }
        "#;

        let ctx = &mut Context::new();
        let root = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
        verify_operation(root, ctx)?;
        let entry = root.deref(ctx).get_region(0).deref(ctx).get_head().unwrap();
        let entry = entry.deref(ctx);
        let mut ops = entry.iter(ctx);
        let from = ops.next().unwrap().deref(ctx).get_result(0);
        let to = ops.next().unwrap().deref(ctx).get_result(0);
        let alias = ops.next().unwrap();
        drop(entry);

        let mut liveness = PointerLiveness::compute(root, ctx, &mut AnalysisManager::default())?;
        assert!(!liveness.is_live_after(ctx, alias, from));

        let affected = get_value_transitive_users(ctx, to);
        to.replace_all_uses_with(ctx, &from);
        liveness.update_in_place(ctx, affected);
        assert!(liveness.is_live_after(ctx, alias, from));
        Ok(())
    }
}
