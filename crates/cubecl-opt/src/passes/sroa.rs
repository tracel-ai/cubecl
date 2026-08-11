use alloc::vec::Vec;
use cubecl_environment::collections::HashMap;
use cubecl_ir::{
    interfaces::memory_slot::{
        DeletionKind, DestructurableAccessorOpInterface, DestructurableConstructorOpInterface,
        DestructurableValueSlot, SafeMemorySlotAccessOpInterface,
    },
    prelude::*,
    rewrite::TraitOp,
};
use pliron::{
    attribute::AttrObj,
    graph::walkers::uninterruptible::immutable::walk_region,
    linked_list::ContainsLinkedList,
    utils::{
        table::{ISet, SmallSet},
        vec_exns::VecExtns,
    },
    value::Use,
};

use crate::analyses::slices::get_value_forward_slice;

#[derive(Default)]
struct ValueDestructuringInfo {
    used_indices: SmallSet<AttrObj, 8>,
    user_to_blocking_uses: HashMap<Ptr<Operation>, SmallSet<Use<Value>, 4>>,
    accessors: Vec<TraitOp<dyn DestructurableAccessorOpInterface>>,
}

fn compute_destructuring_info(
    ctx: &Context,
    value: &DestructurableValueSlot,
) -> Option<ValueDestructuringInfo> {
    if !value.slot.value.is_used(ctx) {
        return None;
    }

    let mut info = ValueDestructuringInfo::default();
    let mut used_safely_worklist = Vec::new();

    let mut schedule_as_blocking_use = |r#use: Use<Value>| {
        let blocking_uses = info
            .user_to_blocking_uses
            .entry(r#use.user_op())
            .or_default();
        blocking_uses.insert(r#use);
    };

    for r#use in value.slot.value.uses(ctx) {
        if let Some(accessor) =
            TraitOp::<dyn DestructurableAccessorOpInterface>::try_from_op(r#use.user_op(), ctx)
            && accessor.can_rewire(
                ctx,
                value,
                &mut info.used_indices,
                &mut used_safely_worklist,
            )
        {
            info.accessors.push_back(accessor);
            continue;
        }

        schedule_as_blocking_use(r#use);
    }

    let mut visited = SmallSet::<_, 16>::new();
    while let Some(must_be_used_safely) = used_safely_worklist.pop() {
        for subvalue_use in must_be_used_safely.value.uses(ctx) {
            if !visited.insert(subvalue_use) {
                continue;
            }
            let subvalue_owner = subvalue_use.user_op().dyn_op(ctx);

            if let Some(mem_op) = op_cast::<dyn SafeMemorySlotAccessOpInterface>(&*subvalue_owner)
                && let Ok(()) = mem_op.ensure_only_safe_accesses(
                    ctx,
                    &must_be_used_safely,
                    &mut used_safely_worklist,
                )
            {
                continue;
            }

            schedule_as_blocking_use(subvalue_use);
        }
    }

    let forward_slice = get_value_forward_slice(ctx, value.slot.value);
    for user in forward_slice {
        let Some(blocking_uses) = info.user_to_blocking_uses.get(&user) else {
            continue;
        };
        if blocking_uses.is_empty() {
            continue;
        }
        return None;
        // Should check if (memory) uses can be erased, but the mem2reg interface is too restrictive
        // right now. Get back to this later.
    }

    Some(info)
}

fn destructure_value(
    ctx: &mut Context,
    value: &DestructurableValueSlot,
    constructor: &TraitOp<dyn DestructurableConstructorOpInterface>,
    rewriter: &mut PassRewriter,
    info: &ValueDestructuringInfo,
    new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
) {
    rewriter.set_insertion_point_before_operation(constructor.get_operation());
    let subvalues =
        constructor.destructure(ctx, value, &info.used_indices, rewriter, new_constructors);

    let mut users_to_rewire = ISet::default();
    users_to_rewire.extend(info.user_to_blocking_uses.keys().copied());
    users_to_rewire.extend(info.accessors.iter().map(|it| it.get_operation()));
    // TODO: Topo sort, may not be needed for structured IR

    let mut to_erase = Vec::new();
    for &to_rewire in users_to_rewire.iter().rev() {
        rewriter.set_insertion_point_after_operation(to_rewire);
        let to_rewire_dyn = to_rewire.dyn_op(ctx);
        if let Some(accessor) = op_cast::<dyn DestructurableAccessorOpInterface>(&*to_rewire_dyn) {
            if accessor.rewire(ctx, value, &subvalues, rewriter) == DeletionKind::Delete {
                to_erase.push(to_rewire);
            }
            continue;
        }

        // TODO: Erasable memory ops, the interface is too limited right now and we don't have any
        // anyways. Should implement this eventually though.
    }

    for to_erase_op in to_erase {
        rewriter.erase_operation(ctx, to_erase_op);
    }

    let new_constructor = constructor.handle_destructuring_complete(ctx, value, rewriter);
    new_constructors.extend(new_constructor);
}

fn try_to_destructure_values(
    ctx: &mut Context,
    constructors: &[TraitOp<dyn DestructurableConstructorOpInterface>],
    rewriter: &mut PassRewriter,
) -> bool {
    let mut destructured_any = false;

    let mut worklist = constructors.to_vec();
    let mut new_worklist = Vec::with_capacity(worklist.len());

    loop {
        let mut changes_in_this_round = false;

        for constructor in worklist.iter() {
            let mut destructured_any_value = false;
            for value in constructor.destructurable_values(ctx) {
                let Some(info) = compute_destructuring_info(ctx, &value) else {
                    continue;
                };

                destructure_value(ctx, &value, constructor, rewriter, &info, &mut new_worklist);
                destructured_any_value = true;

                break;
            }
            if !destructured_any_value {
                new_worklist.push(constructor.clone());
            }
            changes_in_this_round |= destructured_any_value;
        }

        if !changes_in_this_round {
            break;
        }
        destructured_any |= changes_in_this_round;

        core::mem::swap(&mut worklist, &mut new_worklist);
        new_worklist.clear();
    }

    destructured_any
}

pub struct SROAPass;

#[pass_name]
impl Pass for SROAPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        for region in op.regions(ctx) {
            if region.deref(ctx).iter(ctx).count() == 0 {
                continue;
            }

            let mut rewriter = PassRewriter::default();
            let mut constructors = Vec::new();
            walk_region(
                ctx,
                &mut constructors,
                &WALKCONFIG_ANY,
                region,
                |ctx, constructors, op| {
                    if let IRNode::Operation(op) = op
                        && let Some(constructor) = TraitOp::try_from_op(op, ctx)
                    {
                        constructors.push(constructor);
                    }
                },
            );

            if try_to_destructure_values(ctx, &constructors, &mut rewriter) {
                res.ir_changed |= IRStatus::Changed;
            }
        }

        Ok(res)
    }
}
