//! Shared memory lives from its declaration until it is freed: a declaration after a `free`, in
//! program order, may take the freed bytes, and one beside a live allocation never overlaps it.

use cubecl_ir::{
    AddressType, OpInserter, Scope,
    dialect::{branch::IfOp, general::FreeOp, vector::CompositeConstructOp},
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::{ArrayType, aggregate::SliceType, scalar::IndexType},
};
use cubecl_opt::SharedLiveness;
use pliron::{
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    pass::AnalysisManager,
    r#type::Typed,
    value::Value,
};

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

/// A shared array of `len` indices.
fn shared(scope: &Scope, len: usize) -> Value {
    let inner = IndexType::get(scope.ctx_mut()).to_handle();
    let ty = ArrayType::get(scope.ctx_mut(), inner, len);
    scope.create_shared(ty, None)
}

/// Free `memory` as the frontend does: through a slice over it and its pointer field.
fn free_slice(scope: &Scope, memory: Value, len: usize) {
    let list_ty = memory.get_type(scope.ctx());
    let ty = SliceType::get(scope.ctx_mut(), list_ty).to_handle();
    let (offset, length) = (scope.const_usize(0), scope.const_usize(len));
    let op = CompositeConstructOp::new(scope.ctx_mut(), ty, vec![memory, offset, length]);
    let slice = scope.register_with_result(&op);
    let pointer = scope.extract_field(slice, 0);
    scope.register(&FreeOp::new(scope.ctx_mut(), pointer));
}

/// An `if` whose arms `then` and `otherwise` build, as the frontend lowers a runtime condition.
fn branch(scope: &Scope, then: impl FnOnce(&Scope), otherwise: impl FnOnce(&Scope)) {
    let condition = scope.const_bool(true);
    let if_op = IfOp::new(scope.ctx_mut(), condition);
    let then_child = scope.child(OpInserter::new_at_block_end(if_op.then_block(scope.ctx())));
    then(&then_child);
    then_child.terminate_yield();
    let else_child = scope.child(OpInserter::new_at_block_end(if_op.else_block(scope.ctx())));
    otherwise(&else_child);
    else_child.terminate_yield();
    scope.register(&if_op);
}

fn offsets(scope: Scope, of: &[Value]) -> Vec<usize> {
    let module_op: Ptr<Operation> = scope.state().module.get_operation();
    let ctx: Context = scope.into_context().expect("the scope owns its context");
    let mut analyses = AnalysisManager::default();
    let liveness = analyses
        .get_analysis::<SharedLiveness>(module_op, &ctx)
        .expect("the analysis computes");
    of.iter()
        .map(|value| liveness.allocations[value].offset)
        .collect()
}

#[test]
fn a_declaration_after_a_free_takes_the_freed_bytes() {
    let scope = kernel();
    let first = shared(&scope, 256);
    free_slice(&scope, first, 256);
    let second = shared(&scope, 128);
    assert_eq!(offsets(scope, &[first, second]), [0, 0]);
}

#[test]
fn live_allocations_never_overlap() {
    let scope = kernel();
    let first = shared(&scope, 256);
    let second = shared(&scope, 128);
    let [first, second] = offsets(scope, &[first, second])[..] else {
        unreachable!()
    };
    assert_eq!(first, 0);
    assert!(
        second >= 256 * 4,
        "the second starts at {second}, inside the first"
    );
}

/// Only what was freed is reused: a third declaration beside one still live lands past it.
#[test]
fn a_free_releases_only_its_own_allocation() {
    let scope = kernel();
    let freed = shared(&scope, 64);
    let kept = shared(&scope, 64);
    free_slice(&scope, freed, 64);
    let reused = shared(&scope, 64);
    let [freed, kept, reused] = offsets(scope, &[freed, kept, reused])[..] else {
        unreachable!()
    };
    assert_eq!(reused, freed);
    assert_ne!(reused, kept);
}

/// A free in one arm of a branch is no free on the other arm's path: what the other arm declares
/// does not take the bytes.
#[test]
fn a_free_in_one_arm_releases_nothing_to_the_other() {
    let scope = kernel();
    let freed = shared(&scope, 64);
    let mut other = None;
    branch(
        &scope,
        |then| free_slice(then, freed, 64),
        |otherwise| other = Some(shared(otherwise, 64)),
    );
    let other = other.expect("the else arm declares");
    let [freed, other] = offsets(scope, &[freed, other])[..] else {
        unreachable!()
    };
    assert_ne!(freed, other);
}

/// Past the branch that freed it, the memory is live again: the paths through the other arm
/// never freed it.
#[test]
fn a_free_in_a_branch_ends_with_the_branch() {
    let scope = kernel();
    let freed = shared(&scope, 64);
    branch(&scope, |then| free_slice(then, freed, 64), |_| {});
    let after = shared(&scope, 64);
    let [freed, after] = offsets(scope, &[freed, after])[..] else {
        unreachable!()
    };
    assert_ne!(freed, after);
}

/// Within the arm that freed it, what follows the free takes the bytes.
#[test]
fn a_free_releases_its_bytes_to_what_follows_it_in_its_block() {
    let scope = kernel();
    let freed = shared(&scope, 64);
    let mut reused = None;
    branch(
        &scope,
        |then| {
            free_slice(then, freed, 64);
            reused = Some(shared(then, 64));
        },
        |_| {},
    );
    let reused = reused.expect("the then arm declares");
    assert_eq!(offsets(scope, &[freed, reused]), [0, 0]);
}
