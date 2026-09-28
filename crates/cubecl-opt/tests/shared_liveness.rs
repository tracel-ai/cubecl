//! Shared memories share bytes only where no unit can still reach one while another writes the
//! other: one dead before the other is born, and a barrier every unit reaches in between. Each
//! test builds a kernel around two shared arrays and reads where the allocation placed them.

use cubecl_ir::{
    AddressType, OpInserter, Scope,
    dialect::{
        MemoryClobbers,
        asm::InlineAsmOp,
        branch::{ConditionOp, IfOp, WhileOp},
        memory::{IndexOp, StoreOp},
        synchronization::{SyncOp, SyncScope},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::{ArrayType, scalar::IndexType},
};
use cubecl_opt::SharedLiveness;
use pliron::{builtin::ops::FuncOp, context::Context, op::Op, pass::AnalysisManager, value::Value};

const LENGTH: usize = 64;

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 32, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

/// A shared array of `LENGTH` indices.
fn shared(scope: &Scope) -> Value {
    let ctx = scope.ctx_mut();
    let inner = IndexType::get(ctx).into();
    let array = ArrayType::get(ctx, inner, LENGTH).to_handle();
    scope.create_shared(array, None)
}

/// Write element 0 of `memory`.
fn touch(scope: &Scope, memory: Value) {
    let index = scope.const_usize(0);
    let ptr = {
        let op = IndexOp::new(scope.ctx_mut(), memory, index, None);
        scope.register_with_result(&op)
    };
    let value = scope.const_usize(1);
    let op = StoreOp::new(scope.ctx_mut(), ptr, value);
    scope.register(&op);
}

fn barrier(scope: &Scope) {
    let op = SyncOp::new(scope.ctx_mut(), SyncScope::Cube);
    scope.register(&op);
}

/// `body` inside a loop that runs while a runtime flag holds.
fn in_loop(scope: &Scope, body: impl FnOnce(&Scope)) {
    let while_op = WhileOp::new(scope.ctx_mut());
    let cond = scope.child(OpInserter::new_at_block_start(
        while_op.before_block(scope.ctx()),
    ));
    let flag = cond.const_bool(false);
    cond.register(&ConditionOp::new(scope.ctx_mut(), flag));
    let inside = scope.child(OpInserter::new_at_block_end(
        while_op.after_block(scope.ctx()),
    ));
    body(&inside);
    inside.terminate_yield();
    scope.register(&while_op);
}

/// `body` on one side of a branch.
fn in_branch(scope: &Scope, body: impl FnOnce(&Scope)) {
    let cond = scope.const_bool(true);
    let if_op = IfOp::new(scope.ctx_mut(), cond);
    let then = scope.child(OpInserter::new_at_block_end(if_op.then_block(scope.ctx())));
    body(&then);
    then.terminate_yield();
    let otherwise = scope.child(OpInserter::new_at_block_end(if_op.else_block(scope.ctx())));
    otherwise.terminate_yield();
    scope.register(&if_op);
}

/// Whether the allocation placed `a` and `b` on overlapping bytes.
fn share_bytes(scope: Scope, a: Value, b: Value) -> bool {
    let entry: FuncOp = scope.state().entry_func;
    let ctx: Context = scope.into_context().expect("the scope owns its context");
    let mut analyses = AnalysisManager::default();
    let liveness = analyses
        .get_analysis::<SharedLiveness>(entry.get_operation(), &ctx)
        .expect("the analysis computes");
    let (a, b) = (liveness.allocations[&a], liveness.allocations[&b]);
    a.offset < b.end(&ctx) && b.offset < a.end(&ctx)
}

#[test]
fn memories_a_barrier_separates_share_their_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// Deriving a pointer touches no memory: `b` is born where the store through it lands, after the
/// barrier, however early the pointer was taken.
#[test]
fn a_pointer_derived_before_the_barrier_is_not_an_access() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    let index = scope.const_usize(0);
    let into_b = {
        let op = IndexOp::new(scope.ctx_mut(), b, index, None);
        scope.register_with_result(&op)
    };
    touch(&scope, a);
    barrier(&scope);
    let value = scope.const_usize(1);
    let op = StoreOp::new(scope.ctx_mut(), into_b, value);
    scope.register(&op);
    assert!(share_bytes(scope, a, b));
}

/// Per unit, `a` is dead before `b` is born; but another unit may still be writing `a`.
#[test]
fn memories_no_barrier_separates_keep_their_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// The declaration is not an access: `b` is declared first and born after the barrier.
#[test]
fn a_memory_is_born_at_its_first_access_not_its_declaration() {
    let scope = kernel();
    let (b, a) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// The next iteration reaches `a` again after `b`: the loop keeps both alive throughout.
#[test]
fn an_access_inside_a_loop_keeps_the_memory_live_for_the_whole_loop() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    in_loop(&scope, |body| {
        touch(body, a);
        barrier(body);
        touch(body, b);
    });
    assert!(!share_bytes(scope, a, b));
}

/// After the loop has ended, a barrier separates what it touched from what follows.
#[test]
fn a_memory_touched_only_inside_a_loop_is_dead_after_it() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    in_loop(&scope, |body| touch(body, a));
    barrier(&scope);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// A barrier some units may skip separates nothing.
#[test]
fn a_barrier_inside_a_branch_does_not_separate() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    in_branch(&scope, barrier);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// Inline assembly that may touch memory can reach shared memory through an address no pointer
/// names.
#[test]
fn inline_assembly_touching_memory_keeps_every_memory_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope);
    asm(&scope, MemoryClobbers::ReadWrite);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// Inline assembly that touches no memory reaches none of it.
#[test]
fn inline_assembly_touching_no_memory_changes_nothing() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope);
    asm(&scope, MemoryClobbers::Nomem);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

fn asm(scope: &Scope, clobbers: MemoryClobbers) {
    let op = InlineAsmOp::new(
        scope.ctx_mut(),
        vec![],
        vec![],
        "".into(),
        clobbers,
        vec![],
        vec![],
    );
    scope.register(&op);
}
