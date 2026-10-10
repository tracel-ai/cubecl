//! Shared memories share bytes only where no unit can still reach one while another writes the
//! other: one dead before the other is born, and a barrier every unit reaches in between. Each
//! test builds a kernel around shared arrays and reads where the allocation placed them.

use cubecl_ir::{
    AddressType, OpInserter, Scope,
    dialect::{
        MemoryClobbers,
        asm::InlineAsmOp,
        barrier::{ArriveAndWaitOp, CopyAsyncOp},
        branch::{ConditionOp, IfOp, ReturnOp, WhileOp},
        general::SelectOp,
        memory::{IndexOp, StoreOp},
        synchronization::{SyncOp, SyncScope},
        tma::{CommitGroupOp, TmaStoreOp, WaitGroupReadOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::{
        ArrayType,
        barrier::{BarrierLevel, BarrierType},
        scalar::IndexType,
    },
};
use cubecl_opt::SharedLiveness;
use pliron::{builtin::ops::FuncOp, context::Context, op::Op, pass::AnalysisManager, value::Value};

const LENGTH: usize = 64;

#[test]
fn memories_a_barrier_separates_share_their_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// Deriving a pointer touches no memory: `b` is born where the store through it lands, after the
/// barrier, however early the pointer was taken.
#[test]
fn a_pointer_derived_before_the_barrier_is_not_an_access() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    let into_b = element(&scope, b);
    touch(&scope, a);
    barrier(&scope, SyncScope::Cube);
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
    barrier(&scope, SyncScope::Cube);
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
        barrier(body, SyncScope::Cube);
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
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// A barrier some units may skip separates nothing.
#[test]
fn a_barrier_inside_a_branch_does_not_separate() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    in_branch(&scope, |then| barrier(then, SyncScope::Cube));
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// A device barrier waits for the whole cube too.
#[test]
fn a_device_barrier_separates() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope, SyncScope::Device);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// A plane barrier leaves the other planes free to reach `a`.
#[test]
fn a_plane_barrier_does_not_separate() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope, SyncScope::Plane);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// Units that write `a` and return never reach the barrier, so their writes are not ordered
/// before the other units' writes to `b`.
#[test]
fn a_memory_an_exiting_unit_wrote_keeps_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    in_exiting_branch(&scope, |then| touch(then, a));
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// Units that return before `a` is born never touch it: the barrier still separates.
#[test]
fn an_exit_before_a_memory_is_born_leaves_it_separable() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    in_exiting_branch(&scope, |_| {});
    touch(&scope, a);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// `copy_async` writes `a` after the operation that issues it, and no cube barrier waits for it.
#[test]
fn a_memory_an_async_copy_writes_keeps_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    let source = {
        let array = index_array(&scope, LENGTH);
        scope.create_local_mut(array, None)
    };
    let length = scope.const_usize(16);
    let op = CopyAsyncOp::new(scope.ctx_mut(), source, a, length, 16, false);
    scope.register(&op);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// A TMA store reads `a` until `wait_group_read`, which names no memory.
#[test]
fn a_memory_a_tma_store_reads_keeps_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    let map = scope.const_usize(0);
    let coordinate = scope.const_usize(0);
    let op = TmaStoreOp::new(scope.ctx_mut(), a, map, vec![coordinate]);
    scope.register(&op);
    scope.register(&CommitGroupOp::new(scope.ctx_mut()));
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    scope.register(&WaitGroupReadOp::new(scope.ctx_mut(), 0));
    assert!(!share_bytes(scope, a, b));
}

/// Nothing invalidates a barrier object, so its bytes are never another memory's.
#[test]
fn a_barrier_object_keeps_its_own_bytes() {
    let scope = kernel();
    let barrier_object = {
        let ty = BarrierType::get(scope.ctx(), BarrierLevel::Cube);
        scope.create_shared(ty, None)
    };
    let b = shared(&scope);
    let op = ArriveAndWaitOp::new(scope.ctx_mut(), barrier_object);
    scope.register(&op);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(!share_bytes(scope, barrier_object, b));
}

/// Inline assembly that may touch memory can reach shared memory through an address no pointer
/// names.
#[test]
fn inline_assembly_touching_memory_keeps_every_memory_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    touch(&scope, a);
    barrier(&scope, SyncScope::Cube);
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
    barrier(&scope, SyncScope::Cube);
    asm(&scope, MemoryClobbers::Nomem);
    touch(&scope, b);
    assert!(share_bytes(scope, a, b));
}

/// A pointer a `select` chooses hides which memory it points into: every memory keeps its own
/// bytes.
#[test]
fn a_selected_pointer_keeps_every_memory_its_own_bytes() {
    let scope = kernel();
    let (a, b) = (shared(&scope), shared(&scope));
    let chosen = {
        let condition = scope.const_bool(true);
        let op = SelectOp::new(scope.ctx_mut(), condition, a, a);
        scope.register_with_result(&op)
    };
    touch(&scope, chosen);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    assert!(!share_bytes(scope, a, b));
}

/// A memory placed on reused bytes keeps its own alignment, and one live beside two others lands
/// past both.
#[test]
fn a_reused_offset_keeps_the_memory_aligned() {
    let scope = kernel();
    let a = shared(&scope);
    let b = {
        let array = index_array(&scope, 3);
        scope.create_shared(array, None)
    };
    let c = {
        let array = index_array(&scope, LENGTH);
        scope.create_shared(array, Some(1024))
    };
    touch(&scope, a);
    touch(&scope, b);
    barrier(&scope, SyncScope::Cube);
    touch(&scope, b);
    touch(&scope, c);
    let (ctx, liveness) = allocate(scope);
    assert_eq!(liveness.allocations[&c].offset % 1024, 0);
    assert!(!overlap(&ctx, &liveness, b, c));
}

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 32, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

fn index_array(scope: &Scope, length: usize) -> pliron::r#type::TypeHandle {
    let ctx = scope.ctx_mut();
    let inner = IndexType::get(ctx).into();
    ArrayType::get(ctx, inner, length).to_handle()
}

/// A shared array of `LENGTH` indices.
fn shared(scope: &Scope) -> Value {
    let array = index_array(scope, LENGTH);
    scope.create_shared(array, None)
}

/// A pointer to element 0 of `memory`.
fn element(scope: &Scope, memory: Value) -> Value {
    let index = scope.const_usize(0);
    let op = IndexOp::new(scope.ctx_mut(), memory, index, None);
    scope.register_with_result(&op)
}

/// Write element 0 of `memory`.
fn touch(scope: &Scope, memory: Value) {
    let pointer = element(scope, memory);
    let value = scope.const_usize(1);
    let op = StoreOp::new(scope.ctx_mut(), pointer, value);
    scope.register(&op);
}

fn barrier(scope: &Scope, sync_scope: SyncScope) {
    let op = SyncOp::new(scope.ctx_mut(), sync_scope);
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
    branch(scope, body, Scope::terminate_yield);
}

/// `body` on one side of a branch whose units then leave the kernel.
fn in_exiting_branch(scope: &Scope, body: impl FnOnce(&Scope)) {
    branch(scope, body, |then| {
        then.register(&ReturnOp::new(then.ctx_mut()))
    });
}

/// `body` on the `then` side of a branch, closed by `end`; the `else` side is empty.
fn branch(scope: &Scope, body: impl FnOnce(&Scope), end: impl FnOnce(&Scope)) {
    let cond = scope.const_bool(true);
    let if_op = IfOp::new(scope.ctx_mut(), cond);
    let then = scope.child(OpInserter::new_at_block_end(if_op.then_block(scope.ctx())));
    body(&then);
    end(&then);
    let otherwise = scope.child(OpInserter::new_at_block_end(if_op.else_block(scope.ctx())));
    otherwise.terminate_yield();
    scope.register(&if_op);
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

/// Where the allocation placed every shared memory of `scope`'s kernel.
fn allocate(scope: Scope) -> (Context, SharedLiveness) {
    let entry: FuncOp = scope.state().entry_func;
    let ctx: Context = scope.into_context().expect("the scope owns its context");
    let mut analyses = AnalysisManager::default();
    let liveness = analyses
        .get_analysis::<SharedLiveness>(entry.get_operation(), &ctx)
        .expect("the analysis computes")
        .clone();
    (ctx, liveness)
}

/// Whether `liveness` placed `a` and `b` on overlapping bytes.
fn overlap(ctx: &Context, liveness: &SharedLiveness, a: Value, b: Value) -> bool {
    let (a, b) = (liveness.allocations[&a], liveness.allocations[&b]);
    a.offset < b.end(ctx) && b.offset < a.end(ctx)
}

/// Whether the allocation placed `a` and `b` on overlapping bytes.
fn share_bytes(scope: Scope, a: Value, b: Value) -> bool {
    let (ctx, liveness) = allocate(scope);
    overlap(&ctx, &liveness, a, b)
}
