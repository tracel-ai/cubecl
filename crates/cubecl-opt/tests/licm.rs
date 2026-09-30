use cubecl_ir::{
    AddressType, OpInserter, Scope,
    dialect::{
        math::{IAddOp, IMulOp},
        memory::StoreOp,
        scf::ForOp,
    },
    prelude::*,
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::scalar::IndexType,
};
use cubecl_opt::LoopInvariantCodeMotionPass;
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::verify_operation,
    pass::{AnalysisManager, Pass},
};

fn create_kernel_scope() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

fn finish(scope: Scope) -> (Ptr<Operation>, FuncOp, Context) {
    let module_op = scope.state().module.get_operation();
    let entry_func = scope.state().entry_func;
    let ctx = scope.into_context().expect("the scope owns its context");
    (module_op, entry_func, ctx)
}

fn op_index_in_block(
    block: Ptr<BasicBlock>,
    target: Ptr<Operation>,
    ctx: &Context,
) -> Option<usize> {
    block.deref(ctx).iter(ctx).position(|op| op == target)
}

#[test]
fn test_licm_pure_arithmetic() {
    let scope = create_kernel_scope();
    let a = scope.const_usize(10);
    let b = scope.const_usize(20);
    let start = scope.const_usize(0);
    let end = scope.const_usize(100);
    let step = scope.const_usize(1);

    let for_op = ForOp::new(scope.ctx_mut(), vec![], start, end, step, vec![]);
    scope.register(&for_op);
    let body_block = for_op.loop_body(scope.ctx());
    let c_op = {
        let loop_scope = scope.child(OpInserter::new_at_block_end(body_block));
        let c_op = IAddOp::new(loop_scope.ctx_mut(), a, b);
        let _c_val = loop_scope.register_with_result(&c_op);
        loop_scope.terminate_yield();
        c_op
    };

    let (module_op, entry_func, mut ctx) = finish(scope);
    let entry_block = entry_func.get_entry_block(&ctx);
    let for_op_ptr = for_op.get_operation();
    let c_op_ptr = c_op.get_operation();

    // Verify initial positions
    assert_eq!(c_op_ptr.deref(&ctx).get_parent_block(), Some(body_block));
    assert_eq!(for_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));

    // Run LICM pass
    let mut pass = LoopInvariantCodeMotionPass;
    let mut analyses = AnalysisManager::default();
    let pass_result = pass.run(module_op, &mut ctx, &mut analyses).unwrap();
    assert_eq!(pass_result.ir_changed, IRStatus::Changed);

    // Verify IR integrity after LICM
    verify_operation(module_op, &ctx).expect("IR should be valid after LICM");

    // Verify c_op is hoisted to entry_block before for_op
    assert_eq!(c_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));
    let c_idx = op_index_in_block(entry_block, c_op_ptr, &ctx).unwrap();
    let for_idx = op_index_in_block(entry_block, for_op_ptr, &ctx).unwrap();
    assert!(c_idx < for_idx, "c should be hoisted before for_op");
}

#[test]
fn test_licm_chained_dependencies() {
    let scope = create_kernel_scope();
    let a = scope.const_usize(10);
    let start = scope.const_usize(0);
    let end = scope.const_usize(100);
    let step = scope.const_usize(1);

    let for_op = ForOp::new(scope.ctx_mut(), vec![], start, end, step, vec![]);
    scope.register(&for_op);
    let body_block = for_op.loop_body(scope.ctx());
    let (b_op, c_op) = {
        let loop_scope = scope.child(OpInserter::new_at_block_end(body_block));
        let one = loop_scope.const_usize(1);
        let b_op = IAddOp::new(loop_scope.ctx_mut(), a, one);
        let b_val = loop_scope.register_with_result(&b_op);

        let two = loop_scope.const_usize(2);
        let c_op = IMulOp::new(loop_scope.ctx_mut(), b_val, two);
        let _c_val = loop_scope.register_with_result(&c_op);
        loop_scope.terminate_yield();
        (b_op, c_op)
    };

    let (module_op, entry_func, mut ctx) = finish(scope);
    let entry_block = entry_func.get_entry_block(&ctx);
    let for_op_ptr = for_op.get_operation();
    let b_op_ptr = b_op.get_operation();
    let c_op_ptr = c_op.get_operation();

    // Run LICM pass
    let mut pass = LoopInvariantCodeMotionPass;
    let mut analyses = AnalysisManager::default();
    let pass_result = pass.run(module_op, &mut ctx, &mut analyses).unwrap();
    assert_eq!(pass_result.ir_changed, IRStatus::Changed);

    // Verify IR integrity after LICM
    verify_operation(module_op, &ctx).expect("IR should be valid after LICM");

    // Verify both are hoisted before for_op in exact topological order: b then c
    assert_eq!(b_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));
    assert_eq!(c_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));
    let b_idx = op_index_in_block(entry_block, b_op_ptr, &ctx).unwrap();
    let c_idx = op_index_in_block(entry_block, c_op_ptr, &ctx).unwrap();
    let for_idx = op_index_in_block(entry_block, for_op_ptr, &ctx).unwrap();

    assert!(
        b_idx < c_idx,
        "b must be hoisted before c (topological order)"
    );
    assert!(c_idx < for_idx, "c must be hoisted before for_op");
}

#[test]
fn test_licm_variant_remains() {
    let scope = create_kernel_scope();
    let a = scope.const_usize(10);
    let start = scope.const_usize(0);
    let end = scope.const_usize(100);
    let step = scope.const_usize(1);

    let for_op = ForOp::new(scope.ctx_mut(), vec![], start, end, step, vec![]);
    scope.register(&for_op);
    let body_block = for_op.loop_body(scope.ctx());
    let x_op = {
        let loop_scope = scope.child(OpInserter::new_at_block_end(body_block));
        let iv = for_op.iter_var(loop_scope.ctx());
        let x_op = IAddOp::new(loop_scope.ctx_mut(), iv, a);
        let _x_val = loop_scope.register_with_result(&x_op);
        loop_scope.terminate_yield();
        x_op
    };

    let (module_op, entry_func, mut ctx) = finish(scope);
    let entry_block = entry_func.get_entry_block(&ctx);
    let x_op_ptr = x_op.get_operation();

    // Run LICM pass
    let mut pass = LoopInvariantCodeMotionPass;
    let mut analyses = AnalysisManager::default();
    let pass_result = pass.run(module_op, &mut ctx, &mut analyses).unwrap();
    assert_eq!(pass_result.ir_changed, IRStatus::Unchanged);

    // Verify IR integrity after LICM
    verify_operation(module_op, &ctx).expect("IR should be valid after LICM");

    // Verify x remains inside the loop body block
    assert_eq!(x_op_ptr.deref(&ctx).get_parent_block(), Some(body_block));
    assert_ne!(x_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));
}

#[test]
fn test_licm_side_effects_remain() {
    let scope = create_kernel_scope();
    let val_ty = IndexType::get(scope.ctx_mut()).to_handle();
    let ptr = scope.create_local_mut(val_ty, None);
    let val = scope.const_usize(42);
    let start = scope.const_usize(0);
    let end = scope.const_usize(100);
    let step = scope.const_usize(1);

    let for_op = ForOp::new(scope.ctx_mut(), vec![], start, end, step, vec![]);
    scope.register(&for_op);
    let body_block = for_op.loop_body(scope.ctx());
    let store_op = {
        let loop_scope = scope.child(OpInserter::new_at_block_end(body_block));
        let store_op = StoreOp::new(loop_scope.ctx_mut(), ptr, val);
        loop_scope.register(&store_op);
        loop_scope.terminate_yield();
        store_op
    };

    let (module_op, entry_func, mut ctx) = finish(scope);
    let entry_block = entry_func.get_entry_block(&ctx);
    let store_op_ptr = store_op.get_operation();

    // Run LICM pass
    let mut pass = LoopInvariantCodeMotionPass;
    let mut analyses = AnalysisManager::default();
    let pass_result = pass.run(module_op, &mut ctx, &mut analyses).unwrap();
    assert_eq!(pass_result.ir_changed, IRStatus::Unchanged);

    // Verify IR integrity after LICM
    verify_operation(module_op, &ctx).expect("IR should be valid after LICM");

    // Verify store_op remains inside the loop body block
    assert_eq!(
        store_op_ptr.deref(&ctx).get_parent_block(),
        Some(body_block)
    );
    assert_ne!(
        store_op_ptr.deref(&ctx).get_parent_block(),
        Some(entry_block)
    );
}

#[test]
fn test_licm_nested_loops() {
    let scope = create_kernel_scope();
    let a = scope.const_usize(10);
    let b = scope.const_usize(20);
    let start1 = scope.const_usize(0);
    let end1 = scope.const_usize(100);
    let step1 = scope.const_usize(1);

    let for_op1 = ForOp::new(scope.ctx_mut(), vec![], start1, end1, step1, vec![]);
    scope.register(&for_op1);
    let body_block1 = for_op1.loop_body(scope.ctx());

    let (for_op2, body_block2, c_op) = {
        let loop_scope1 = scope.child(OpInserter::new_at_block_end(body_block1));
        let start2 = loop_scope1.const_usize(0);
        let end2 = loop_scope1.const_usize(50);
        let step2 = loop_scope1.const_usize(1);
        let for_op2 = ForOp::new(loop_scope1.ctx_mut(), vec![], start2, end2, step2, vec![]);
        loop_scope1.register(&for_op2);
        let body_block2 = for_op2.loop_body(loop_scope1.ctx());

        let c_op = {
            let loop_scope2 = loop_scope1.child(OpInserter::new_at_block_end(body_block2));
            // c = a + b inside loop 2 (invariant to both loop 1 and loop 2)
            let c_op = IAddOp::new(loop_scope2.ctx_mut(), a, b);
            let _c_val = loop_scope2.register_with_result(&c_op);
            loop_scope2.terminate_yield();
            c_op
        };
        loop_scope1.terminate_yield();
        (for_op2, body_block2, c_op)
    };

    let (module_op, entry_func, mut ctx) = finish(scope);
    let entry_block = entry_func.get_entry_block(&ctx);
    let for_op1_ptr = for_op1.get_operation();
    let c_op_ptr = c_op.get_operation();

    // Run LICM pass
    let mut pass = LoopInvariantCodeMotionPass;
    let mut analyses = AnalysisManager::default();
    let pass_result = pass.run(module_op, &mut ctx, &mut analyses).unwrap();
    assert_eq!(pass_result.ir_changed, IRStatus::Changed);

    // Verify IR integrity after LICM
    verify_operation(module_op, &ctx).expect("IR should be valid after LICM");

    // Verify c_op is hoisted all the way into entry_block before for_op1
    assert_eq!(c_op_ptr.deref(&ctx).get_parent_block(), Some(entry_block));
    assert_ne!(c_op_ptr.deref(&ctx).get_parent_block(), Some(body_block1));
    assert_ne!(c_op_ptr.deref(&ctx).get_parent_block(), Some(body_block2));

    // Verify for_op2 remains inside body_block1
    assert_eq!(
        for_op2.get_operation().deref(&ctx).get_parent_block(),
        Some(body_block1)
    );

    let c_idx = op_index_in_block(entry_block, c_op_ptr, &ctx).unwrap();
    let for1_idx = op_index_in_block(entry_block, for_op1_ptr, &ctx).unwrap();
    assert!(
        c_idx < for1_idx,
        "c should be hoisted before the outermost loop for_op1"
    );
}
