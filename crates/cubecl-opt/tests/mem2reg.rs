//! Mem2reg on the shape a fully unrolled register kernel lowers to: many local
//! variables, each loaded and stored many times, in one straight-line block.

use cubecl_ir::{
    AddressSpace,
    dialect::memory::{DeclareVariableOp, LoadOp, StoreOp},
    interfaces::TypedExt,
    prelude::*,
};
use cubecl_opt::passes::mem2reg::Mem2RegPass;
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    init_env_logger_for_tests,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    pass::{AnalysisManager, Pass},
    result::ExpectOk,
};

/// A function whose entry block declares `variables` locals, then `rounds`
/// times loads each one and stores it into its neighbour.
fn unrolled_registers(ctx: &mut Context, variables: usize, rounds: usize) -> Ptr<Operation> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i32) -> ()> [] {
      ^entry(x: builtin.integer i32):
      branch.return
    }
  "#;
    let func = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    let entry = entry_block(ctx, func);
    let terminator = entry.deref(ctx).get_terminator(ctx).unwrap();
    let x = entry.deref(ctx).get_argument(0);
    let ty = x.get_type(ctx);

    let slots: Vec<Value> = (0..variables)
        .map(|_| {
            let align = x.align(ctx);
            let declare = DeclareVariableOp::new(ctx, ty, AddressSpace::Local, align, None);
            declare.get_operation().insert_before(ctx, terminator);
            let slot = declare.get_result(ctx);
            StoreOp::new(ctx, slot, x)
                .get_operation()
                .insert_before(ctx, terminator);
            slot
        })
        .collect();

    for _ in 0..rounds {
        for (i, &slot) in slots.iter().enumerate() {
            let load = LoadOp::new(ctx, slot);
            load.get_operation().insert_before(ctx, terminator);
            let value = load.get_result(ctx);
            StoreOp::new(ctx, slots[(i + 1) % variables], value)
                .get_operation()
                .insert_before(ctx, terminator);
        }
    }
    func
}

fn entry_block(ctx: &Context, func: Ptr<Operation>) -> Ptr<BasicBlock> {
    let region = func.deref(ctx).get_region(0);
    region.deref(ctx).get_head().unwrap()
}

fn count_ops(ctx: &Context, func: Ptr<Operation>, name: &str) -> usize {
    entry_block(ctx, func)
        .deref(ctx)
        .iter(ctx)
        .filter(|op| Operation::get_opid(*op, ctx).to_string() == name)
        .count()
}

#[test]
fn promotes_every_register_of_an_unrolled_block() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    // Large enough that a pass quadratic in the block's length takes seconds
    // here rather than milliseconds.
    let (variables, rounds) = (2048, 8);
    let func = unrolled_registers(ctx, variables, rounds);

    Mem2RegPass
        .run(func, ctx, &mut AnalysisManager::default())
        .expect_ok(ctx);

    verify_operation(func, ctx).expect_ok(ctx);
    for name in ["memory.load", "memory.store", "memory.declare_variable"] {
        assert_eq!(count_ops(ctx, func, name), 0, "{name} left after mem2reg");
    }
}
