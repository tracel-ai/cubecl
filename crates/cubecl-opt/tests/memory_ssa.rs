use cubecl_ir::{
    dialect::memory::AddressSpaceAttr,
    interfaces::{MemoryEffect, MemoryEffects},
    prelude::*,
};
use cubecl_opt::analyses::{alias_analysis::address_space::AddressSpaceAA, memory_ssa::memory_ssa};
use expect_test::expect;
use pliron::{
    graph::dominance::DomInfo, init_env_logger_for_tests, irfmt::parsers::spaced,
    operation::verify_operation, parsable::parse_from_str, printable::Printable, result::ExpectOk,
};

#[pliron_op(
    name = "memory_ssa.unsupported_region",
    format = "region($0)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ"
)]
pub struct UnsupportedRegionOp;

#[pliron_op(
    name = "memory_ssa.unsupported_op",
    format = "",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ"
)]
pub struct UnsupportedOp;

#[pliron_op(
    name = "memory_ssa.memory_def",
    format = "",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ"
)]
pub struct MemoryDefOp;

#[op_interface_impl]
impl MemoryEffects for MemoryDefOp {
    fn memory_effects(&self, _ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::WriteAll]
    }
}

#[pliron_op(
    name = "memory_ssa.memory_use",
    format = "",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ"
)]
pub struct MemoryUseOp;

#[op_interface_impl]
impl MemoryEffects for MemoryUseOp {
    fn memory_effects(&self, _ctx: &Context) -> Vec<MemoryEffect> {
        vec![MemoryEffect::ReadAll]
    }
}

#[pliron_op(
    name = "memory_ssa.memory_def_in_space",
    format = "attr($def_address_space, $AddressSpaceAttr)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (def_address_space: AddressSpaceAttr)
)]
pub struct MemoryDefInSpaceOp;

#[op_interface_impl]
impl MemoryEffects for MemoryDefInSpaceOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_def_address_space(ctx).unwrap().0;
        vec![MemoryEffect::WriteAllInSpace(space)]
    }
}

#[pliron_op(
    name = "memory_ssa.memory_use_in_space",
    format = "attr($use_address_space, $AddressSpaceAttr)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (use_address_space: AddressSpaceAttr)
)]
pub struct MemoryUseInSpaceOp;

#[op_interface_impl]
impl MemoryEffects for MemoryUseInSpaceOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_use_address_space(ctx).unwrap().0;
        vec![MemoryEffect::ReadAllInSpace(space)]
    }
}

fn run_memory_ssa_on_text(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let dom_info = &mut DomInfo::default();
    let memory_ssa = memory_ssa(ctx, op, dom_info);

    Ok(memory_ssa.disp(ctx).to_string())
}

fn run_optimized_memory_ssa_on_text(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let dom_info = &mut DomInfo::default();
    let mut memory_ssa = memory_ssa(ctx, op, dom_info);
    memory_ssa.add_alias_analysis(AddressSpaceAA);
    memory_ssa.ensure_optimized_uses(ctx);

    Ok(memory_ssa.disp(ctx).to_string())
}

#[test]
fn memory_ssa_is_path_sensitive() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_use;
        memory_ssa.memory_def;
        memory_ssa.memory_use;

        true = builtin.constant <cube.bool true> : cube.bool;
        cf.branch_conditional if true ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_use;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_use;
        memory_ssa.memory_def;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; MemoryUse(LiveOnEntry, ReadAll)
            memory_ssa.memory_use;
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            true_v0 = builtin.constant <cube.bool true> : cube.bool;
            cf.branch_conditional if true_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            ; M3 = MemoryDef(M1, WriteAll)
            memory_ssa.memory_def;
            cf.branch ^bb2_block3v3();
  
          ; M2 = MemoryPhi({bb0_block4v1 -> M1}, {bb1_block5v1 -> M3})
          bb2_block3v3:
            ; MemoryUse(M2, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_does_not_version_reads() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_use;
        memory_ssa.memory_use;
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; MemoryUse(LiveOnEntry, ReadAll)
            memory_ssa.memory_use;
            ; MemoryUse(LiveOnEntry, ReadAll)
            memory_ssa.memory_use;
            ; MemoryUse(LiveOnEntry, ReadAll)
            memory_ssa.memory_use;
            branch.return;

        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_versions_each_write() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_use;
        memory_ssa.memory_def;
        memory_ssa.memory_use;
        memory_ssa.memory_def;
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; MemoryUse(LiveOnEntry, ReadAll)
            memory_ssa.memory_use;
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            ; M2 = MemoryDef(M1, WriteAll)
            memory_ssa.memory_def;
            ; MemoryUse(M2, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_merges_distinct_cfg_paths() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        memory_ssa.memory_def;
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_def;
        memory_ssa.memory_use;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_use;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; M3 = MemoryDef(M1, WriteAll)
            memory_ssa.memory_def;
            ; MemoryUse(M3, ReadAll)
            memory_ssa.memory_use;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            cf.branch ^bb2_block3v3();
  
          ; M2 = MemoryPhi({bb0_block4v1 -> M3}, {bb1_block5v1 -> M1})
          bb2_block3v3:
            ; MemoryUse(M2, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_merges_multiple_defining_paths() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_def;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_def;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; M2 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; M3 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            cf.branch ^bb2_block3v3();
  
          ; M1 = MemoryPhi({bb0_block4v1 -> M2}, {bb1_block5v1 -> M3})
          bb2_block3v3:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_preserves_entry_value_through_def_free_branch() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        memory_ssa.memory_def;
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_use;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_use;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            cf.branch ^bb2_block3v3();
  
          bb2_block3v3:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_versions_memory_inside_if_regions() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
        x = scf.if cond : builtin.integer i64 then {
          ^then():
            memory_ssa.memory_use;
            memory_ssa.memory_def;
            memory_ssa.memory_use;
            branch.yield (c)
        } else {
          ^else():
            memory_ssa.memory_use;
            branch.yield (c)
        };
        memory_ssa.memory_use;
        branch.return x
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            c_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            ; M2 = MemoryRegionPhi({then_block2v1 -> M1}, {else_block3v1 -> LiveOnEntry})
            x_v2 = scf.if cond_v0 : builtin.integer i64 then {..} else {..};
              then_block2v1:
                ; MemoryUse(LiveOnEntry, ReadAll)
                memory_ssa.memory_use;
                ; M1 = MemoryDef(LiveOnEntry, WriteAll)
                memory_ssa.memory_def;
                ; MemoryUse(M1, ReadAll)
                memory_ssa.memory_use;
                branch.yield (c_v1);
      
              else_block3v1:
                ; MemoryUse(LiveOnEntry, ReadAll)
                memory_ssa.memory_use;
                branch.yield (c_v1);
      
            ; MemoryUse(M2, ReadAll)
            memory_ssa.memory_use;
            branch.return x_v2;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_versions_memory_through_for_loop() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i32) -> ()> [] {
      ^entry(end: builtin.integer i32):
        start = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32;
        step = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
        y = scf.for start to end step step iter_args(c) {
          ^body(i: builtin.integer i32, c2: builtin.integer i64):
            memory_ssa.memory_use;
            memory_ssa.memory_def;
            memory_ssa.memory_use;
            branch.yield (c2)
        };
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            start_v1 = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32;
            step_v2 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
            c_v3 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            ; M3 = MemoryRegionPhi({Parent -> LiveOnEntry}, {body_block2v1 -> M2})
            y_v4 = scf.for start_v1 to end_v0 step step_v2 iter_args(c_v3) {..};
              ; M1 = MemoryRegionPhi({Parent -> LiveOnEntry}, {body_block2v1 -> M2})
              body_block2v1:
                ; MemoryUse(M1, ReadAll)
                memory_ssa.memory_use;
                ; M2 = MemoryDef(M1, WriteAll)
                memory_ssa.memory_def;
                ; MemoryUse(M2, ReadAll)
                memory_ssa.memory_use;
                branch.yield (c2_v6);
      
            ; MemoryUse(M3, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_versions_memory_through_while_loop() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
        y = scf.while c : builtin.integer i64 {
          ^before(x: builtin.integer i64):
            memory_ssa.memory_use;
            branch.condition (cond, x)
        } do {
          ^after(x2: builtin.integer i64):
            memory_ssa.memory_use;
            memory_ssa.memory_def;
            memory_ssa.memory_use;
            branch.yield (x2)
        };
        memory_ssa.memory_use;
        branch.return y
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            c_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            y_v4 = scf.while c_v1 : builtin.integer i64 {..} do {..};
              ; M1 = MemoryRegionPhi({Parent -> LiveOnEntry}, {after_block3v1 -> M3})
              before_block2v1:
                ; MemoryUse(M1, ReadAll)
                memory_ssa.memory_use;
                branch.condition (cond_v0, x_v2);
      
              ; M2 = MemoryRegionPhi({before_block2v1 -> M1})
              after_block3v1:
                ; MemoryUse(M2, ReadAll)
                memory_ssa.memory_use;
                ; M3 = MemoryDef(M2, WriteAll)
                memory_ssa.memory_def;
                ; MemoryUse(M3, ReadAll)
                memory_ssa.memory_use;
                branch.yield (x2_v3);
      
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            branch.return y_v4;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_versions_inside_unsupported_region_then_clobbers_at_boundary() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def;

        memory_ssa.unsupported_region {
          ^region_entry():
            memory_ssa.memory_use;
            memory_ssa.memory_def;
            memory_ssa.memory_use;
            branch.return
        };

        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            ; M4 = MemoryDef(M1, Opaque)
            memory_ssa.unsupported_region {..};
              ; M2 = MemoryDef(M1, Opaque)
              region_entry_block2v1:
                ; MemoryUse(M2, ReadAll)
                memory_ssa.memory_use;
                ; M3 = MemoryDef(M2, WriteAll)
                memory_ssa.memory_def;
                ; MemoryUse(M3, ReadAll)
                memory_ssa.memory_use;
                branch.return;
      
            ; MemoryUse(M4, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_unknown_operations_are_opaque_defs() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def;
        memory_ssa.unsupported_op;
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            ; M2 = MemoryDef(M1, Opaque)
            memory_ssa.unsupported_op;
            ; MemoryUse(M2, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_nested_supported_and_unsupported_regions() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        memory_ssa.memory_def;

        x = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
        y = scf.if cond : builtin.integer i64 then {
          ^then():
            memory_ssa.memory_use;
            memory_ssa.unsupported_region {
              ^nested():
                memory_ssa.memory_use;
                memory_ssa.memory_def;
                memory_ssa.memory_use;
                branch.return
            };
            memory_ssa.memory_use;
            branch.yield (x)
        } else {
          ^else():
            memory_ssa.memory_use;
            branch.yield (x)
        };

        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            x_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            ; M5 = MemoryRegionPhi({then_block2v1 -> M4}, {else_block4v1 -> M1})
            y_v2 = scf.if cond_v0 : builtin.integer i64 then {..} else {..};
              then_block2v1:
                ; MemoryUse(M1, ReadAll)
                memory_ssa.memory_use;
                ; M4 = MemoryDef(M1, Opaque)
                memory_ssa.unsupported_region {..};
                  ; M2 = MemoryDef(M1, Opaque)
                  nested_block3v1:
                    ; MemoryUse(M2, ReadAll)
                    memory_ssa.memory_use;
                    ; M3 = MemoryDef(M2, WriteAll)
                    memory_ssa.memory_def;
                    ; MemoryUse(M3, ReadAll)
                    memory_ssa.memory_use;
                    branch.return;

                ; MemoryUse(M4, ReadAll)
                memory_ssa.memory_use;
                branch.yield (x_v1);

              else_block4v1:
                ; MemoryUse(M1, ReadAll)
                memory_ssa.memory_use;
                branch.yield (x_v1);

            ; MemoryUse(M5, ReadAll)
            memory_ssa.memory_use;
            branch.return;

        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_non_aliasing_defs_are_skipped() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_use_in_space Local;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_use_in_space Shared;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; MemoryUse(M1, ReadAllInSpace(Shared))
            memory_ssa.memory_use_in_space Shared;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_places_phi_after_iterated_dominance_frontier() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, cube.bool) -> ()> [] {
      ^entry(outer_cond: cube.bool, inner_cond: cube.bool):
        cf.branch_conditional if outer_cond ^outer() else ^exit()

      ^outer():
        cf.branch_conditional if inner_cond ^write() else ^merge()

      ^write():
        memory_ssa.memory_def;
        cf.branch ^merge()

      ^merge():
        cf.branch ^exit()

      ^exit():
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            cf.branch_conditional if outer_cond_v0 ^outer_block4v1() else ^exit_block5v3();

          outer_block4v1:
            cf.branch_conditional if inner_cond_v1 ^write_block6v1() else ^merge_block2v5();

          write_block6v1:
            ; M3 = MemoryDef(LiveOnEntry, WriteAll)
            memory_ssa.memory_def;
            cf.branch ^merge_block2v5();

          ; M2 = MemoryPhi({outer_block4v1 -> LiveOnEntry}, {write_block6v1 -> M3})
          merge_block2v5:
            cf.branch ^exit_block5v3();

          ; M1 = MemoryPhi({entry_block1v1 -> LiveOnEntry}, {merge_block2v5 -> M2})
          exit_block5v3:
            ; MemoryUse(M1, ReadAll)
            memory_ssa.memory_use;
            branch.return;

        }"#]].assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_skips_non_aliasing_defs() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_use_in_space Shared;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; MemoryUse(M1, ReadAllInSpace(Shared))
            memory_ssa.memory_use_in_space Shared;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_stops_at_aliasing_def() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; MemoryUse(M2, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_skips_multiple_non_aliasing_defs() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Global<0>;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_def_in_space Global<0>;
        memory_ssa.memory_use_in_space Shared;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Global<0>))
            memory_ssa.memory_def_in_space Global<0>;
            ; M3 = MemoryDef(M2, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; M4 = MemoryDef(M3, WriteAllInSpace(Global<0>))
            memory_ssa.memory_def_in_space Global<0>;
            ; MemoryUse(M1, ReadAllInSpace(Shared))
            memory_ssa.memory_use_in_space Shared;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_finds_first_aliasing_def() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_def_in_space Global<0>;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; M3 = MemoryDef(M2, WriteAllInSpace(Global<0>))
            memory_ssa.memory_def_in_space Global<0>;
            ; M4 = MemoryDef(M3, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; MemoryUse(M4, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_skips_defs_before_an_aliasing_def() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Local;
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.memory_def_in_space Global<0>;
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M2 = MemoryDef(M1, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            ; M3 = MemoryDef(M2, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M4 = MemoryDef(M3, WriteAllInSpace(Global<0>))
            memory_ssa.memory_def_in_space Global<0>;
            ; MemoryUse(M2, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_at_live_on_entry() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_through_non_aliasing_cfg_path() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        memory_ssa.memory_def_in_space Shared;
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_def_in_space Shared;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_def_in_space Shared;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; M3 = MemoryDef(M1, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; M4 = MemoryDef(M1, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch ^bb2_block3v3();
  
          ; M2 = MemoryPhi({bb0_block4v1 -> M3}, {bb1_block5v1 -> M4})
          bb2_block3v3:
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_stops_at_cfg_phi_when_aliasing_paths_differ() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_def_in_space Local;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_def_in_space Shared;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; M2 = MemoryDef(LiveOnEntry, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; M3 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch ^bb2_block3v3();
  
          ; M1 = MemoryPhi({bb0_block4v1 -> M2}, {bb1_block5v1 -> M3})
          bb2_block3v3:
            ; MemoryUse(M1, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_collapses_cfg_phi_with_same_clobber() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> ()> [] {
      ^entry(cond: cube.bool):
        memory_ssa.memory_def_in_space Local;
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory_ssa.memory_def_in_space Shared;
        cf.branch ^bb2()

      ^bb1():
        memory_ssa.memory_def_in_space Shared;
        cf.branch ^bb2()

      ^bb2():
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Local))
            memory_ssa.memory_def_in_space Local;
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1();
  
          bb0_block4v1:
            ; M3 = MemoryDef(M1, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch ^bb2_block3v3();
  
          bb1_block5v1:
            ; M4 = MemoryDef(M1, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            cf.branch ^bb2_block3v3();
  
          ; M2 = MemoryPhi({bb0_block4v1 -> M3}, {bb1_block5v1 -> M4})
          bb2_block3v3:
            ; MemoryUse(M1, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_through_if_region_with_non_aliasing_def() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
        memory_ssa.memory_def_in_space Shared;

        x = scf.if cond : builtin.integer i64 then {
          ^then():
            memory_ssa.memory_def_in_space Shared;
            branch.yield (c)
        } else {
          ^else():
            memory_ssa.memory_def_in_space Shared;
            branch.yield (c)
        };

        memory_ssa.memory_use_in_space Local;
        branch.return x
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            c_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M4 = MemoryRegionPhi({then_block2v1 -> M2}, {else_block3v1 -> M3})
            x_v2 = scf.if cond_v0 : builtin.integer i64 then {..} else {..};
              then_block2v1:
                ; M2 = MemoryDef(M1, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                branch.yield (c_v1);
      
              else_block3v1:
                ; M3 = MemoryDef(M1, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                branch.yield (c_v1);
      
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return x_v2;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_stops_at_if_region_phi_when_one_path_clobbers() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;

        x = scf.if cond : builtin.integer i64 then {
          ^then():
            memory_ssa.memory_def_in_space Local;
            branch.yield (c)
        } else {
          ^else():
            memory_ssa.memory_def_in_space Shared;
            branch.yield (c)
        };

        memory_ssa.memory_use_in_space Local;
        branch.return x
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            c_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            ; M3 = MemoryRegionPhi({then_block2v1 -> M1}, {else_block3v1 -> M2})
            x_v2 = scf.if cond_v0 : builtin.integer i64 then {..} else {..};
              then_block2v1:
                ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Local))
                memory_ssa.memory_def_in_space Local;
                branch.yield (c_v1);
      
              else_block3v1:
                ; M2 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                branch.yield (c_v1);
      
            ; MemoryUse(M3, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return x_v2;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_through_for_loop_non_aliasing_defs() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i32) -> ()> [] {
      ^entry(end: builtin.integer i32):
        start = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32;
        step = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;

        scf.for start to end step step iter_args() {
          ^body(i: builtin.integer i32):
            memory_ssa.memory_def_in_space Shared;
            memory_ssa.memory_def_in_space Global<0>;
            branch.yield ()
        };

        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            start_v1 = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32;
            step_v2 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
            ; M4 = MemoryRegionPhi({Parent -> LiveOnEntry}, {body_block2v1 -> M3})
            scf.for start_v1 to end_v0 step step_v2 {..};
              ; M1 = MemoryRegionPhi({Parent -> LiveOnEntry}, {body_block2v1 -> M3})
              body_block2v1:
                ; M2 = MemoryDef(M1, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                ; M3 = MemoryDef(M2, WriteAllInSpace(Global<0>))
                memory_ssa.memory_def_in_space Global<0>;
                branch.yield ();
      
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_through_while_loop_non_aliasing_defs() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        c = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;

        y = scf.while c : builtin.integer i64 {
          ^before(x: builtin.integer i64):
            branch.condition (cond, x)
        } do {
          ^after(x2: builtin.integer i64):
            memory_ssa.memory_def_in_space Shared;
            memory_ssa.memory_def_in_space Global<0>;
            branch.yield (x2)
        };

        memory_ssa.memory_use_in_space Local;
        branch.return y
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            c_v1 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
            y_v4 = scf.while c_v1 : builtin.integer i64 {..} do {..};
              ; M1 = MemoryRegionPhi({Parent -> LiveOnEntry}, {after_block3v1 -> M4})
              before_block2v1:
                branch.condition (cond_v0, x_v2);
      
              ; M2 = MemoryRegionPhi({before_block2v1 -> M1})
              after_block3v1:
                ; M3 = MemoryDef(M2, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                ; M4 = MemoryDef(M3, WriteAllInSpace(Global<0>))
                memory_ssa.memory_def_in_space Global<0>;
                branch.yield (x2_v3);
      
            ; MemoryUse(LiveOnEntry, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return y_v4;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_respects_opaque_boundary() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.unsupported_region {
          ^region_entry():
            memory_ssa.memory_def_in_space Shared;
            branch.return
        };
        memory_ssa.memory_use_in_space Local;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M4 = MemoryDef(M1, Opaque)
            memory_ssa.unsupported_region {..};
              ; M2 = MemoryDef(M1, Opaque)
              region_entry_block2v1:
                ; M3 = MemoryDef(M2, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                branch.return;
      
            ; MemoryUse(M4, ReadAllInSpace(Local))
            memory_ssa.memory_use_in_space Local;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn memory_ssa_optimized_use_clobbered_by_opaque_boundary() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> ()> [] {
      ^entry():
        memory_ssa.memory_def_in_space Shared;
        memory_ssa.unsupported_region {
          ^region_entry():
            memory_ssa.memory_def_in_space Shared;
            branch.return
        };
        memory_ssa.memory_use;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let printed = run_optimized_memory_ssa_on_text(ctx, input)?;

    expect![[r#"
        MemorySSA {
          entry_block1v1:
            ; M1 = MemoryDef(LiveOnEntry, WriteAllInSpace(Shared))
            memory_ssa.memory_def_in_space Shared;
            ; M4 = MemoryDef(M1, Opaque)
            memory_ssa.unsupported_region {..};
              ; M2 = MemoryDef(M1, Opaque)
              region_entry_block2v1:
                ; M3 = MemoryDef(M2, WriteAllInSpace(Shared))
                memory_ssa.memory_def_in_space Shared;
                branch.return;
      
            ; MemoryUse(M4, ReadAll)
            memory_ssa.memory_use;
            branch.return;
  
        }"#]]
    .assert_eq(&printed);
    Ok(())
}
