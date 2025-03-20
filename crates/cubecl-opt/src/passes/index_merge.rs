use std::collections::{HashMap, HashSet};

use cubecl_ir::{CopyMemoryOperator, Id, Instruction, Operation, Operator, Variable, VariableKind};

use crate::{AtomicCounter, Optimizer};

use super::{OptimizerPass, item_compatible};

/// Transform consecutive `Index` and `IndexAssign` operations that don't modify the value into a
/// single `Copy` operation.
pub struct CopyTransform;

impl OptimizerPass for CopyTransform {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let mut reads = HashMap::new();
            let mut writes = HashMap::new();
            let ops = opt.program[block].ops.clone();
            let indices = ops.borrow().indices().collect::<Vec<_>>();
            for idx in indices {
                let inst = ops.borrow()[idx].clone();
                match &inst.operation {
                    Operation::Operator(Operator::Index(op))
                        if op.lhs.is_array()
                            && item_compatible(op.lhs.item, inst.item())
                            && !is_reused(opt, &inst.out) =>
                    {
                        if let Some(id) = as_versioned(&inst.out()) {
                            reads.insert(id, (idx, op.lhs, op.rhs));
                        }
                    }
                    Operation::Operator(Operator::IndexAssign(op))
                        if inst.out().is_array() && item_compatible(inst.item(), op.rhs.item) =>
                    {
                        if let Some(id) = as_versioned(&op.rhs) {
                            writes.insert(id, (idx, inst.out(), op.lhs));
                        }
                    }
                    _ => {}
                }
            }
            let read_ids: HashSet<_> = reads.keys().collect();
            let write_ids: HashSet<_> = writes.keys().collect();
            let copy_ids = read_ids.intersection(&write_ids);
            for id in copy_ids {
                let (read_idx, input, in_index) = reads[*id];
                let (write_idx, out, out_index) = writes[*id];
                ops.borrow_mut().remove(read_idx);
                let copy = Operator::CopyMemory(CopyMemoryOperator {
                    out_index,
                    input,
                    in_index,
                });
                ops.borrow_mut()[write_idx] = Instruction::new(copy, out);
                changes.inc();
            }
        }
    }
}

fn as_versioned(var: &Variable) -> Option<(Id, u16)> {
    match var.kind {
        VariableKind::LocalConst { id } => Some((id, 0)),
        VariableKind::Versioned { id, version } => Some((id, version)),
        _ => None,
    }
}

fn is_reused(opt: &mut Optimizer, var: &Option<Variable>) -> bool {
    if let Some(var) = var.as_ref() {
        let count = AtomicCounter::new(0);
        opt.visit_all(
            |_, other| {
                if other == var {
                    count.inc();
                }
            },
            |_, _| {},
        );
        count.get() > 1
    } else {
        false
    }
}
