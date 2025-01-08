use cubecl_core::ir::{Operation, Operator, Variable, VariableKind};

use crate::{AtomicCounter, Optimizer};

use super::{range_of, OptimizerPass};

/// Try to find any constant length slices by cancelling common factors in `start` and `end`
pub struct FindConstSliceLen;

impl OptimizerPass for FindConstSliceLen {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for operator in ops.borrow().values() {
                let op = match &operator.operation {
                    Operation::Operator(op) => op,
                    _ => continue,
                };
                // Only handle the simplest cases for now
                if let Operator::Add(op) = op {
                    let mut slices = opt.program.slices.values_mut();
                    let slice =
                        slices.find(|it| it.end == operator.out() && it.const_len.is_none());
                    if let Some(slice) = slice {
                        slice.end_op = Some(Operator::Add(op.clone()).into());
                        if op.lhs == slice.start && op.rhs.as_const().is_some() {
                            slice.const_len = Some(op.rhs.as_const().unwrap().as_u32());
                            changes.inc();
                        } else if op.rhs == slice.start && op.lhs.as_const().is_some() {
                            slice.const_len = Some(op.lhs.as_const().unwrap().as_u32());
                            changes.inc();
                        }
                    }
                }
            }
        }
    }
}

/// Use the results from integer range analysis to find indexes that are always in bounds, then
/// transform them to unchecked indexes.
pub struct InBoundsToUnchecked;

impl OptimizerPass for InBoundsToUnchecked {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for inst in ops.borrow_mut().values_mut() {
                let op = match &inst.operation {
                    Operation::Operator(op) => op,
                    _ => continue,
                };
                match op {
                    Operator::Index(op) => {
                        if let Some(const_len) = const_len(opt, &op.lhs) {
                            let range = range_of(opt, &op.rhs);
                            if let Some((lower, upper)) = range.lower_bound.zip(range.upper_bound) {
                                if lower >= 0 && (upper as u32) < const_len {
                                    inst.operation = Operator::UncheckedIndex(op.clone()).into();
                                    changes.inc();
                                }
                            }
                        }
                    }
                    Operator::IndexAssign(op) => {
                        if let Some(const_len) = const_len(opt, &inst.out()) {
                            let range = range_of(opt, &op.lhs);
                            if let Some((lower, upper)) = range.lower_bound.zip(range.upper_bound) {
                                if lower >= 0 && (upper as u32) < const_len {
                                    inst.operation =
                                        Operator::UncheckedIndexAssign(op.clone()).into();
                                    changes.inc();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn const_len(opt: &Optimizer, var: &Variable) -> Option<u32> {
    match var.kind {
        VariableKind::ConstantArray { length, .. } => Some(length),
        VariableKind::SharedMemory { length, .. } => Some(length),
        VariableKind::LocalArray { length, .. } => Some(length),
        VariableKind::Slice { id } => opt.program.slices.get(&id).and_then(|it| it.const_len),
        _ => None,
    }
}
