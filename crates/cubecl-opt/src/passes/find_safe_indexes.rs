use cubecl_ir::{Operation, Operator, Variable, VariableKind};

use crate::{
    AtomicCounter, Optimizer,
    analyses::{const_len::Slices, integer_range::Ranges},
};

use super::OptimizerPass;

/// Use the results from integer range analysis to find indexes that are always in bounds, then
/// transform them to unchecked indexes.
pub struct InBoundsToUnchecked;

impl OptimizerPass for InBoundsToUnchecked {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let ranges = opt.analysis::<Ranges>();

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
                            let range = ranges.range_of(opt, &op.rhs);
                            if let Some((_, upper)) = range.lower_bound.zip(range.upper_bound) {
                                if (upper as u32) < const_len {
                                    inst.operation = Operator::UncheckedIndex(op.clone()).into();
                                    changes.inc();
                                }
                            }
                        }
                    }
                    Operator::IndexAssign(op) => {
                        if let Some(const_len) = const_len(opt, &inst.out()) {
                            let range = ranges.range_of(opt, &op.lhs);
                            if let Some((_, upper)) = range.lower_bound.zip(range.upper_bound) {
                                if (upper as u32) < const_len {
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

fn const_len(opt: &mut Optimizer, var: &Variable) -> Option<u32> {
    let slices = opt.analysis::<Slices>();
    match var.kind {
        VariableKind::ConstantArray { length, .. } => Some(length),
        VariableKind::SharedMemory { length, .. } => Some(length),
        VariableKind::LocalArray { length, .. } => Some(length),
        VariableKind::Slice { id } => slices.get(&id).and_then(|it| it.const_len),
        _ => None,
    }
}
