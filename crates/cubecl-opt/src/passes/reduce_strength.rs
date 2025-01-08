use std::mem::take;

use cubecl_core::ir::{BinaryOperator, Elem, Instruction, Operation, Operator, UIntKind, Variable};

use crate::{AtomicCounter, Optimizer};

use super::OptimizerPass;

/// Replace expensive operations with less expensive equivalent ones.
/// Example
/// ```rust,ignore
/// let a = x % 16;
/// let b = x / 8;
/// let c = x * 15;
/// ```
/// to
/// ```rust,ignore
/// let a = x & 15;
/// let b = x >> 3;
/// let c = (x << 4) - x;
/// ```
pub struct ReduceStrength;

impl OptimizerPass for ReduceStrength {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let ops = take(&mut *opt.block(block).ops.borrow_mut());
            let mut new_ops = Vec::with_capacity(ops.capacity());
            for (_, inst) in ops.into_iter() {
                let op = match inst.operation.clone() {
                    Operation::Operator(op) => op,
                    _ => {
                        new_ops.push(inst);
                        continue;
                    }
                };
                match op {
                    Operator::Mul(op) if inst.item().elem() == Elem::UInt(UIntKind::U32) => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Instruction::new(Operator::Mul(op), inst.out()));
                                continue;
                            }
                        };
                        match const_val {
                            val if val.is_power_of_two() => {
                                new_ops.push(Instruction::new(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: val.trailing_zeros().into(),
                                    }),
                                    inst.out(),
                                ));
                                changes.inc();
                            }
                            val if (val + 1).is_power_of_two() => {
                                let temp = *opt.allocator.create_local_restricted(inst.item());
                                new_ops.push(Instruction::new(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: (val + 1).trailing_zeros().into(),
                                    }),
                                    temp,
                                ));
                                new_ops.push(Instruction::new(
                                    Operator::Sub(BinaryOperator {
                                        lhs: temp,
                                        rhs: dyn_val,
                                    }),
                                    inst.out(),
                                ));
                                changes.inc();
                            }
                            val if (val - 1).is_power_of_two() => {
                                let temp = *opt.allocator.create_local_restricted(inst.item());
                                new_ops.push(Instruction::new(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: (val - 1).trailing_zeros().into(),
                                    }),
                                    temp,
                                ));
                                new_ops.push(Instruction::new(
                                    Operator::Add(BinaryOperator {
                                        lhs: temp,
                                        rhs: dyn_val,
                                    }),
                                    inst.out(),
                                ));
                                changes.inc();
                            }
                            _ => {
                                new_ops.push(Instruction::new(Operator::Mul(op), inst.out()));
                            }
                        }
                    }
                    Operator::Div(op) if is_pow2(op.rhs) => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Instruction::new(Operator::Div(op), inst.out()));
                                continue;
                            }
                        };
                        new_ops.push(Instruction::new(
                            Operator::ShiftRight(BinaryOperator {
                                lhs: dyn_val,
                                rhs: const_val.trailing_zeros().into(),
                            }),
                            inst.out(),
                        ));
                        changes.inc();
                    }
                    Operator::Modulo(op) if is_pow2(op.rhs) => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Instruction::new(Operator::Div(op), inst.out()));
                                continue;
                            }
                        };
                        new_ops.push(Instruction::new(
                            Operator::BitwiseAnd(BinaryOperator {
                                lhs: dyn_val,
                                rhs: (const_val - 1).into(),
                            }),
                            inst.out(),
                        ));
                        changes.inc();
                    }
                    _ => {
                        new_ops.push(inst);
                    }
                }
            }
            opt.block(block).ops.borrow_mut().extend(new_ops);
        }
    }
}

fn is_pow2(var: Variable) -> bool {
    var.item.elem() == Elem::UInt(UIntKind::U32)
        && var
            .as_const()
            .map(|it| it.as_u32().is_power_of_two())
            .unwrap_or(false)
}
