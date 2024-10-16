use std::mem::take;

use cubecl_core::ir::{BinaryOperator, Elem, Operation, Operator, Variable};

use crate::{AtomicCounter, Optimizer};

use super::OptimizerPass;

pub struct ReduceStrength;

impl OptimizerPass for ReduceStrength {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let ops = take(&mut *opt.block(block).ops.borrow_mut());
            let mut new_ops = Vec::with_capacity(ops.capacity());
            for (_, operation) in ops.into_iter() {
                let op = match operation {
                    Operation::Operator(op) => op,
                    _ => {
                        new_ops.push(operation);
                        continue;
                    }
                };
                match op {
                    Operator::Mul(op) if op.out.item().elem() == Elem::UInt => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Operator::Mul(op).into());
                                continue;
                            }
                        };
                        match const_val {
                            val if val.is_power_of_two() => {
                                new_ops.push(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: val.trailing_zeros().into(),
                                        out: op.out,
                                    })
                                    .into(),
                                );
                                changes.inc();
                            }
                            val if (val + 1).is_power_of_two() => {
                                let temp = opt.create_temporary(op.out.item());
                                new_ops.push(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: (val + 1).trailing_zeros().into(),
                                        out: temp,
                                    })
                                    .into(),
                                );
                                new_ops.push(
                                    Operator::Sub(BinaryOperator {
                                        lhs: temp,
                                        rhs: dyn_val,
                                        out: op.out,
                                    })
                                    .into(),
                                );
                                changes.inc();
                            }
                            val if (val - 1).is_power_of_two() => {
                                let temp = opt.create_temporary(op.out.item());
                                new_ops.push(
                                    Operator::ShiftLeft(BinaryOperator {
                                        lhs: dyn_val,
                                        rhs: (val - 1).trailing_zeros().into(),
                                        out: temp,
                                    })
                                    .into(),
                                );
                                new_ops.push(
                                    Operator::Add(BinaryOperator {
                                        lhs: temp,
                                        rhs: dyn_val,
                                        out: op.out,
                                    })
                                    .into(),
                                );
                                changes.inc();
                            }
                            _ => {
                                new_ops.push(Operator::Mul(op).into());
                            }
                        }
                    }
                    Operator::Div(op) if is_pow2(op.rhs) => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Operator::Div(op).into());
                                continue;
                            }
                        };
                        new_ops.push(
                            Operator::ShiftRight(BinaryOperator {
                                lhs: dyn_val,
                                rhs: const_val.trailing_zeros().into(),
                                out: op.out,
                            })
                            .into(),
                        );
                        changes.inc();
                    }
                    Operator::Modulo(op) if is_pow2(op.rhs) => {
                        let (const_val, dyn_val) = match (op.lhs.as_const(), op.rhs.as_const()) {
                            (None, Some(val)) => (val.as_u32(), op.lhs),
                            (Some(val), None) => (val.as_u32(), op.rhs),
                            _ => {
                                new_ops.push(Operator::Div(op).into());
                                continue;
                            }
                        };
                        new_ops.push(
                            Operator::BitwiseAnd(BinaryOperator {
                                lhs: dyn_val,
                                rhs: (const_val - 1).into(),
                                out: op.out,
                            })
                            .into(),
                        );
                        changes.inc();
                    }
                    op => {
                        new_ops.push(op.into());
                    }
                }
            }
            opt.block(block).ops.borrow_mut().extend(new_ops);
        }
    }
}

fn is_pow2(var: Variable) -> bool {
    var.item().elem() == Elem::UInt
        && var
            .as_const()
            .map(|it| it.as_u32().is_power_of_two())
            .unwrap_or(false)
}
