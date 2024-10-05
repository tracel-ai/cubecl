use cubecl_core::ir::{ConstantScalarValue, Operation, Operator, UnaryOperator, Variable};

use crate::{AtomicCounter, Optimizer};

use super::OptimizationPass;

pub struct ZeroOperandSimplify;

impl OptimizationPass for ZeroOperandSimplify {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

            for idx in ops {
                let op = &mut opt.program[node].ops.borrow_mut()[idx];
                match op {
                    Operation::Operator(Operator::Mul(bin_op)) => {
                        let lhs_zero = bin_op
                            .lhs
                            .as_const()
                            .map(|it| it.is_zero())
                            .unwrap_or(false);
                        let rhs_zero = bin_op
                            .rhs
                            .as_const()
                            .map(|it| it.is_zero())
                            .unwrap_or(false);
                        if lhs_zero || rhs_zero {
                            let zero = ConstantScalarValue::UInt(0);
                            let input = Variable::ConstantScalar(zero);
                            *op = Operator::Assign(UnaryOperator {
                                input,
                                out: bin_op.out,
                            })
                            .into();
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Add(bin_op)) => {
                        let lhs_zero = bin_op
                            .lhs
                            .as_const()
                            .map(|it| it.is_zero())
                            .unwrap_or(false);
                        let rhs_zero = bin_op
                            .rhs
                            .as_const()
                            .map(|it| it.is_zero())
                            .unwrap_or(false);
                        if lhs_zero {
                            *op = Operator::Assign(UnaryOperator {
                                input: bin_op.rhs,
                                out: bin_op.out,
                            })
                            .into();
                            changes.inc();
                        } else if rhs_zero {
                            *op = Operator::Assign(UnaryOperator {
                                input: bin_op.lhs,
                                out: bin_op.out,
                            })
                            .into();
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Sub(bin_op)) => {
                        let rhs_zero = bin_op
                            .rhs
                            .as_const()
                            .map(|it| it.is_zero())
                            .unwrap_or(false);
                        if rhs_zero {
                            *op = Operator::Assign(UnaryOperator {
                                input: bin_op.lhs,
                                out: bin_op.out,
                            })
                            .into();
                            changes.inc();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
