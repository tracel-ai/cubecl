use std::{collections::HashMap, ops::Deref};

use cubecl_ir::{Arithmetic, Id, Operation, Operator, Variable, VariableKind};

use crate::Optimizer;

use super::Analysis;

#[derive(Debug, Clone)]
pub struct Slice {
    pub start: Variable,
    pub end: Variable,
    pub end_op: Option<Operation>,
    pub const_len: Option<u32>,
}

/// Try to find any constant length slices by cancelling common factors in `start` and `end`
#[derive(Default, Debug)]
pub struct Slices {
    slices: HashMap<Id, Slice>,
}

impl Deref for Slices {
    type Target = HashMap<Id, Slice>;

    fn deref(&self) -> &Self::Target {
        &self.slices
    }
}

impl Analysis for Slices {
    fn init(opt: &mut Optimizer) -> Self {
        let mut this = Slices::default();
        this.populate_slices(opt);
        this.find_end_ops(opt);
        this
    }
}

impl Slices {
    fn populate_slices(&mut self, opt: &mut Optimizer) {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for operator in ops.borrow().values() {
                let op = match &operator.operation {
                    Operation::Operator(op) => op,
                    _ => continue,
                };
                let out = operator.out.as_ref();
                if let Operator::Slice(slice_op) = op {
                    let out_id = match out.unwrap().kind {
                        VariableKind::Slice { id } => id,
                        _ => unreachable!(),
                    };
                    let const_len = slice_op.start.as_const().zip(slice_op.end.as_const());
                    let const_len = const_len.map(|(start, end)| end.as_u32() - start.as_u32());
                    self.slices.insert(
                        out_id,
                        Slice {
                            start: slice_op.start,
                            end: slice_op.end,
                            end_op: None,
                            const_len,
                        },
                    );
                };
            }
        }
    }

    fn find_end_ops(&mut self, opt: &mut Optimizer) {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for operator in ops.borrow().values() {
                let op = match &operator.operation {
                    Operation::Arithmetic(op) => op,
                    _ => continue,
                };
                // Only handle the simplest cases for now
                if let Arithmetic::Add(op) = op {
                    let mut slices = self.slices.values_mut();
                    let slice =
                        slices.find(|it| it.end == operator.out() && it.const_len.is_none());
                    if let Some(slice) = slice {
                        slice.end_op = Some(Arithmetic::Add(op.clone()).into());
                        if op.lhs == slice.start && op.rhs.as_const().is_some() {
                            slice.const_len = Some(op.rhs.as_const().unwrap().as_u32());
                        } else if op.rhs == slice.start && op.lhs.as_const().is_some() {
                            slice.const_len = Some(op.lhs.as_const().unwrap().as_u32());
                        }
                    }
                };
            }
        }
    }
}
