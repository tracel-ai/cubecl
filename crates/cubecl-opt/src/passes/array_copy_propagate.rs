use std::collections::HashMap;

use cubecl_ir::{Id, Instruction, Item, Operation, Operator, Variable, VariableKind};

use crate::{AtomicCounter, Optimizer, analyses::writes::Writes};

use super::OptimizerPass;

/// Split arrays with only constant indices into a set of local intermediates. This allows the
/// compiler to reorder them and optimize memory layout, along with enabling more inlining and
/// expression merging.
///
/// # Example
///
/// ```ignore
/// arr[0] = a;
/// arr[1] = b;
/// arr[2] = c;
/// ```
/// transforms to
/// ```ignore
/// let a1 = a;
/// let b1 = b;
/// let c1 = c;
/// ```
/// which can usually be completely merged out and inlined.
///
pub struct CopyPropagateArray;

impl OptimizerPass for CopyPropagateArray {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let arrays = find_const_arrays(opt);

        for Array { id, length, item } in arrays {
            let arr_id = id;
            let vars = (0..length)
                .map(|_| *opt.root_scope.create_local_restricted(item))
                .collect::<Vec<_>>();
            for var in &vars {
                let local_id = opt.local_variable_id(var).unwrap();
                opt.program.variables.insert(local_id, var.item);
            }
            replace_const_arrays(opt, arr_id, &vars);
            changes.inc();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Array {
    id: Id,
    length: u32,
    item: Item,
}

fn find_const_arrays(opt: &mut Optimizer) -> Vec<Array> {
    let mut track_consts = HashMap::new();
    let mut arrays = HashMap::new();

    for block in opt.node_ids() {
        let ops = opt.program[block].ops.clone();
        for op in ops.borrow().values() {
            match &op.operation {
                Operation::Operator(Operator::Index(index) | Operator::UncheckedIndex(index)) => {
                    if let VariableKind::LocalArray { id, length } = index.lhs.kind {
                        let item = index.lhs.item;
                        arrays.insert(id, Array { id, length, item });
                        let is_const = index.rhs.as_const().is_some();
                        *track_consts.entry(id).or_insert(is_const) &= is_const;
                    }
                }
                Operation::Operator(
                    Operator::IndexAssign(assign) | Operator::UncheckedIndexAssign(assign),
                ) => {
                    if let VariableKind::LocalArray { id, length } = op.out().kind {
                        let item = op.out().item;
                        arrays.insert(id, Array { id, length, item });
                        let is_const = assign.lhs.as_const().is_some();
                        *track_consts.entry(id).or_insert(is_const) &= is_const;
                    }
                }
                _ => {}
            }
        }
    }

    track_consts
        .iter()
        .filter(|(_, is_const)| **is_const)
        .map(|(id, _)| arrays[id])
        .collect()
}

fn replace_const_arrays(opt: &mut Optimizer, arr_id: Id, vars: &[Variable]) {
    for block in opt.node_ids() {
        let ops = opt.program[block].ops.clone();
        for op in ops.borrow_mut().values_mut() {
            match &mut op.operation.clone() {
                Operation::Operator(Operator::Index(index) | Operator::UncheckedIndex(index)) => {
                    if let VariableKind::LocalArray { id, .. } = index.lhs.kind {
                        if id == arr_id {
                            let const_index = index.rhs.as_const().unwrap().as_i64() as usize;
                            op.operation = Operation::Copy(vars[const_index]);
                        }
                    }
                }
                Operation::Operator(
                    Operator::IndexAssign(assign) | Operator::UncheckedIndexAssign(assign),
                ) => {
                    if let VariableKind::LocalArray { id, .. } = op.out.unwrap().kind {
                        if id == arr_id {
                            let const_index = assign.lhs.as_const().unwrap().as_i64() as usize;
                            let out = vars[const_index];
                            *op = Instruction::new(Operation::Copy(assign.rhs), out);
                            opt.invalidate_analysis::<Writes>();
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
