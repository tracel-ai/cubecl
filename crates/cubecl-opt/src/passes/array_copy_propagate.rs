use std::collections::HashMap;

use cubecl_core::ir::{Item, Operation, Operator, UnaryOperator, Variable};

use crate::{AtomicCounter, Optimizer};

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
                .map(|_| opt.root_scope.create_local_undeclared(item))
                .collect::<Vec<_>>();
            for var in &vars {
                let local_id = opt.local_variable_id(var).unwrap();
                opt.program.variables.insert(local_id, var.item());
            }
            replace_const_arrays(opt, arr_id, &vars);
            changes.inc();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Array {
    id: (u16, u8),
    length: u32,
    item: Item,
}

fn find_const_arrays(opt: &mut Optimizer) -> Vec<Array> {
    let mut track_consts = HashMap::new();
    let mut arrays = HashMap::new();

    for block in opt.node_ids() {
        let ops = opt.program[block].ops.clone();
        for op in ops.borrow().values() {
            match op {
                Operation::Operator(Operator::Index(index)) => {
                    if let Variable::LocalArray {
                        id,
                        length,
                        item,
                        depth,
                    } = index.lhs
                    {
                        let id = (id, depth);
                        arrays.insert(id, Array { id, length, item });
                        let is_const = index.rhs.as_const().is_some();
                        *track_consts.entry(id).or_insert(is_const) &= is_const;
                    }
                }
                Operation::Operator(Operator::IndexAssign(assign)) => {
                    if let Variable::LocalArray {
                        id,
                        length,
                        item,
                        depth,
                    } = assign.out
                    {
                        let id = (id, depth);
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

fn replace_const_arrays(opt: &mut Optimizer, arr_id: (u16, u8), vars: &[Variable]) {
    for block in opt.node_ids() {
        let ops = opt.program[block].ops.clone();
        for op in ops.borrow_mut().values_mut() {
            match op {
                Operation::Operator(Operator::Index(index)) => {
                    if let Variable::LocalArray { id, depth, .. } = index.lhs {
                        let const_index = index.rhs.as_const().unwrap().as_i64() as usize;
                        if (id, depth) == arr_id {
                            *op = Operator::Assign(UnaryOperator {
                                input: vars[const_index],
                                out: index.out,
                            })
                            .into();
                        }
                    }
                }
                Operation::Operator(Operator::IndexAssign(assign)) => {
                    if let Variable::LocalArray { id, depth, .. } = assign.out {
                        let const_index = assign.lhs.as_const().unwrap().as_i64() as usize;
                        if (id, depth) == arr_id {
                            let out = vars[const_index];
                            let out_id = opt.local_variable_id(&out).unwrap();
                            *op = Operator::Assign(UnaryOperator {
                                input: assign.rhs,
                                out,
                            })
                            .into();
                            opt.program[block].writes.insert(out_id);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
