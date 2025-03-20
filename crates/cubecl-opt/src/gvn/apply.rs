use std::collections::{HashMap, HashSet};

use cubecl_ir::{self as ir, Operation};
use petgraph::graph::NodeIndex;

use crate::{
    AtomicCounter, Optimizer, PhiInstruction,
    analyses::dominance::Dominators,
    gvn::{convert::value_of_var, phi_translate},
    version::PhiEntry,
};

use super::GvnState;

impl GvnState {
    /// Find places where an expression is partially but not fully available, and hoist the
    /// computation into the blocks that do not currently have the value available to make the
    /// expression fully redundant
    pub fn insert(&mut self, opt: &mut Optimizer, changes: &AtomicCounter) {
        let mut loops = 1;
        let changes_pre = changes.get();

        let mut new_expr = HashMap::new();

        while self.insert_block(opt, opt.entry(), &mut new_expr, changes) {
            loops += 1;
        }
        let inserted = changes.get() - changes_pre;
        log::debug!("Insert loops: {loops}");
        log::debug!("Hoisted {inserted} expressions");
    }

    fn insert_block(
        &mut self,
        opt: &mut Optimizer,
        current: NodeIndex,
        new_expr: &mut HashMap<NodeIndex, HashSet<u32>>,
        changes: &AtomicCounter,
    ) -> bool {
        let mut changed = false;
        let dominators = opt.analysis::<Dominators>();

        let predecessors = opt.predecessors(current);
        if predecessors.len() > 1 {
            new_expr.entry(current).or_default();
            for pred in predecessors.iter() {
                new_expr.entry(*pred).or_default();
            }
            let sets = self.block_sets[&current].clone();
            let antic = &sets.antic_in;
            let phi_gen = &sets.phi_gen;
            let translated = predecessors
                .iter()
                .map(|pred| {
                    (
                        *pred,
                        phi_translate(opt, phi_gen, antic, current, *pred, &mut self.values),
                    )
                })
                .collect::<Vec<_>>();
            let partially_avail = translated
                .iter()
                .flat_map(|(pred, exprs)| {
                    let leaders = &self.block_sets[pred].leaders;
                    exprs
                        .iter()
                        .zip(antic)
                        .enumerate()
                        .filter(|(_, ((val, expr), (val_here, _)))| {
                            leaders.contains_key(val)
                                && !expr.is_simple()
                                && !new_expr[&current].contains(val_here)
                        })
                        .map(|it| it.0)
                })
                .collect::<HashSet<_>>();
            let mut new_phis = vec![Vec::default(); partially_avail.len()];
            for (pred, exprs) in translated {
                let mut i = 0;
                for (k, (val, expr)) in exprs.into_iter().enumerate() {
                    if !partially_avail.contains(&k) {
                        continue;
                    }
                    let leaders = &mut self.block_sets.get_mut(&pred).unwrap().leaders;
                    if !leaders.contains_key(&val) {
                        let new_temp = *opt.allocator.create_local(expr.item());
                        let new_op = ir::Instruction::new(expr.to_operation(leaders), new_temp);
                        opt.program[pred].ops.borrow_mut().push(new_op);
                        leaders.insert(val, value_of_var(&new_temp).unwrap());
                        new_expr.get_mut(&pred).unwrap().insert(val);
                        changed = true;
                        changes.inc();
                    }
                    let value = leaders.get(&val).unwrap();
                    new_phis[i].push(PhiEntry {
                        block: pred,
                        value: value.as_var(),
                    });
                    i += 1;
                }
            }
            let new_phis = new_phis
                .into_iter()
                .map(|entries| PhiInstruction {
                    out: *opt.allocator.create_local(entries[0].value.item),
                    entries,
                })
                .collect::<Vec<_>>();
            let mut phi_idx = 0;
            let leaders = &mut self.block_sets.get_mut(&current).unwrap().leaders;
            for (i, (val, _)) in antic.iter().enumerate() {
                if !partially_avail.contains(&i) {
                    continue;
                }
                let phi = &new_phis[phi_idx];
                let value = value_of_var(&phi.out).unwrap();
                self.values.insert_phi(phi, *val);
                leaders.insert(*val, value);
                new_expr.get_mut(&current).unwrap().insert(*val);
                phi_idx += 1;
            }
            opt.program[current].phi_nodes.borrow_mut().extend(new_phis);
        }

        let children = dominators
            .immediately_dominated_by(current)
            .collect::<Vec<_>>();
        for child in children {
            if child != current {
                let add_exprs = new_expr.entry(current).or_default().clone();
                for val in add_exprs.iter() {
                    let leader = self.block_sets[&current].leaders[val];
                    self.block_sets
                        .get_mut(&child)
                        .unwrap()
                        .leaders
                        .insert(*val, leader);
                }
                new_expr.entry(child).or_default().extend(add_exprs);
                changed |= self.insert_block(opt, child, new_expr, changes);
            }
        }

        changed
    }

    /// Find fully redundant expressions and replace them with trivial assignments. These can later
    /// be eliminated in a copy-propagation pass.
    pub fn eliminate(&mut self, opt: &mut Optimizer, changes: &AtomicCounter) {
        let changes_pre = changes.get();
        for block in opt.node_ids() {
            let leaders = &self.block_sets[&block].leaders;
            for op in opt.program[block].ops.borrow_mut().values_mut() {
                if let Some(leader) = self.values.lookup_op(op).and_then(|val| leaders.get(&val)) {
                    let var = leader.as_var();
                    let out = op.out;
                    if Some(var) != out {
                        op.operation = Operation::Copy(var);
                        changes.inc();
                    }
                }
            }
        }
        let eliminated = changes.get() - changes_pre;
        log::debug!("Eliminated {eliminated} redundant expressions");
    }
}
