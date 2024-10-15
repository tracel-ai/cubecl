use std::collections::{HashMap, HashSet, LinkedList};

use petgraph::{
    algo::dominators::{self},
    graph::NodeIndex,
    Graph,
};
use smallvec::SmallVec;

use crate::{BasicBlock, Optimizer};

use super::{convert::value_of_var, Expression, GvnPass, Value, ValueTable};

const MAX_SET_PASSES: usize = 10;

/// The set annotations for a given block
#[derive(Debug, Clone, Default)]
pub struct BlockSets {
    /// Expressions generated in this block
    pub exp_gen: LinkedList<(u32, Expression)>,
    /// Phi nodes that create new values in this block
    pub phi_gen: HashMap<u32, Value>,
    /// Temporaries that are assigned black box values (i.e. atomics, index to mutable array)
    pub tmp_gen: HashSet<Value>,
    /// The set of leaders for each value. This is the first temporary that contains the expression
    /// on any given path.
    pub leaders: HashMap<u32, Value>,

    /// The set of anticipated ("requested") expressions at the output point of the block
    pub antic_out: LinkedList<(u32, Expression)>,
    /// The set of anticipated ("requested") expressions at the input point of the block
    pub antic_in: LinkedList<(u32, Expression)>,
}

impl Default for GvnPass {
    fn default() -> Self {
        let mut dummy = Graph::<BasicBlock, ()>::new();
        let root = dummy.add_node(BasicBlock::default());
        Self {
            values: Default::default(),
            block_sets: Default::default(),
            dominators: dominators::simple_fast(&dummy, root),
            post_doms: dominators::simple_fast(&dummy, root),
        }
    }
}

impl GvnPass {
    /// Build set annotations for each block. Executes two steps:
    /// 1. Forward DFA that generates the available expressions, values and leaders for each block
    /// 2. Backward fixed-point DFA that generates the anticipated expressions/antileaders for each
    ///     block
    pub fn build_sets(&mut self, opt: &Optimizer) {
        self.build_block_sets_fwd(opt, self.dominators.root(), HashMap::new());
        let mut build_passes = 0;
        while self.build_block_sets_bckwd(opt, self.post_doms.root())
            && build_passes < MAX_SET_PASSES
        {
            build_passes += 1;
        }

        let global_leaders = self.values.value_numbers.iter();
        let global_leaders = global_leaders
            .filter(|(k, _)| {
                matches!(
                    k,
                    Value::Constant(_)
                        | Value::Input(_, _)
                        | Value::Scalar(_, _)
                        | Value::ConstArray(_, _, _)
                        | Value::Builtin(_)
                        | Value::Output(_, _)
                )
            })
            .map(|(k, v)| (*v, *k))
            .collect::<HashMap<_, _>>();
        for set in self.block_sets.values_mut() {
            set.leaders.extend(global_leaders.clone());
        }
    }

    fn build_block_sets_fwd(
        &mut self,
        opt: &Optimizer,
        block: NodeIndex,
        mut leaders: HashMap<u32, Value>,
    ) {
        let mut exp_gen = LinkedList::new();
        let mut phi_gen = HashMap::new();
        let mut tmp_gen = HashSet::new();
        let mut added_exprs = HashSet::new();

        for phi in opt.program[block].phi_nodes.borrow().iter() {
            let (num, val) = self.values.lookup_or_add_phi(phi);
            leaders.entry(num).or_insert(val);
            phi_gen.entry(num).or_insert(val);
        }

        for op in opt.program[block].ops.borrow().values() {
            match self
                .values
                .maybe_insert_op(op, &mut exp_gen, &mut added_exprs)
            {
                Ok((num, Some(val), _)) => {
                    leaders.entry(num).or_insert(val);
                }
                Err(Some(killed)) => {
                    tmp_gen.insert(killed);
                }
                _ => {}
            }
        }

        let sets = BlockSets {
            exp_gen,
            phi_gen,
            tmp_gen,
            leaders: leaders.clone(),

            antic_out: Default::default(),
            antic_in: Default::default(),
        };
        self.block_sets.insert(block, sets);
        let successors: Vec<_> = self.dominators.immediately_dominated_by(block).collect();
        for successor in successors {
            // Work around dominator bug
            if successor != block {
                self.build_block_sets_fwd(opt, successor, leaders.clone());
            }
        }
    }

    fn build_block_sets_bckwd(&mut self, opt: &Optimizer, current: NodeIndex) -> bool {
        let mut changed = false;

        let successors = opt.sucessors(current);
        // Since we have no critical edges, if successors > 1 then they must have only one entry,
        // So no phi nodes.
        #[allow(clippy::comparison_chain)]
        if successors.len() > 1 {
            let potential_out = &self.block_sets[&successors[0]].antic_in;
            let mut result = LinkedList::new();
            let rest = successors[1..]
                .iter()
                .map(|child| &self.block_sets[child].antic_in);
            for (val, expr) in potential_out {
                if rest
                    .clone()
                    .map(|child| child.iter().any(|v| v.0 == *val))
                    .all(|b| b)
                {
                    result.push_back((*val, expr.clone()));
                }
            }
            if self.block_sets[&current].antic_out != result {
                changed = true;
            }
            self.block_sets.get_mut(&current).unwrap().antic_out = result;
        } else if successors.len() == 1 {
            let child = successors[0];
            let antic_in_succ = &self.block_sets[&child].antic_in;
            let phi_gen = &self.block_sets[&child].phi_gen;
            let result = phi_translate(
                opt,
                phi_gen,
                antic_in_succ,
                child,
                current,
                &mut self.values,
            );
            if self.block_sets[&current].antic_out != result {
                changed = true;
            }
            self.block_sets.get_mut(&current).unwrap().antic_out = result;
        }

        let mut killed = HashSet::new();
        let cleaned = self.block_sets[&current]
            .exp_gen
            .iter()
            .chain(self.block_sets[&current].antic_out.iter())
            .filter_map(|(val, exp)| {
                for dependency in exp.depends_on() {
                    if killed.contains(&dependency) {
                        killed.insert(*val);
                        return None;
                    }
                }
                if let Expression::Volatile(_) = exp {
                    killed.insert(*val);
                    return None;
                }
                Some((*val, exp.clone()))
            });
        let mut added = HashSet::new();
        let mut result = LinkedList::new();
        for v in cleaned {
            if !added.contains(&v.0) {
                added.insert(v.0);
                result.push_back(v);
            }
        }
        if self.block_sets[&current].antic_in != result {
            changed = true;
        }
        self.block_sets.get_mut(&current).unwrap().antic_in = result;

        let predecessors: Vec<_> = self.post_doms.immediately_dominated_by(current).collect();
        for predecessor in predecessors {
            // Work around dominator bug
            if predecessor != current {
                changed |= self.build_block_sets_bckwd(opt, predecessor);
            }
        }
        changed
    }
}

pub fn phi_translate(
    opt: &Optimizer,
    phi_gen: &HashMap<u32, Value>,
    antic: &LinkedList<(u32, Expression)>,
    child: NodeIndex,
    parent: NodeIndex,
    values: &mut ValueTable,
) -> LinkedList<(u32, Expression)> {
    let mut result = LinkedList::new();
    let mut translated = HashMap::new();

    for phi in opt.block(child).phi_nodes.borrow().iter() {
        let (num, _) = values.lookup_or_add_phi(phi);
        let here = phi.entries.iter().find(|it| it.block == parent).unwrap();
        let num_here = values.lookup_or_add_var(&here.value).unwrap();
        translated.insert(num, num_here);
    }

    for (val, expr) in antic {
        if let Some(value) = phi_gen.get(val) {
            let nodes = opt.block(child).phi_nodes.borrow();
            let phi = nodes
                .iter()
                .find(|it| &value_of_var(&it.out).unwrap() == value);

            if let Some(phi) = phi {
                let value_here = phi.entries.iter().find(|it| it.block == parent).unwrap();
                let value_here = value_of_var(&value_here.value).unwrap();
                let num_here = values.lookup_or_add_expr(Expression::Value(value_here), None);
                result.push_back((num_here, Expression::Value(value_here)));
                translated.insert(*val, num_here);
            }
        } else {
            let t = |val: &u32| *translated.get(val).unwrap_or(val);

            let updated = match expr {
                Expression::Instruction(inst) => {
                    let args = inst.args.iter().map(t).collect::<SmallVec<[u32; 4]>>();
                    let mut inst = inst.clone();
                    inst.args = args;
                    Expression::Instruction(inst)
                }
                Expression::Copy(val, item) => Expression::Copy(t(val), *item),
                Expression::Phi(_) => continue,
                other => other.clone(),
            };
            let updated_val = values.lookup_or_add_expr(updated.clone(), None);
            result.push_back((updated_val, updated));
            translated.insert(*val, updated_val);
        }
    }
    result
}
