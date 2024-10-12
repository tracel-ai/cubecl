use std::collections::{HashMap, HashSet, LinkedList};

use petgraph::{
    algo::dominators::{self, Dominators},
    graph::NodeIndex,
};

use crate::Optimizer;

use super::{value_of_var, Expression, Value, ValueTable};

#[derive(Debug)]
pub struct GvnPass {
    table: ValueTable,
    block_sets: HashMap<NodeIndex, BlockSets>,
    dominators: Dominators<NodeIndex>,
    post_doms: Dominators<NodeIndex>,
}

#[derive(Debug)]
struct BlockSets {
    exp_gen: LinkedList<(u32, Expression)>,
    phi_gen: HashMap<u32, Value>,
    tmp_gen: HashSet<Value>,
    leaders: HashMap<u32, Value>,

    antic_out: LinkedList<(u32, Expression)>,
    antic_in: LinkedList<(u32, Expression)>,
}

impl GvnPass {
    pub fn new(opt: &Optimizer) -> GvnPass {
        let doms = dominators::simple_fast(&opt.program.graph, opt.entry());
        let mut rev_graph = opt.program.graph.clone();
        rev_graph.reverse();
        let post_doms = dominators::simple_fast(&rev_graph, opt.ret);
        GvnPass {
            table: Default::default(),
            block_sets: Default::default(),
            dominators: doms,
            post_doms,
        }
    }

    pub fn build_sets(&mut self, opt: &Optimizer) {
        self.build_block_sets_fwd(opt, self.dominators.root(), HashMap::new());
        self.build_block_sets_bckwd(opt, self.post_doms.root());
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
            let value = value_of_var(&phi.out).unwrap();
            let num = self.table.lookup_or_add_var(&phi.out).unwrap();
            phi_gen.insert(num, value);
        }

        for op in opt.program[block].ops.borrow().values() {
            match self
                .table
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
            self.build_block_sets_fwd(opt, successor, leaders.clone());
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
            let result =
                phi_translate(opt, phi_gen, antic_in_succ, child, current, &mut self.table);
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
            changed |= self.build_block_sets_bckwd(opt, predecessor);
        }
        changed
    }
}

fn phi_translate(
    opt: &Optimizer,
    phi_gen: &HashMap<u32, Value>,
    antic: &LinkedList<(u32, Expression)>,
    child: NodeIndex,
    parent: NodeIndex,
    values: &mut ValueTable,
) -> LinkedList<(u32, Expression)> {
    let mut result = LinkedList::new();
    let mut translated = HashMap::new();

    for (val, _) in antic {
        if let Some(val) = phi_gen.get(val) {
            let nodes = opt.block(child).phi_nodes.borrow();
            let phi = nodes
                .iter()
                .find(|it| &value_of_var(&it.out).unwrap() == val);
            if let Some(phi) = phi {
                let value_here = phi.entries.iter().find(|it| it.block == parent).unwrap();
                let value_here = value_of_var(&value_here.value).unwrap();
                let num_here = values.lookup_or_add_expr(Expression::Value(value_here), None);
                result.push_back((num_here, Expression::Value(value_here)));
                translated.insert(*val, value_here);
            }
        }
    }
    result
}
