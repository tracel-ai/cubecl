use std::{
    collections::{HashMap, VecDeque},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use cubecl_core::ir::{self as core, Operator, Variable};
use cubecl_core::ir::{Item, Operation, Scope};
use passes::{CompositeMerge, EliminateUnusedVariables, InlineAssignments, OptimizationPass};
use petgraph::{prelude::StableDiGraph, visit::EdgeRef, Direction};

mod block;
mod control_flow;
mod debug;
mod instructions;
mod passes;
mod phi_frontiers;
mod version;

pub use block::*;
pub use control_flow::*;
pub use petgraph::graph::NodeIndex;

#[derive(Clone, Debug, Default)]
pub struct AtomicCounter {
    inner: Rc<AtomicUsize>,
}

impl AtomicCounter {
    pub fn new(val: usize) -> Self {
        Self {
            inner: Rc::new(AtomicUsize::new(val)),
        }
    }

    pub fn inc(&self) -> usize {
        self.inner.fetch_add(1, Ordering::AcqRel)
    }

    pub fn get(&self) -> usize {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Default, Debug, Clone)]
struct Program {
    pub variables: HashMap<(u16, u8), Item>,
    pub graph: StableDiGraph<BasicBlock, u32>,
    root: NodeIndex,
}

impl Deref for Program {
    type Target = StableDiGraph<BasicBlock, u32>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Program {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

#[derive(Default, Debug, Clone)]
pub struct Optimizer {
    program: Program,
    pub current_block: Option<NodeIndex>,
    loop_break: VecDeque<NodeIndex>,
    pub ret: NodeIndex,
    edge_id: Rc<AtomicU32>,
}

impl Optimizer {
    pub fn new(expand: Scope) -> Self {
        let mut opt = Self::default();
        opt.parse_graph(expand);
        opt.analyze_liveness();
        opt.apply_pre_ssa_passes();
        opt.exempt_index_assign_locals();
        opt.ssa_transform();
        opt.apply_post_ssa_passes();
        opt
    }

    pub fn entry(&self) -> NodeIndex {
        self.program.root
    }

    fn edge_id(&self) -> u32 {
        self.edge_id.fetch_add(1, Ordering::AcqRel)
    }

    fn parse_graph(&mut self, scope: Scope) {
        let entry = self.program.add_node(BasicBlock::default());
        self.program.root = entry;
        self.current_block = Some(entry);
        self.ret = self.program.add_node(BasicBlock::default());
        *self.program[self.ret].control_flow.borrow_mut() = ControlFlow::Return;
        self.parse_scope(scope);
        if let Some(current_block) = self.current_block {
            let edge_id = self.edge_id();
            self.program.add_edge(current_block, self.ret, edge_id);
        }
    }

    fn apply_pre_ssa_passes(&mut self) {
        let mut passes = vec![CompositeMerge];
        loop {
            let counter = AtomicCounter::default();

            for pass in &mut passes {
                pass.apply_pre_ssa(self, counter.clone());
            }

            if counter.get() == 0 {
                break;
            }
        }
    }

    fn apply_post_ssa_passes(&mut self) {
        let mut passes: Vec<Box<dyn OptimizationPass>> = vec![
            Box::new(InlineAssignments),
            Box::new(EliminateUnusedVariables),
        ];
        loop {
            let counter = AtomicCounter::default();
            for pass in &mut passes {
                pass.apply_post_ssa(self, counter.clone());
            }

            if counter.get() == 0 {
                break;
            }
        }
    }

    /// Remove non-constant index vectors from SSA transformation because they currently must be
    /// mutated
    fn exempt_index_assign_locals(&mut self) {
        for node in self.node_ids() {
            let ops = self.program[node].ops.clone();
            for op in ops.borrow().values() {
                if let Operation::Operator(Operator::IndexAssign(op)) = op {
                    if let Variable::Local { id, depth, .. } = &op.out {
                        self.program.variables.remove(&(*id, *depth));
                    }
                }
            }
        }
    }

    fn node_ids(&self) -> Vec<NodeIndex> {
        self.program.node_indices().collect()
    }

    fn ssa_transform(&mut self) {
        self.program.fill_dom_frontiers();
        self.program.place_phi_nodes();
        self.version_program();
    }

    pub fn current_block_mut(&mut self) -> &mut BasicBlock {
        &mut self.program[self.current_block.unwrap()]
    }

    pub fn current_block(&self) -> &BasicBlock {
        &self.program[self.current_block.unwrap()]
    }

    pub fn predecessors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.program
            .edges_directed(block, Direction::Incoming)
            .map(|it| it.source())
            .collect()
    }

    pub fn sucessors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.program
            .edges_directed(block, Direction::Outgoing)
            .map(|it| it.target())
            .collect()
    }

    pub fn block(&self, block: NodeIndex) -> &BasicBlock {
        &self.program[block]
    }

    pub fn parse_scope(&mut self, mut scope: Scope) {
        let processed = scope.process();

        for var in processed.variables {
            if let Variable::Local { id, item, depth } = var {
                self.program.variables.insert((id, depth), item);
            }
        }

        for instruction in processed.operations {
            match instruction {
                Operation::Branch(branch) => self.parse_control_flow(branch),
                mut other => {
                    self.visit_operation(&mut other, |_, _| {}, |opt, var| opt.write_var(var));
                    self.current_block_mut().ops.borrow_mut().push(other);
                }
            }
        }
    }

    pub fn local_variable_id(&mut self, variable: &core::Variable) -> Option<(u16, u8)> {
        match variable {
            core::Variable::Local { id, depth, item } if !item.elem.is_atomic() => {
                Some((*id, *depth))
            }
            _ => None,
        }
    }
}

pub fn visit_noop(_opt: &mut Optimizer, _var: &mut Variable) {}

#[cfg(test)]
mod tests {
    use cubecl_core::cube;
    use cubecl_core::{self as cubecl, ir::HybridAllocator, prelude::CubeContext};

    use super::Optimizer;

    #[cube]
    fn test_if_kernel() -> u32 {
        let mut cond: bool = true;
        if cond {
            cond = false;
        }
        cond as u32
    }

    #[cube]
    fn test_while_kernel() -> u32 {
        let mut i = 0;
        while i < 4 {
            i += 1;
        }
        i
    }

    #[test]
    fn test_if() {
        let mut context = CubeContext::root(HybridAllocator::default());
        test_if_kernel::expand(&mut context);
        let opt = Optimizer::new(context.into_scope());

        panic!("{opt}");
    }

    #[test]
    fn test_while() {
        let mut context = CubeContext::root(HybridAllocator::default());
        test_while_kernel::expand(&mut context);
        let opt = Optimizer::new(context.into_scope());

        panic!("{opt}");
    }
}
