use std::{
    collections::{HashMap, VecDeque},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use cubecl_core::ir::{self as core, Operator, Procedure, Variable};
use cubecl_core::ir::{Item, Operation, Scope};
use passes::{
    CompositeMerge, ConstEval, ConstOperandSimplify, CopyPropagateArray, CopyTransform,
    EliminateConstBranches, EliminateDeadBlocks, EliminateUnusedVariables, InlineAssignments,
    MergeSameExpressions, OptimizationPass, RemoveIndexScalar,
};
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

#[derive(Debug, Clone)]
pub struct Optimizer {
    program: Program,
    pub current_block: Option<NodeIndex>,
    loop_break: VecDeque<NodeIndex>,
    pub ret: NodeIndex,
    edge_id: Rc<AtomicU32>,
    root_scope: Scope,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            program: Default::default(),
            current_block: Default::default(),
            loop_break: Default::default(),
            ret: Default::default(),
            edge_id: Default::default(),
            root_scope: Scope::root(),
        }
    }
}

impl Optimizer {
    pub fn new(expand: Scope) -> Self {
        let mut opt = Self {
            root_scope: expand.clone(),
            ..Default::default()
        };
        opt.run_opt(expand);

        opt
    }

    fn run_opt(&mut self, expand: Scope) {
        self.parse_graph(expand);
        self.analyze_liveness();
        self.apply_pre_ssa_passes();
        self.exempt_index_assign_locals();
        self.ssa_transform();
        self.apply_post_ssa_passes();
        let arrays_prop = AtomicCounter::new(0);
        CopyPropagateArray.apply_post_ssa(self, arrays_prop.clone());
        if arrays_prop.get() > 0 {
            self.analyze_liveness();
            self.ssa_transform();
            self.apply_post_ssa_passes();
        }
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
            Box::new(ConstOperandSimplify),
            Box::new(MergeSameExpressions),
            Box::new(ConstEval),
            Box::new(RemoveIndexScalar),
            Box::new(CopyTransform),
            Box::new(EliminateConstBranches),
            Box::new(EliminateDeadBlocks),
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
                if let Operation::Operator(Operator::IndexAssign(binop)) = op {
                    if let Variable::Local { id, depth, .. } = &binop.out {
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
        self.program.variables.clear();
        for block in self.node_ids() {
            self.program[block].writes.clear();
        }
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
                Operation::Procedure(proc) => self.compile_procedure(proc, scope.clone()),
                mut other => {
                    self.visit_operation(&mut other, |_, _| {}, |opt, var| opt.write_var(var));
                    self.current_block_mut().ops.borrow_mut().push(other);
                }
            }
        }
    }

    fn compile_procedure(&mut self, proc: Procedure, mut scope: Scope) {
        let mut compile = |scope: Scope| {
            self.parse_scope(scope);
        };

        match proc {
            Procedure::ReadGlobalWithLayout(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::ReadGlobal(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::WriteGlobal(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::ConditionalAssign(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::CheckedIndex(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::CheckedIndexAssign(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::IndexOffsetGlobalWithLayout(proc) => {
                proc.expand(&mut scope);
                compile(scope);
            }
            Procedure::EarlyReturn(proc) => {
                proc.expand(&mut scope);
                compile(scope);
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
