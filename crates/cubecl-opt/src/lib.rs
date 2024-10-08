use std::{
    collections::{HashMap, VecDeque},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use cubecl_core::{
    ir::{self as core, Operator, Procedure, Variable},
    CubeDim,
};
use cubecl_core::{
    ir::{Item, Operation, Scope},
    ExecutionMode,
};
use passes::{
    CompositeMerge, ConstEval, ConstOperandSimplify, CopyPropagateArray, CopyTransform,
    EliminateConstBranches, EliminateDeadBlocks, EliminateUnusedVariables, FindConstSliceLen,
    InBoundsToUnchecked, InlineAssignments, IntegerRangeAnalysis, MergeSameExpressions,
    OptimizationPass, RemoveIndexScalar,
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
pub use petgraph::graph::{EdgeIndex, NodeIndex};

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

#[derive(Debug, Clone)]
pub(crate) struct Slice {
    pub(crate) start: Variable,
    pub(crate) end: Variable,
    pub(crate) end_op: Option<Operation>,
    pub(crate) const_len: Option<u32>,
}

#[derive(Default, Debug, Clone)]
struct Program {
    pub variables: HashMap<(u16, u8), Item>,
    pub slices: HashMap<(u16, u8), Slice>,
    pub graph: StableDiGraph<BasicBlock, ()>,
    root: NodeIndex,
    int_ranges: HashMap<VarId, Range>,
}

impl Deref for Program {
    type Target = StableDiGraph<BasicBlock, ()>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Program {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

type VarId = (u16, u8, u16);

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Range {
    lower_bound: Option<i64>,
    upper_bound: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct Optimizer {
    program: Program,
    pub current_block: Option<NodeIndex>,
    loop_break: VecDeque<NodeIndex>,
    pub ret: NodeIndex,
    root_scope: Scope,
    pub(crate) cube_dim: CubeDim,
    pub(crate) mode: ExecutionMode,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            program: Default::default(),
            current_block: Default::default(),
            loop_break: Default::default(),
            ret: Default::default(),
            root_scope: Scope::root(),
            cube_dim: Default::default(),
            mode: Default::default(),
        }
    }
}

impl Optimizer {
    pub fn new(expand: Scope, cube_dim: CubeDim, mode: ExecutionMode) -> Self {
        let mut opt = Self {
            root_scope: expand.clone(),
            cube_dim,
            mode,
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

    fn parse_graph(&mut self, scope: Scope) {
        let entry = self.program.add_node(BasicBlock::default());
        self.program.root = entry;
        self.current_block = Some(entry);
        self.ret = self.program.add_node(BasicBlock::default());
        *self.program[self.ret].control_flow.borrow_mut() = ControlFlow::Return;
        self.parse_scope(scope);
        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, self.ret, ());
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
            Box::new(EliminateConstBranches),
            Box::new(EliminateDeadBlocks),
        ];
        let checked_passes: Vec<Box<dyn OptimizationPass>> = vec![
            Box::new(IntegerRangeAnalysis),
            Box::new(FindConstSliceLen),
            Box::new(InBoundsToUnchecked),
        ];
        if matches!(self.mode, ExecutionMode::Checked) {
            passes.extend(checked_passes);
        }
        passes.push(Box::new(CopyTransform));

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
                Operation::Operator(Operator::Slice(slice_op)) => {
                    let out_id = match &slice_op.out {
                        Variable::Slice { id, depth, .. } => (*id, *depth),
                        _ => unreachable!(),
                    };
                    self.program.slices.insert(
                        out_id,
                        Slice {
                            start: slice_op.start,
                            end: slice_op.end,
                            end_op: None,
                            const_len: None,
                        },
                    );
                    let mut op = Operation::Operator(Operator::Slice(slice_op));
                    self.visit_operation(&mut op, |_, _| {}, |opt, var| opt.write_var(var));
                    self.current_block_mut().ops.borrow_mut().push(op);
                }
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
