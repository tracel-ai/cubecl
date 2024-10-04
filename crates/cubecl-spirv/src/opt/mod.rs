use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem::transmute,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
};

use cubecl_core::ir::{
    self as core, Branch, ConstantScalarValue, If, IfElse, Loop, RangeLoop, Switch, Variable,
};
use cubecl_core::ir::{BinaryOperator, Elem, Item, Operation, Operator, Scope, UnaryOperator};
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef, Direction};
use serde::{Deserialize, Serialize};
use version::PhiInstruction;

mod debug;
mod instructions;
mod pass;
mod phi_frontiers;
mod version;

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
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

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub enum ControlFlow {
    If {
        cond: Variable,
        then: NodeIndex,
        merge: NodeIndex,
    },
    IfElse {
        cond: Variable,
        then: NodeIndex,
        or_else: NodeIndex,
        merge: NodeIndex,
    },
    Switch {
        value: Variable,
        default: NodeIndex,
        branches: Vec<(u32, NodeIndex)>,
        merge: NodeIndex,
    },
    Loop {
        body: NodeIndex,
        continue_target: NodeIndex,
        merge: NodeIndex,
    },
    Return,
    #[default]
    None,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct BasicBlock {
    annotations: HashSet<Annotation>,
    pub phi_nodes: Vec<PhiInstruction>,
    writes: HashSet<(u16, u8)>,
    dom_frontiers: HashSet<NodeIndex>,
    pub ops: Vec<Operation>,
    pub control_flow: ControlFlow,
}

/// Annotations to prevent merging some kinds of blocks and allowing update of others
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub enum Annotation {
    ContinueTarget,
    /// Merge target from block x
    Merge(NodeIndex),
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
        let entry = opt.program.add_node(BasicBlock::default());
        opt.program.root = entry;
        opt.current_block = Some(entry);
        opt.ret = opt.program.add_node(BasicBlock::default());
        opt.program[opt.ret].control_flow = ControlFlow::Return;
        opt.parse_scope(expand);
        if let Some(current_block) = opt.current_block {
            let edge_id = opt.edge_id();
            opt.program.add_edge(current_block, opt.ret, edge_id);
        }
        opt.program.fill_dom_frontiers();
        opt.program.place_phi_nodes();
        opt.version_program();
        opt
    }

    pub fn entry(&self) -> NodeIndex {
        self.program.root
    }

    fn edge_id(&self) -> u32 {
        self.edge_id.fetch_add(1, Ordering::AcqRel)
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

    pub fn annotate(&mut self, block: NodeIndex, annotation: Annotation) {
        self.program[block].annotations.insert(annotation);
    }

    pub fn block(&self, block: NodeIndex) -> &BasicBlock {
        &self.program[block]
    }

    pub fn parse_scope(&mut self, mut scope: Scope) {
        let processed = scope.process();

        for var in processed.variables {
            match var {
                Variable::Local {
                    id,
                    item:
                        Item {
                            elem,
                            vectorization: Some(vec),
                        },
                    depth,
                } if vec.get() > 1 => {
                    let out = Variable::Local {
                        id,
                        item: Item::vectorized(elem, Some(vec)),
                        depth,
                    };
                    let mut init = Operator::Assign(UnaryOperator {
                        input: Variable::ConstantScalar(ConstantScalarValue::UInt(0)),
                        out,
                    })
                    .into();
                    self.visit_operation(&mut init, |_, _| {}, |opt, var| opt.write_var(var));
                    self.current_block_mut().ops.push(init);
                }
                Variable::Local { id, item, depth } => {
                    self.program.variables.insert((id, depth), item);
                }
                _ => {}
            }
        }

        for instruction in processed.operations {
            match instruction {
                Operation::Branch(branch) => self.parse_control_flow(branch),
                mut other => {
                    self.visit_operation(&mut other, |_, _| {}, |opt, var| opt.write_var(var));
                    self.current_block_mut().ops.push(other);
                }
            }
        }
    }

    pub fn parse_control_flow(&mut self, branch: Branch) {
        match branch {
            Branch::If(if_) => self.parse_if(if_),
            Branch::IfElse(if_else) => self.parse_if_else(if_else),
            Branch::Select(mut select) => {
                self.find_writes_select(&mut select);
                self.current_block_mut()
                    .ops
                    .push(Branch::Select(select).into());
            }
            Branch::Switch(switch) => self.parse_switch(switch),
            Branch::RangeLoop(range_loop) => {
                self.parse_for_loop(range_loop);
            }
            Branch::Loop(loop_) => self.parse_loop(loop_),
            Branch::Return => {
                let current_block = self.current_block.take().unwrap();
                let id = self.edge_id();
                self.program.add_edge(current_block, self.ret, id);
            }
            Branch::Break => {
                let current_block = self.current_block.take().unwrap();
                let loop_break = self.loop_break.back().expect("Can't break outside loop");
                let id = self.edge_id();
                self.program.add_edge(current_block, *loop_break, id);
            }
        }
    }

    pub fn parse_if(&mut self, if_: If) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());
        self.annotate(next, Annotation::Merge(current_block));

        self.program[current_block].control_flow = ControlFlow::If {
            cond: if_.cond,
            then,
            merge: next,
        };

        let id = self.edge_id();
        self.program.add_edge(current_block, then, id);
        let id = self.edge_id();
        self.program.add_edge(current_block, next, id);

        self.current_block = Some(then);
        self.parse_scope(if_.scope);
        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, next, id);
        }
        self.current_block = Some(next);
    }

    pub fn parse_if_else(&mut self, if_else: IfElse) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let or_else = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program[current_block].control_flow = ControlFlow::IfElse {
            cond: if_else.cond,
            then,
            or_else,
            merge: next,
        };

        let id = self.edge_id();
        self.program.add_edge(current_block, then, id);
        let id = self.edge_id();
        self.program.add_edge(current_block, or_else, id);

        self.current_block = Some(then);
        self.parse_scope(if_else.scope_if);

        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, next, id);
        }

        self.current_block = Some(or_else);
        self.parse_scope(if_else.scope_else);

        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, next, id);
        }

        self.current_block = Some(next);
    }

    pub fn parse_switch(&mut self, switch: Switch) {
        let current_block = self.current_block.unwrap();
        let next = self.program.add_node(BasicBlock::default());

        let branches = switch
            .cases
            .into_iter()
            .map(|(val, case)| {
                let case_id = self.program.add_node(BasicBlock::default());
                let id = self.edge_id();
                self.program.add_edge(current_block, case_id, id);
                self.current_block = Some(case_id);
                self.parse_scope(case);
                if let Some(current_block) = self.current_block {
                    let id = self.edge_id();
                    self.program.add_edge(current_block, next, id);
                }
                let val = match val.as_const().expect("Switch value must be constant") {
                    core::ConstantScalarValue::Int(val, _) => unsafe {
                        transmute::<i32, u32>(val as i32)
                    },
                    core::ConstantScalarValue::UInt(val) => val as u32,
                    _ => unreachable!("Switch cases must be integer"),
                };
                (val, case_id)
            })
            .collect::<Vec<_>>();

        let default = self.program.add_node(BasicBlock::default());
        let id = self.edge_id();
        self.program.add_edge(current_block, default, id);
        self.current_block = Some(default);
        self.parse_scope(switch.scope_default);

        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, next, id);
        }

        self.program[current_block].control_flow = ControlFlow::Switch {
            value: switch.value,
            default,
            branches,
            merge: next,
        };

        self.current_block = Some(next);
    }

    fn parse_loop(&mut self, loop_: Loop) {
        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        let id = self.edge_id();
        self.program.add_edge(current_block, header, id);

        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        let id = self.edge_id();
        self.program.add_edge(header, body, id);

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(loop_.scope);
        let continue_target = self.program.add_node(BasicBlock::default());

        self.loop_break.pop_back();

        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, continue_target, id);
        }

        let id = self.edge_id();
        self.program.add_edge(continue_target, header, id);

        self.program[header].control_flow = ControlFlow::Loop {
            body,
            continue_target,
            merge: next,
        };
        self.current_block = Some(next);
    }

    fn parse_for_loop(&mut self, range_loop: RangeLoop) {
        let step = range_loop
            .step
            .unwrap_or(Variable::ConstantScalar(ConstantScalarValue::UInt(1)));

        let i_id = match range_loop.i {
            Variable::LocalBinding { id, depth, .. } => (id, depth),
            _ => unreachable!(),
        };
        let i = range_loop.i;
        self.program.variables.insert(i_id, i.item());

        let mut assign = Operator::Assign(UnaryOperator {
            input: range_loop.start,
            out: i,
        })
        .into();
        self.visit_operation(&mut assign, |_, _| {}, |opt, var| opt.write_var(var));
        self.current_block_mut().ops.push(assign);

        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        let id = self.edge_id();
        self.program.add_edge(current_block, header, id);

        let break_cond = self.program.add_node(BasicBlock::default());
        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        let id = self.edge_id();
        self.program.add_edge(header, break_cond, id);
        let id = self.edge_id();
        self.program.add_edge(break_cond, next, id);
        let id = self.edge_id();
        self.program.add_edge(break_cond, body, id);

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(range_loop.scope);
        let continue_target = self.program.add_node(BasicBlock::default());

        self.loop_break.pop_back();

        if let Some(current_block) = self.current_block {
            let id = self.edge_id();
            self.program.add_edge(current_block, continue_target, id);
        }

        let id = self.edge_id();
        self.program.add_edge(continue_target, header, id);

        self.program[header].control_flow = ControlFlow::Loop {
            body: break_cond,
            continue_target,
            merge: next,
        };
        self.current_block = Some(next);

        // For loop constructs
        self.program
            .insert_phi(header, i_id, range_loop.start.item());
        {
            let op = match range_loop.inclusive {
                true => Operator::LowerEqual,
                false => Operator::Lower,
            };
            let tmp = Variable::Local {
                id: 60000 + i_id.0,
                item: Item::new(Elem::Bool),
                depth: i_id.1,
            };
            self.program[break_cond].ops.push(
                op(BinaryOperator {
                    lhs: i,
                    rhs: range_loop.end,
                    out: tmp,
                })
                .into(),
            );
            self.program[break_cond].control_flow = ControlFlow::If {
                cond: tmp,
                then: body,
                merge: next,
            };
        }
        self.program[continue_target].ops.push(
            Operator::Add(BinaryOperator {
                lhs: i,
                rhs: step,
                out: i,
            })
            .into(),
        );
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
