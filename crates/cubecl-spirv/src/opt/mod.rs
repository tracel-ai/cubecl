use std::{
    collections::{HashMap, HashSet, VecDeque},
    mem::transmute,
    ops::{Deref, DerefMut},
};

use cubecl_core::ir::{
    self as core, Branch, ConstantScalarValue, If, IfElse, Loop, RangeLoop, Switch, Variable,
};
use cubecl_core::ir::{BinaryOperator, Elem, Item, Operation, Operator, Scope};
use expand::ExpandState;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

mod expand;
mod instructions;
mod threshold;

#[derive(Default, Debug, Serialize, Deserialize)]
struct Program {
    pub variables: HashSet<(u16, u8)>,
    pub graph: DiGraph<BasicBlock, ()>,
    root: NodeIndex,
}
type Ident = u32;

impl Deref for Program {
    type Target = DiGraph<BasicBlock, ()>;

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
        continue_target: NodeIndex,
        merge: NodeIndex,
    },
    Return,
    #[default]
    None,
}

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct BasicBlock {
    phi_nodes: Vec<(u16, u8)>,
    writes: HashSet<(u16, u8)>,
    dom_frontiers: HashSet<NodeIndex>,
    ops: Vec<Operation>,
    control_flow: ControlFlow,
    expand: ExpandState,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Optimizer {
    id: u32,
    program: Program,
    current_block: Option<NodeIndex>,
    loop_break: VecDeque<NodeIndex>,
    ret: NodeIndex,
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
            opt.program.add_edge(current_block, opt.ret, ());
        }
        opt.program.fill_dom_frontiers();
        opt.program.place_phi_nodes();
        opt
    }

    pub fn current_block(&mut self) -> &mut BasicBlock {
        &mut self.program[self.current_block.unwrap()]
    }

    pub fn parse_scope(&mut self, scope: Scope) {
        let processed = scope.clone().process();

        for instruction in processed.operations {
            match instruction {
                Operation::Branch(branch) => self.parse_control_flow(branch),
                mut other => {
                    self.find_writes(&mut other);
                    self.current_block().ops.push(other);
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
                self.current_block().ops.push(Branch::Select(select).into());
            }
            Branch::Switch(switch) => self.parse_switch(switch),
            Branch::RangeLoop(range_loop) => {
                self.parse_for_loop(range_loop);
            }
            Branch::Loop(loop_) => self.parse_loop(loop_),
            Branch::Return => {
                let current_block = self.current_block.take().unwrap();
                self.program.add_edge(current_block, self.ret, ());
            }
            Branch::Break => {
                let current_block = self.current_block.take().unwrap();
                let loop_break = self.loop_break.back().expect("Can't break outside loop");
                self.program.add_edge(current_block, *loop_break, ());
            }
        }
    }

    pub fn parse_if(&mut self, mut if_: If) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.read_var(&mut if_.cond);

        self.program[current_block].control_flow = ControlFlow::If {
            cond: if_.cond,
            then,
            merge: next,
        };

        self.program.add_edge(current_block, then, ());
        self.program.add_edge(current_block, next, ());

        self.current_block = Some(then);
        self.parse_scope(if_.scope);
        if self.current_block.is_some() {
            self.program.add_edge(then, next, ());
        }
        self.current_block = Some(next);
    }

    pub fn parse_if_else(&mut self, mut if_else: IfElse) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let or_else = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.read_var(&mut if_else.cond);

        self.program[current_block].control_flow = ControlFlow::IfElse {
            cond: if_else.cond,
            then,
            or_else,
            merge: next,
        };

        self.program.add_edge(current_block, then, ());
        self.program.add_edge(current_block, or_else, ());

        self.current_block = Some(then);
        self.parse_scope(if_else.scope_if);

        if self.current_block.is_some() {
            self.program.add_edge(then, next, ());
        }

        self.current_block = Some(or_else);
        self.parse_scope(if_else.scope_else);

        if self.current_block.is_some() {
            self.program.add_edge(or_else, next, ());
        }

        self.current_block = Some(next);
    }

    pub fn parse_switch(&mut self, mut switch: Switch) {
        let current_block = self.current_block.unwrap();
        let next = self.program.add_node(BasicBlock::default());
        self.read_var(&mut switch.value);

        let branches = switch
            .cases
            .into_iter()
            .map(|(val, case)| {
                let case_id = self.program.add_node(BasicBlock::default());
                self.program.add_edge(current_block, case_id, ());
                self.current_block = Some(case_id);
                self.parse_scope(case);
                if self.current_block.is_some() {
                    self.program.add_edge(case_id, next, ());
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
        self.program.add_edge(current_block, default, ());
        self.current_block = Some(default);
        self.parse_scope(switch.scope_default);

        if self.current_block.is_some() {
            self.program.add_edge(default, next, ());
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
        self.program.add_edge(current_block, header, ());

        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program.add_edge(header, body, ());

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(loop_.scope);
        let continue_target = self.program.add_node(BasicBlock::default());

        self.loop_break.pop_back();

        if self.current_block.is_some() {
            self.program.add_edge(body, continue_target, ());
        }

        self.program.add_edge(continue_target, header, ());

        self.program[header].control_flow = ControlFlow::Loop {
            continue_target,
            merge: next,
        };
    }

    fn parse_for_loop(&mut self, mut range_loop: RangeLoop) {
        let mut step = range_loop
            .step
            .unwrap_or(Variable::ConstantScalar(ConstantScalarValue::UInt(1)));

        self.read_var(&mut range_loop.start);
        self.read_var(&mut range_loop.end);
        self.read_var(&mut step);

        let (id, depth) = match range_loop.i {
            Variable::LocalBinding { id, depth, .. } => (id, depth),
            _ => unreachable!(),
        };
        let i = range_loop.i;

        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, header, ());

        let break_cond = self.program.add_node(BasicBlock::default());
        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program.add_edge(header, break_cond, ());
        self.program.add_edge(break_cond, next, ());
        self.program.add_edge(break_cond, body, ());

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(range_loop.scope);
        let continue_target = self.program.add_node(BasicBlock::default());

        self.loop_break.pop_back();

        if self.current_block.is_some() {
            self.program.add_edge(body, continue_target, ());
        }

        self.program.add_edge(continue_target, header, ());

        self.program[header].control_flow = ControlFlow::Loop {
            continue_target,
            merge: next,
        };

        // For loop constructs
        self.program[header].phi_nodes.push((id, depth));
        {
            let op = match range_loop.inclusive {
                true => Operator::LowerEqual,
                false => Operator::Lower,
            };
            let tmp_id = self.id();
            let tmp = Variable::LocalBinding {
                id: 60000 + tmp_id as u16,
                item: Item::new(Elem::Bool),
                depth: 0,
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

    fn id(&mut self) -> Ident {
        self.id += 1;
        self.id
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

    #[test]
    fn test_if() {
        let mut context = CubeContext::root(HybridAllocator::default());
        test_if_kernel::expand(&mut context);
        let opt = Optimizer::new(context.into_scope());
        let blocks = opt
            .program
            .node_indices()
            .map(|index| opt.program[index].clone())
            .collect::<Vec<_>>();

        panic!("{blocks:#?}");
    }
}
