#![allow(unknown_lints, unnecessary_transmutes)]

use std::mem::transmute;

use crate::{BasicBlock, BlockUse, NodeIndex, Optimizer};
use cubecl_ir::{
    Arithmetic, BinaryOperator, Branch, Comparison, ConstantValue, ElemType, If, IfElse,
    Instruction, Loop, Marker, Operation, RangeLoop, Switch, Type, Variable, VariableKind,
};
use petgraph::{Direction, graph::EdgeIndex, visit::EdgeRef};
use stable_vec::StableVec;

/// Control flow that terminates a block
#[derive(Default, Debug, Clone)]
pub enum ControlFlow {
    /// An if or if-else branch that should be structured if applicable.
    IfElse {
        cond: Variable,
        then: NodeIndex,
        or_else: NodeIndex,
        merge: Option<NodeIndex>,
    },
    /// A switch branch that paths based on `value`
    Switch {
        value: Variable,
        default: NodeIndex,
        branches: Vec<(u32, NodeIndex)>,
        merge: Option<NodeIndex>,
    },
    /// A loop with a header (the block that contains this variant), a `body` and a `continue target`.
    /// `merge` is the block that gets executed as soon as the loop terminates.
    Loop {
        body: NodeIndex,
        continue_target: NodeIndex,
        merge: NodeIndex,
    },
    /// A loop with a header (the block that contains this variant), a `body` and a `continue target`.
    /// `merge` is the block that gets executed as soon as the loop terminates. The header contains
    /// the break condition.
    LoopBreak {
        break_cond: Variable,
        body: NodeIndex,
        continue_target: NodeIndex,
        merge: NodeIndex,
    },
    /// A return statement. This should only occur once in the program and all other returns should
    /// instead branch to this single return block.
    Return,
    /// No special control flow. The block must have exactly one edge that should be followed.
    #[default]
    None,
}

impl Optimizer {
    pub(crate) fn parse_control_flow(&mut self, branch: Branch) {
        match branch {
            Branch::If(if_) => self.parse_if(*if_),
            Branch::IfElse(if_else) => self.parse_if_else(if_else),
            Branch::Switch(switch) => self.parse_switch(*switch),
            Branch::RangeLoop(range_loop) => {
                self.parse_for_loop(*range_loop);
            }
            Branch::Loop(loop_) => self.parse_loop(*loop_),
            Branch::Return => {
                let current_block = self.current_block.take().unwrap();
                let ret = self.ret();
                self.program.add_edge(current_block, ret, 0);
            }
            Branch::Break => {
                let current_block = self.current_block.take().unwrap();
                let loop_break = self.loop_break.back().expect("Can't break outside loop");
                self.program.add_edge(current_block, *loop_break, 0);
            }
        }
    }

    pub(crate) fn parse_if(&mut self, if_: If) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());
        let mut merge = next;

        self.program.add_edge(current_block, then, 0);
        self.program.add_edge(current_block, next, 0);

        self.current_block = Some(then);
        let is_break = self.parse_scope(if_.scope);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, 0);
        } else {
            // Returned
            merge = self.ret;
        }

        let merge = if is_break { None } else { Some(merge) };

        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::IfElse {
            cond: if_.cond,
            then,
            or_else: next,
            merge,
        };
        if let Some(merge) = merge {
            self.program[merge].block_use.push(BlockUse::Merge);
        }
        self.current_block = Some(next);
    }

    pub(crate) fn parse_if_else(&mut self, if_else: Box<IfElse>) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let or_else = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());
        let mut merge = next;

        self.program.add_edge(current_block, then, 0);
        self.program.add_edge(current_block, or_else, 0);

        self.current_block = Some(then);
        let is_break = self.parse_scope(if_else.scope_if);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, 0);
        } else {
            // Returned
            merge = self.ret;
        }

        self.current_block = Some(or_else);
        let is_break = self.parse_scope(if_else.scope_else) || is_break;

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, 0);
        } else {
            // Returned
            merge = self.ret;
        }

        let merge = if is_break { None } else { Some(merge) };
        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::IfElse {
            cond: if_else.cond,
            then,
            or_else,
            merge,
        };
        if let Some(merge) = merge {
            self.program[merge].block_use.push(BlockUse::Merge);
        }
        self.current_block = Some(next);
    }

    pub(crate) fn parse_switch(&mut self, switch: Switch) {
        let current_block = self.current_block.unwrap();
        let next = self.program.add_node(BasicBlock::default());

        let branches = switch
            .cases
            .into_iter()
            .map(|(val, case)| {
                let case_id = self.program.add_node(BasicBlock::default());
                self.program.add_edge(current_block, case_id, 0);
                self.current_block = Some(case_id);
                let is_break = self.parse_scope(case);
                let is_ret = if let Some(current_block) = self.current_block {
                    self.program.add_edge(current_block, next, 0);
                    false
                } else {
                    !is_break
                };
                let val = match val.as_const().expect("Switch value must be constant") {
                    ConstantValue::Int(val) => unsafe { transmute::<i32, u32>(val as i32) },
                    ConstantValue::UInt(val) => val as u32,
                    _ => unreachable!("Switch cases must be integer"),
                };
                (val, case_id, is_break, is_ret)
            })
            .collect::<Vec<_>>();

        let is_break_branch = branches.iter().any(|it| it.2);
        let mut is_ret = branches.iter().any(|it| it.3);
        let branches = branches
            .into_iter()
            .map(|it| (it.0, it.1))
            .collect::<Vec<_>>();

        let default = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, default, 0);
        self.current_block = Some(default);
        let is_break_def = self.parse_scope(switch.scope_default);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, 0);
        } else {
            is_ret = !is_break_def;
        }

        let merge = if is_break_def || is_break_branch {
            None
        } else if is_ret {
            Some(self.ret)
        } else {
            self.program[next].block_use.push(BlockUse::Merge);
            Some(next)
        };

        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::Switch {
            value: switch.value,
            default,
            branches,
            merge,
        };

        self.current_block = Some(next);
    }

    fn parse_loop(&mut self, loop_: Loop) {
        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, header, 0);

        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program.add_edge(header, body, 0);

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(loop_.scope);
        let continue_target = self.program.add_node(BasicBlock::default());
        self.program[continue_target]
            .block_use
            .push(BlockUse::ContinueTarget);

        self.loop_break.pop_back();

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, continue_target, 0);
        }

        self.program.add_edge(continue_target, header, 0);

        *self.program[header].control_flow.borrow_mut() = ControlFlow::Loop {
            body,
            continue_target,
            merge: next,
        };
        self.program[next].block_use.push(BlockUse::Merge);
        self.current_block = Some(next);
    }

    fn parse_for_loop(&mut self, range_loop: RangeLoop) {
        let step = range_loop.step.unwrap_or(1.into());

        let i_id = match range_loop.i.kind {
            VariableKind::LocalMut { id, .. } => id,
            _ => unreachable!(),
        };
        let i = range_loop.i;
        self.program.variables.insert(i_id, i.ty);

        let assign = Instruction::new(Operation::Copy(range_loop.start), i);
        self.current_block_mut().ops.borrow_mut().push(assign);

        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, header, 0);

        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program.add_edge(header, body, 0);
        self.program.add_edge(header, next, 0);

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(range_loop.scope);

        self.loop_break.pop_back();

        let current_block = self.current_block.expect("For loop has no loopback path");

        let continue_target = if self.program[current_block]
            .block_use
            .contains(&BlockUse::Merge)
        {
            let target = self.program.add_node(BasicBlock::default());
            self.program.add_edge(current_block, target, 0);
            target
        } else {
            current_block
        };

        self.program.add_edge(continue_target, header, 0);

        self.program[continue_target]
            .block_use
            .push(BlockUse::ContinueTarget);
        self.program[next].block_use.push(BlockUse::Merge);
        self.current_block = Some(next);

        // For loop constructs
        self.insert_phi(header, i_id, range_loop.start.ty);
        {
            let op = match range_loop.inclusive {
                true => Comparison::LowerEqual,
                false => Comparison::Lower,
            };
            let tmp = *self.allocator.create_local(Type::scalar(ElemType::Bool));
            self.program[header].ops.borrow_mut().push(Instruction::new(
                op(BinaryOperator {
                    lhs: i,
                    rhs: range_loop.end,
                }),
                tmp,
            ));

            *self.program[header].control_flow.borrow_mut() = ControlFlow::LoopBreak {
                break_cond: tmp,
                body,
                continue_target,
                merge: next,
            };
        }
        self.program[current_block]
            .ops
            .borrow_mut()
            .push(Instruction::new(
                Arithmetic::Add(BinaryOperator { lhs: i, rhs: step }),
                i,
            ));
    }

    pub(crate) fn split_critical_edges(&mut self) {
        for block in self.node_ids() {
            let successors = self.program.edges(block);
            let successors = successors.map(|edge| (edge.id(), edge.target()));
            let successors: Vec<_> = successors.collect();

            if successors.len() > 1 {
                let crit = successors
                    .iter()
                    .filter(|(_, b)| self.predecessors(*b).len() > 1)
                    .collect::<Vec<_>>();
                for (edge, successor) in crit {
                    self.program.remove_edge(*edge);
                    let new_block = self.program.add_node(BasicBlock::default());
                    self.program.add_edge(block, new_block, 0);
                    self.program.add_edge(new_block, *successor, 0);
                    self.invalidate_structure();
                    update_phi(self, *successor, block, new_block);
                    update_control_flow(self, block, *successor, new_block);
                }
            }
        }
    }

    /// Split blocks at a `free` call because we only track liveness at a block level
    /// It's easier than doing liveness per-instruction, and free calls are rare anyways
    pub(crate) fn split_free(&mut self) {
        let mut splits = 0;
        while self.split_free_inner() {
            splits += 1;
        }
        if splits > 0 {
            self.invalidate_structure();
        }
    }

    fn split_free_inner(&mut self) -> bool {
        let is_free =
            |inst: &Instruction| matches!(inst.operation, Operation::Marker(Marker::Free(_)));

        for block in self.node_ids() {
            let ops = self.block(block).ops.clone();
            let len = ops.borrow().num_elements();
            let idx = ops.borrow().values().position(is_free);
            if let Some(idx) = idx {
                // Separate free into its own block. They can be merged again later.
                if idx > 0 {
                    self.split_block_after(block, idx - 1);
                    return true;
                }
                if idx < len - 1 {
                    self.split_block_after(block, idx);
                    return true;
                }
            }
        }

        false
    }

    /// Split block after `idx` and return the new block
    fn split_block_after(&mut self, block: NodeIndex, idx: usize) -> NodeIndex {
        let successors = self.successors(block);
        let edges: Vec<EdgeIndex> = self
            .program
            .edges_directed(block, Direction::Outgoing)
            .map(|it| it.id())
            .collect();
        for edge in edges {
            self.program.remove_edge(edge);
        }

        let ops = self.block(block).ops.take();
        let before: Vec<_> = ops.values().take(idx + 1).cloned().collect();
        let after: Vec<_> = ops.values().skip(idx + 1).cloned().collect();
        *self.block(block).ops.borrow_mut() = StableVec::from_iter(before);

        let new_block = BasicBlock::default();
        new_block.control_flow.swap(&self.block(block).control_flow);
        new_block.ops.borrow_mut().extend(after);
        let new_block = self.program.graph.add_node(new_block);

        self.program.add_edge(block, new_block, 0);
        for successor in successors {
            self.program.add_edge(new_block, successor, 0);
        }
        new_block
    }
}

fn update_control_flow(opt: &mut Optimizer, block: NodeIndex, from: NodeIndex, to: NodeIndex) {
    let update = |id: &mut NodeIndex| {
        if *id == from {
            *id = to
        }
    };

    match &mut *opt.program[block].control_flow.borrow_mut() {
        ControlFlow::IfElse { then, or_else, .. } => {
            update(then);
            update(or_else);
        }
        ControlFlow::Switch {
            default, branches, ..
        } => {
            update(default);

            for branch in branches {
                update(&mut branch.1);
            }
        }
        _ => {}
    }
}

fn update_phi(opt: &mut Optimizer, block: NodeIndex, from: NodeIndex, to: NodeIndex) {
    for phi in opt.program[block].phi_nodes.borrow_mut().iter_mut() {
        for entry in phi.entries.iter_mut() {
            if entry.block == from {
                entry.block = to;
            }
        }
    }
}
