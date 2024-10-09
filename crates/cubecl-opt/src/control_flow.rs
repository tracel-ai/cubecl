use std::mem::transmute;

use crate::{BasicBlock, BlockUse, NodeIndex, Optimizer};
use cubecl_core::ir::{
    BinaryOperator, Branch, ConstantScalarValue, Elem, If, IfElse, Item, Loop, Operator, RangeLoop,
    Switch, UnaryOperator, Variable,
};

/// Control flow that terminates a block
#[derive(Default, Debug, Clone)]
pub enum ControlFlow {
    /// A break branch, which does not have an `OpSelectionMerge` in SPIR-V since it's part of the
    /// loop construct.
    Break {
        cond: Variable,
        body: NodeIndex,
        or_break: NodeIndex,
    },
    /// An if or if-else branch that should be structured if applicable.
    IfElse {
        cond: Variable,
        then: NodeIndex,
        or_else: NodeIndex,
        merge: NodeIndex,
    },
    /// A switch branch that paths based on `value`
    Switch {
        value: Variable,
        default: NodeIndex,
        branches: Vec<(u32, NodeIndex)>,
        merge: NodeIndex,
    },
    /// A loop with a header (the block that contains this variant), a `body` and a `continue target`.
    /// `merge` is the block that gets executed as soon as the loop terminates.
    Loop {
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
            Branch::If(if_) => self.parse_if(if_),
            Branch::IfElse(if_else) => self.parse_if_else(if_else),
            Branch::Select(mut select) => {
                self.find_writes_select(&mut select);
                self.current_block_mut()
                    .ops
                    .borrow_mut()
                    .push(Branch::Select(select).into());
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
                let current_block = self.current_block.unwrap();
                let loop_break = self.loop_break.back().expect("Can't break outside loop");
                self.program.add_edge(current_block, *loop_break, ());
            }
        }
    }

    pub(crate) fn parse_if(&mut self, if_: If) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());
        let mut merge = next;

        self.program.add_edge(current_block, then, ());
        self.program.add_edge(current_block, next, ());

        self.current_block = Some(then);
        self.parse_scope(if_.scope);
        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, ());
        } else {
            // Returned
            merge = self.ret;
        }

        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::IfElse {
            cond: if_.cond,
            then,
            or_else: next,
            merge,
        };
        self.program[merge].block_use.push(BlockUse::Merge);
        self.current_block = Some(next);
    }

    pub(crate) fn parse_if_else(&mut self, if_else: IfElse) {
        let current_block = self.current_block.unwrap();
        let then = self.program.add_node(BasicBlock::default());
        let or_else = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());
        let mut merge = next;

        self.program.add_edge(current_block, then, ());
        self.program.add_edge(current_block, or_else, ());

        self.current_block = Some(then);
        self.parse_scope(if_else.scope_if);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, ());
        } else {
            // Returned
            merge = self.ret;
        }

        self.current_block = Some(or_else);
        self.parse_scope(if_else.scope_else);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, ());
        } else {
            // Returned
            merge = self.ret;
        }

        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::IfElse {
            cond: if_else.cond,
            then,
            or_else,
            merge,
        };
        self.program[merge].block_use.push(BlockUse::Merge);

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
                self.program.add_edge(current_block, case_id, ());
                self.current_block = Some(case_id);
                self.parse_scope(case);
                if let Some(current_block) = self.current_block {
                    self.program.add_edge(current_block, next, ());
                }
                let val = match val.as_const().expect("Switch value must be constant") {
                    ConstantScalarValue::Int(val, _) => unsafe {
                        transmute::<i32, u32>(val as i32)
                    },
                    ConstantScalarValue::UInt(val) => val as u32,
                    _ => unreachable!("Switch cases must be integer"),
                };
                (val, case_id)
            })
            .collect::<Vec<_>>();

        let default = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, default, ());
        self.current_block = Some(default);
        self.parse_scope(switch.scope_default);

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, next, ());
        }

        *self.program[current_block].control_flow.borrow_mut() = ControlFlow::Switch {
            value: switch.value,
            default,
            branches,
            merge: next,
        };
        self.program[next].block_use.push(BlockUse::Merge);

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
        self.program[continue_target]
            .block_use
            .push(BlockUse::ContinueTarget);

        self.loop_break.pop_back();

        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, continue_target, ());
        }

        self.program.add_edge(continue_target, header, ());

        *self.program[header].control_flow.borrow_mut() = ControlFlow::Loop {
            body,
            continue_target,
            merge: next,
        };
        self.program[next].block_use.push(BlockUse::Merge);
        self.current_block = Some(next);
    }

    fn parse_for_loop(&mut self, range_loop: RangeLoop) {
        let step = range_loop
            .step
            .unwrap_or(Variable::ConstantScalar(ConstantScalarValue::UInt(1)));

        let i_id = match range_loop.i {
            Variable::Local { id, depth, .. } => (id, depth),
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
        self.current_block_mut().ops.borrow_mut().push(assign);

        let current_block = self.current_block.unwrap();
        let header = self.program.add_node(BasicBlock::default());
        self.program.add_edge(current_block, header, ());

        let break_cond = self.program.add_node(BasicBlock::default());
        let body = self.program.add_node(BasicBlock::default());
        let next = self.program.add_node(BasicBlock::default());

        self.program.add_edge(header, break_cond, ());

        self.program.add_edge(break_cond, body, ());
        self.program.add_edge(break_cond, next, ());

        self.loop_break.push_back(next);

        self.current_block = Some(body);
        self.parse_scope(range_loop.scope);

        self.loop_break.pop_back();

        let current_block = self.current_block.expect("For loop has no loopback path");

        self.program.add_edge(current_block, header, ());

        *self.program[header].control_flow.borrow_mut() = ControlFlow::Loop {
            body: break_cond,
            continue_target: current_block,
            merge: next,
        };
        self.program[current_block]
            .block_use
            .push(BlockUse::ContinueTarget);
        self.program[next].block_use.push(BlockUse::Merge);
        self.current_block = Some(next);

        // For loop constructs
        self.program
            .insert_phi(header, i_id, range_loop.start.item());
        {
            let op = match range_loop.inclusive {
                true => Operator::LowerEqual,
                false => Operator::Lower,
            };
            let mut tmp = self.root_scope.create_local_binding(Item::new(Elem::Bool));
            // Try to fix collisions
            if let Variable::LocalBinding { id, .. } = &mut tmp {
                *id += 20000
            }
            self.program[break_cond].ops.borrow_mut().push(
                op(BinaryOperator {
                    lhs: i,
                    rhs: range_loop.end,
                    out: tmp,
                })
                .into(),
            );

            *self.program[break_cond].control_flow.borrow_mut() = ControlFlow::Break {
                cond: tmp,
                body,
                or_break: next,
            };
        }
        self.program[current_block].ops.borrow_mut().push(
            Operator::Add(BinaryOperator {
                lhs: i,
                rhs: step,
                out: i,
            })
            .into(),
        );
    }
}
