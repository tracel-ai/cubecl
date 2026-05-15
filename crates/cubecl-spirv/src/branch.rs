use cubecl_core::ir as core;
use cubecl_opt::{ControlFlow, NodeIndex};
use rspirv::{
    dr::Operand,
    spirv::{LoopControl, SelectionControl},
};

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_control_flow(&mut self, control_flow: ControlFlow) {
        match control_flow {
            ControlFlow::IfElse {
                cond,
                then,
                or_else,
                merge,
            } => self.compile_if_else(cond, then, or_else, merge),
            ControlFlow::Switch {
                value,
                default,
                branches,
                merge,
            } => self.compile_switch(value, default, branches, merge),
            ControlFlow::Loop {
                body,
                continue_target,
                merge,
            } => self.compile_loop(body, continue_target, merge),
            ControlFlow::LoopBreak {
                break_cond,
                body,
                continue_target,
                merge,
            } => self.compile_loop_break(break_cond, body, continue_target, merge),
            ControlFlow::Return { value } => {
                if let Some(value) = value {
                    let value = self.compile_variable(value);
                    let value_id = self.read(&value);
                    self.ret_value(value_id).unwrap();
                } else {
                    self.ret().unwrap();
                }
                self.current_block = None;
            }
            ControlFlow::Unreachable => {
                self.unreachable().unwrap();
                self.current_block = None;
            }
            ControlFlow::None => {
                let opt = self.opt.clone();
                let func = self
                    .current_func
                    .map(|id| &opt.global_state.extra_functions[&id])
                    .unwrap_or(&opt.main);
                let children = func.successors(self.current_block.unwrap());
                assert_eq!(
                    children.len(),
                    1,
                    "None control flow should have only 1 outgoing edge"
                );
                let label = self.label(children[0]);
                self.branch(label).unwrap();
            }
        }
    }

    fn compile_if_else(
        &mut self,
        cond: core::Variable,
        then: NodeIndex,
        or_else: NodeIndex,
        merge: Option<NodeIndex>,
    ) {
        let cond = self.compile_variable(cond);
        let then_label = self.label(then);
        let else_label = self.label(or_else);
        let cond_id = self.read(&cond);

        if let Some(merge) = merge {
            let merge_label = self.label(merge);
            self.selection_merge(merge_label, SelectionControl::NONE)
                .unwrap();
        }
        self.branch_conditional(cond_id, then_label, else_label, None)
            .unwrap();
    }

    fn compile_switch(
        &mut self,
        value: core::Variable,
        default: NodeIndex,
        branches: Vec<(u32, NodeIndex)>,
        merge: Option<NodeIndex>,
    ) {
        let value = self.compile_variable(value);
        let value_id = self.read(&value);

        let default_label = self.label(default);
        let targets = branches
            .iter()
            .map(|(value, block)| {
                let label = self.label(*block);
                (Operand::LiteralBit32(*value), label)
            })
            .collect::<Vec<_>>();

        if let Some(merge) = merge {
            let merge_label = self.label(merge);
            self.selection_merge(merge_label, SelectionControl::NONE)
                .unwrap();
        }

        self.switch(value_id, default_label, targets).unwrap();
    }

    fn compile_loop(&mut self, body: NodeIndex, continue_target: NodeIndex, merge: NodeIndex) {
        let body_label = self.label(body);
        let continue_label = self.label(continue_target);
        let merge_label = self.label(merge);

        self.loop_merge(merge_label, continue_label, LoopControl::NONE, vec![])
            .unwrap();
        self.branch(body_label).unwrap();
    }

    fn compile_loop_break(
        &mut self,
        break_cond: core::Variable,
        body: NodeIndex,
        continue_target: NodeIndex,
        merge: NodeIndex,
    ) {
        let break_cond = self.compile_variable(break_cond);
        let cond_id = self.read(&break_cond);
        let body_label = self.label(body);
        let continue_label = self.label(continue_target);
        let merge_label = self.label(merge);

        self.loop_merge(merge_label, continue_label, LoopControl::NONE, [])
            .unwrap();
        self.branch_conditional(cond_id, body_label, merge_label, [])
            .unwrap();
    }
}
