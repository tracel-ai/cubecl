use std::ops::Deref;

use cubecl_opt::{ControlFlow, NodeIndex};
use tracel_llvm::melior::{
    dialect::cf,
    ir::{Block, BlockLike, BlockRef, RegionLike},
};

use super::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_basic_block(&mut self, block_id: NodeIndex, opt: &Optimizer) -> BlockRef<'a, 'a> {
        if let Some(block) = self.blocks.get(&block_id) {
            return *block;
        }

        let basic_block = opt.block(block_id);

        let arguments: Vec<_> = basic_block
            .phi_nodes
            .borrow()
            .iter()
            .map(|phi_node| {
                let argument_type = phi_node.out.item.to_type(self.context);
                for entry in phi_node.entries.iter() {
                    self.blocks_args
                        .entry((entry.block, block_id))
                        .or_default()
                        .push(entry.value);
                }
                (argument_type, self.location)
            })
            .collect();

        let block = Block::new(&arguments);
        for (i, phi_node) in basic_block.phi_nodes.borrow().iter().enumerate() {
            self.insert_variable(phi_node.out, block.argument(i).unwrap().into());
        }
        let this_block = self
            .current_region
            .insert_block_before(self.last_block, block);
        self.block = this_block;

        self.blocks.insert(block_id, this_block);
        for (_, instruction) in basic_block.ops.borrow().iter() {
            self.visit_instruction(instruction);
        }

        match basic_block.control_flow.borrow().deref() {
            ControlFlow::IfElse {
                cond,
                then,
                or_else,
                merge,
            } => {
                let condition = self.get_variable(*cond);
                if let Some(merge) = merge {
                    self.visit_basic_block(*merge, opt);
                }
                let true_successor = self.visit_basic_block(*then, opt);
                let false_successor = self.visit_basic_block(*or_else, opt);
                let true_successor_operands = self.get_block_args(block_id, *then);
                let false_successor_operands = self.get_block_args(block_id, *or_else);
                this_block.append_operation(cf::cond_br(
                    self.context,
                    condition,
                    true_successor.deref(),
                    false_successor.deref(),
                    &true_successor_operands,
                    &false_successor_operands,
                    self.location,
                ));
            }
            ControlFlow::Switch {
                value,
                default,
                branches,
                merge,
            } => {
                let case_values: Vec<_> = branches.iter().map(|(n, _)| *n as i64).collect();
                let operand = self.get_variable(*value);
                let operand_type = value.item.to_type(self.context);
                if let Some(merge) = merge {
                    self.visit_basic_block(*merge, opt);
                }
                let default_block = self.visit_basic_block(*default, opt);
                let attributes: Vec<Value<'a, 'a>> = self.get_block_args(block_id, *default);
                let default_destination = (default_block.deref(), attributes.as_slice());
                let blocks: Vec<_> = branches
                    .iter()
                    .map(|(_, block_id)| self.visit_basic_block(*block_id, opt))
                    .collect();
                let attributes_vec: Vec<Vec<Value<'a, 'a>>> = branches
                    .iter()
                    .map(|(_, dest)| self.get_block_args(block_id, *dest))
                    .collect();
                let case_destinations: Vec<_> = (0..branches.len())
                    .map(|i| (blocks[i].deref(), attributes_vec[i].as_slice()))
                    .collect();
                this_block.append_operation(
                    cf::switch(
                        self.context,
                        &case_values,
                        operand,
                        operand_type,
                        default_destination,
                        &case_destinations,
                        self.location,
                    )
                    .unwrap(),
                );
            }
            ControlFlow::Loop {
                body,
                continue_target,
                merge,
            } => {
                let body_block = self.visit_basic_block(*body, opt);
                let destination_operands = self.get_block_args(block_id, *body);
                self.visit_basic_block(*continue_target, opt);
                self.visit_basic_block(*merge, opt);
                this_block.append_operation(cf::br(
                    body_block.deref(),
                    &destination_operands,
                    self.location,
                ));
            }
            ControlFlow::LoopBreak {
                break_cond,
                body,
                continue_target,
                merge,
            } => {
                let condition = self.get_variable(*break_cond);
                let body_block = self.visit_basic_block(*body, opt);
                self.visit_basic_block(*continue_target, opt);
                let next_block = self.visit_basic_block(*merge, opt);
                let body_argument = self.get_block_args(block_id, *body);
                let next_argument = self.get_block_args(block_id, *continue_target);
                this_block.append_operation(cf::cond_br(
                    self.context,
                    condition,
                    body_block.deref(),
                    next_block.deref(),
                    &body_argument,
                    &next_argument,
                    self.location,
                ));
            }
            ControlFlow::Return => {
                this_block.append_operation(cf::br(&self.last_block, &[], self.location));
            }
            ControlFlow::None => {
                let destination = opt.successors(block_id)[0];
                let successor = self.visit_basic_block(destination, opt);
                let block_arg = self.get_block_args(block_id, destination);
                this_block.append_operation(cf::br(successor.deref(), &block_arg, self.location));
            }
        };
        this_block
    }
}
