use std::ops::Deref;

use cubecl_opt::{ControlFlow, NodeIndex, Optimizer};
use melior::{
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

        let arguments = vec![];

        // for phi_nodes in basic_block.phi_nodes.borrow().iter() {
        //     let argument_type = phi_nodes.out.item.to_type(self.context);
        //     arguments.push((argument_type, self.location));
        // }

        let block = Block::new(&arguments);
        let this_block = self
            .current_region
            .insert_block_before(self.last_block, block);
        self.current_block = this_block;

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
                let true_successor = self.visit_basic_block(*then, opt);
                let false_successor = self.visit_basic_block(*or_else, opt);
                this_block.append_operation(cf::cond_br(
                    self.context,
                    condition,
                    true_successor.deref(),
                    false_successor.deref(),
                    &[],
                    &[],
                    self.location,
                ));
                if let Some(merge) = merge {
                    self.visit_basic_block(*merge, opt);
                }
            }
            // ControlFlow::Switch {
            //     value,
            //     default,
            //     branches,
            //     merge,
            // } => {

            // }
            //ControlFlow::Loop {
            //    body,
            //    continue_target,
            //    merge,
            //} => todo!(),
            //ControlFlow::LoopBreak {
            //    break_cond,
            //    body,
            //    continue_target,
            //    merge,
            //} => todo!(),
            ControlFlow::Return => {
                this_block.append_operation(cf::br(&self.last_block, &[], self.location));
            }
            ControlFlow::None => {
                let successor = self.visit_basic_block(opt.successors(block_id)[0], opt);
                this_block.append_operation(cf::br(successor.deref(), &[], self.location));
            }
            _ => todo!("{:?}", basic_block.control_flow),
        };
        this_block
    }
}
