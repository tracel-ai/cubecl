use std::ops::Deref;

use cubecl_opt::{ControlFlow, NodeIndex, Optimizer};
use melior::{
    dialect::scf,
    ir::{Block, BlockLike, Region, RegionLike},
};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_basic_block(&mut self, block_id: NodeIndex, opt: &Optimizer) {
        let basic_block = opt.block(block_id);
        let ops = opt.block(block_id).ops.borrow();

        for (_, instruction) in ops.iter() {
            self.visit_instruction(instruction);
        }

        drop(ops);
        match basic_block.control_flow.borrow().deref() {
            ControlFlow::IfElse {
                cond,
                then,
                or_else,
                merge,
            } => {
                let condition = self.get_variable(*cond);
                self.block().append_operation(scf::r#if(
                    condition,
                    &[],
                    {
                        let region = Region::new();
                        let block = Block::new(&[]);
                        region.append_block(block);
                        let block = region.first_block().unwrap();
                        self.block_stack.push(block);
                        self.visit_basic_block(*then, opt);
                        self.block()
                            .append_operation(scf::r#yield(&[], self.location));
                        self.block_stack.pop();
                        region
                    },
                    {
                        if let Some(merge) = merge
                            && merge != or_else
                        {
                            let region = Region::new();
                            let block = Block::new(&[]);
                            region.append_block(block);
                            let block = region.first_block().unwrap();
                            self.block_stack.push(block);
                            self.visit_basic_block(*or_else, opt);
                            self.block_stack.pop();
                            region
                        } else {
                            Region::new()
                        }
                    },
                    self.location,
                ));
                if let Some(merge) = merge {
                    self.visit_basic_block(*merge, opt);
                }
            }
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
                // Implementation needs to jump to the last return because func.return is not really conceived for return inside the function
                ()
            }
            ControlFlow::None => (),
            _ => todo!("{:?}", basic_block.control_flow),
        };
    }
}
