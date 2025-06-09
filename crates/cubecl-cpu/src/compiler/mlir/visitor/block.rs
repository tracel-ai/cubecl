use std::ops::Deref;

use cubecl_opt::{ControlFlow, NodeIndex, Optimizer};

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
            // ControlFlow::IfElse {
            //     cond,
            //     then,
            //     or_else,
            //     merge,
            // } => {}
            // ControlFlow::Switch {
            //     value,
            //     default,
            //     branches,
            //     merge,
            // } => {}
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
            }
            ControlFlow::None => (),
            _ => todo!("{:?}", basic_block.control_flow),
        };
    }
}
