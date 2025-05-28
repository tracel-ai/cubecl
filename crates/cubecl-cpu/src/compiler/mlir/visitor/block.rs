use cubecl_opt::{NodeIndex, Optimizer};
use melior::ir::BlockRef;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_basic_block(
        &mut self,
        block: BlockRef<'a, 'a>,
        block_id: NodeIndex,
        opt: &Optimizer,
    ) {
        let ops = opt.block(block_id).ops.borrow();
        self.block_stack.push(block);

        for (_, instruction) in ops.iter() {
            self.visit_instruction(instruction);
        }
    }
}
