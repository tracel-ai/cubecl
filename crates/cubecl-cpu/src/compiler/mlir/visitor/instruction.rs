use cubecl_core::ir::Instruction;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_instruction(&mut self, instruction: &Instruction) {
        match instruction.out {
            Some(out) => {
                self.visit_operation_with_out(&instruction.operation, out);
            }
            None => {
                self.visit_operation(&instruction.operation);
            }
        }
    }
}
