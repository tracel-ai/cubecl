use cubecl_core::ir::Scope;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_scope(&mut self, scope: &Scope) {
        for instruction in scope.instructions.iter().take(4) {
            self.visit_instruction(instruction);
        }
    }
}
