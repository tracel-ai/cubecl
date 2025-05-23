use cubecl_core::ir::Scope;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_scope<'b: 'a>(&'a mut self, scope: &'b Scope) {
        for instruction in scope.instructions.iter().take(2) {
            self.visit_instruction(instruction);
        }
    }
}
