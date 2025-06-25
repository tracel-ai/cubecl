use cubecl_core::ir::Scope;

use super::MLIRCompiler;

impl MLIRCompiler {
    pub(super) fn visit(&mut self, scope: &Scope) {
        println!("{:?}", scope.locals);
    }
}
