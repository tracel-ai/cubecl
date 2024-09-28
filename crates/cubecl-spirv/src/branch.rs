use cubecl_core::ir as core;
use cubecl_core::ir::{Branch, If, Scope};
use rspirv::spirv::SelectionControl;

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_branch(&mut self, branch: Branch) {
        match branch {
            Branch::If(If { cond, scope }) => self.compile_if(cond, scope),
            b => todo!("{b:?}"),
        }
    }

    fn compile_if(&mut self, cond: core::Variable, scope: Scope) {
        let cond = self.compile_variable(cond);
        let cond_id = self.read(&cond);
        let current_block = self.selected_block();
        self.select_block(None).unwrap(); // pop block
        let label = self.compile_scope(scope);
        let label_next = self.id();
        self.branch(label_next).unwrap();
        self.select_block(current_block).unwrap();
        self.selection_merge(label_next, SelectionControl::NONE)
            .unwrap();
        self.branch_conditional(cond_id, label, label_next, vec![])
            .unwrap();
        self.begin_block(Some(label_next)).unwrap();
    }
}
