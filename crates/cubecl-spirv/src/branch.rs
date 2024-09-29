use cubecl_core::ir as core;
use cubecl_core::ir::{Branch, If, Scope};
use rspirv::spirv::{SelectionControl, Word};

use crate::{item::Item, variable::Variable, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_branch(&mut self, branch: Branch) {
        match branch {
            Branch::If(If { cond, scope }) => self.compile_if(cond, scope),
            b => todo!("{b:?}"),
        }
    }

    pub fn compile_read_bound(
        &mut self,
        arr: &Variable,
        index: Word,
        item: Item,
        read: impl FnOnce(&mut Self) -> Word,
    ) -> Word {
        let ty = item.id(self);
        let len = self.length(arr, None);
        let bool = self.type_bool();
        let cond = self.u_less_than(bool, None, index, len).unwrap();

        let in_bounds = self.id();
        let fallback = self.id();
        let next = self.id();

        self.selection_merge(next, SelectionControl::DONT_FLATTEN)
            .unwrap();
        self.branch_conditional(cond, in_bounds, fallback, vec![1, 0])
            .unwrap();

        self.begin_block(Some(in_bounds)).unwrap();
        let value = read(self);
        self.branch(next).unwrap();

        self.begin_block(Some(fallback)).unwrap();
        let fallback_value = item.constant(self, 0);
        self.branch(next).unwrap();

        self.begin_block(Some(next)).unwrap();
        self.phi(
            ty,
            None,
            vec![(value, in_bounds), (fallback_value, fallback)],
        )
        .unwrap()
    }

    pub fn compile_write_bound(
        &mut self,
        arr: &Variable,
        index: Word,
        write: impl FnOnce(&mut Self),
    ) {
        let len = self.length(arr, None);
        let bool = self.type_bool();
        let cond = self.u_less_than(bool, None, index, len).unwrap();

        let in_bounds = self.id();
        let next = self.id();

        self.selection_merge(next, SelectionControl::DONT_FLATTEN)
            .unwrap();
        self.branch_conditional(cond, in_bounds, next, vec![1, 0])
            .unwrap();

        self.begin_block(Some(in_bounds)).unwrap();
        write(self);
        self.branch(next).unwrap();

        self.begin_block(Some(next)).unwrap();
    }

    fn compile_if(&mut self, cond: core::Variable, scope: Scope) {
        let cond = self.compile_variable(cond);
        let cond_id = self.read(&cond);

        let then = self.id();
        let next = self.id();

        self.selection_merge(next, SelectionControl::NONE).unwrap();
        self.branch_conditional(cond_id, then, next, vec![])
            .unwrap();

        self.compile_scope(scope, Some(then));
        self.branch(next).unwrap();
        self.begin_block(Some(next)).unwrap();
    }
}
