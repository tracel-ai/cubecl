use cubecl_core::ir::{self as core, Loop, RangeLoop};
use cubecl_core::ir::{Branch, If, Scope};
use rspirv::spirv::{LoopControl, SelectionControl, Word};

use crate::{
    item::{Elem, Item},
    variable::Variable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_branch(&mut self, branch: Branch) {
        match branch {
            Branch::If(If { cond, scope }) => self.compile_if(cond, scope),
            Branch::Loop(Loop { scope }) => self.compile_loop(scope),
            Branch::RangeLoop(RangeLoop {
                i,
                start,
                end,
                step,
                inclusive,
                scope,
            }) => {
                let i = self.compile_variable(i);
                let start = self.compile_variable(start);
                let end = self.compile_variable(end);
                let step = step.map(|it| self.compile_variable(it));

                self.compile_range_loop(i, start, end, step, inclusive, scope);
            }
            Branch::Return => self.ret().unwrap(),
            b => todo!("{b:?}"),
        }
    }

    pub fn compile_range_loop(
        &mut self,
        i: Variable,
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        scope: Scope,
    ) {
        let i_ty = start.item().id(self);
        let bool = self.type_bool();

        let i_value = self.read(&i);
        let start_id = self.read(&start);
        let end_id = self.read(&end);

        let current_func = self.selected_function().unwrap();
        let current_block = self.selected_block().unwrap();

        let pre = self.module_ref().functions[current_func].blocks[current_block]
            .label_id()
            .unwrap();
        let header = self.id();
        let break_cond = self.id();
        let body = self.id();
        let continue_target = self.id();
        let post = self.id();

        let inc = self.id();

        self.branch(header).unwrap();
        self.begin_block(Some(header)).unwrap();

        self.phi(
            i_ty,
            Some(i_value),
            vec![(start_id, pre), (inc, continue_target)],
        )
        .unwrap();
        self.loop_merge(post, continue_target, LoopControl::NONE, vec![])
            .unwrap();
        self.branch(break_cond).unwrap();

        self.begin_block(Some(break_cond)).unwrap();
        let cond = match (inclusive, i.elem()) {
            (true, Elem::Int(_, false)) => self.u_less_than_equal(bool, None, i_value, end_id),
            (true, Elem::Int(_, true)) => self.s_less_than_equal(bool, None, i_value, end_id),
            (false, Elem::Int(_, false)) => self.u_less_than(bool, None, i_value, end_id),
            (false, Elem::Int(_, true)) => self.s_less_than(bool, None, i_value, end_id),
            _ => panic!("For loop should be integer"),
        }
        .unwrap();
        self.branch_conditional(cond, body, post, vec![]).unwrap();

        self.compile_scope(scope, Some(body));
        if self.selected_block().is_some() {
            self.branch(continue_target).unwrap();
        }

        let step_id = step
            .map(|it| self.read(&it))
            .unwrap_or_else(|| self.const_u32(1));

        self.begin_block(Some(continue_target)).unwrap();
        self.i_add(i_ty, Some(inc), i_value, step_id).unwrap();
        self.branch(header).unwrap();

        self.begin_block(Some(post)).unwrap();
    }

    pub fn compile_loop(&mut self, scope: Scope) {
        let header = self.id();
        let body = self.id();
        let continue_target = self.id();
        let post = self.id();

        self.branch(header).unwrap();
        self.begin_block(Some(header)).unwrap();

        self.loop_merge(post, continue_target, LoopControl::NONE, vec![])
            .unwrap();
        self.branch(body).unwrap();

        self.compile_scope(scope, Some(body));
        if self.selected_block().is_some() {
            self.branch(continue_target).unwrap();
        }

        self.begin_block(Some(continue_target)).unwrap();
        self.branch(header).unwrap();

        self.begin_block(Some(post)).unwrap();
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
        if self.selected_block().is_some() {
            self.branch(next).unwrap();
        }
        self.begin_block(Some(next)).unwrap();
    }
}
