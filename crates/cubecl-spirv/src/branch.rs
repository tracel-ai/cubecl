use cubecl_core::ir::{self as core, IfElse, Loop, RangeLoop, Select, Switch};
use cubecl_core::ir::{Branch, If, Scope};
use rspirv::{
    dr::Operand,
    spirv::{LoopControl, SelectionControl, Word},
};

use crate::{
    item::{Elem, Item},
    lookups,
    variable::Variable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_branch(&mut self, branch: Branch) {
        match branch {
            Branch::If(If { cond, scope }) => self.compile_if(cond, scope),
            Branch::IfElse(IfElse {
                cond,
                scope_if,
                scope_else,
            }) => self.compile_if_else(cond, scope_if, scope_else),
            Branch::Switch(Switch {
                value,
                scope_default,
                cases,
            }) => {
                let value = self.compile_variable(value);
                let cases = cases
                    .into_iter()
                    .map(|(var, case)| (self.compile_variable(var), case))
                    .collect();
                self.compile_switch(value, scope_default, cases)
            }
            Branch::Loop(Loop { scope }) => self.compile_loop(scope),
            Branch::RangeLoop(RangeLoop {
                i,
                start,
                end,
                step,
                inclusive,
                scope,
            }) => {
                let i_key = match i {
                    core::Variable::LocalBinding { id, depth, .. } => (id, depth),
                    _ => unreachable!(),
                };
                let start = self.compile_variable(start);
                let end = self.compile_variable(end);
                let step = step.map(|it| self.compile_variable(it));

                self.compile_range_loop(i_key, start, end, step, inclusive, scope);
            }
            Branch::Return => self.ret().unwrap(),
            Branch::Break => {
                let current = self.state.loops.back();
                let post = current.expect("Can't break when not in loop").post;
                self.branch(post).unwrap();
            }
            Branch::Select(Select {
                cond,
                then,
                or_else,
                out,
            }) => self.compile_select(cond, then, or_else, out),
        }
    }

    pub fn compile_range_loop(
        &mut self,
        i_key: (u16, u8),
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        scope: Scope,
    ) {
        let i_ty = start.item().id(self);
        let bool = self.type_bool();

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

        self.state.loops.push_back(lookups::Loop {
            header,
            continue_target,
            post,
        });

        let inc = self.id();

        self.branch(header).unwrap();
        self.begin_block(Some(header)).unwrap();

        let i_value = self
            .phi(i_ty, None, vec![(start_id, pre), (inc, continue_target)])
            .unwrap();
        self.state.bindings.insert(i_key, i_value);

        self.loop_merge(post, continue_target, LoopControl::NONE, vec![])
            .unwrap();
        self.branch(break_cond).unwrap();

        self.begin_block(Some(break_cond)).unwrap();
        let cond = match (inclusive, start.elem()) {
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

        self.state.loops.pop_back();

        self.begin_block(Some(post)).unwrap();
    }

    pub fn compile_loop(&mut self, scope: Scope) {
        let header = self.id();
        let body = self.id();
        let continue_target = self.id();
        let post = self.id();

        self.state.loops.push_back(lookups::Loop {
            header,
            continue_target,
            post,
        });

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

        self.state.loops.pop_back();

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
        let fallback_value = item.constant(self, 0u32.into());
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

    fn compile_if_else(&mut self, cond: core::Variable, scope_if: Scope, scope_else: Scope) {
        let cond = self.compile_variable(cond);
        let cond_id = self.read(&cond);

        let then = self.id();
        let or_else = self.id();
        let next = self.id();

        self.selection_merge(next, SelectionControl::NONE).unwrap();
        self.branch_conditional(cond_id, then, or_else, vec![])
            .unwrap();

        self.compile_scope(scope_if, Some(then));
        if self.selected_block().is_some() {
            self.branch(next).unwrap();
        }
        self.compile_scope(scope_else, Some(or_else));
        if self.selected_block().is_some() {
            self.branch(next).unwrap();
        }

        self.begin_block(Some(next)).unwrap();
    }

    fn compile_select(
        &mut self,
        cond: core::Variable,
        then: core::Variable,
        or_else: core::Variable,
        out: core::Variable,
    ) {
        let cond = self.compile_variable(cond);
        let then = self.compile_variable(then);
        let or_else = self.compile_variable(or_else);
        let out = self.compile_variable(out);

        let then_ty = then.item();
        let ty = then_ty.id(self);

        let cond_id = self.read(&cond);
        let then = self.read(&then);
        let or_else = self.read_as(&or_else, &then_ty);
        let out_id = self.write_id(&out);

        self.select(ty, Some(out_id), cond_id, then, or_else)
            .unwrap();
        self.write(&out, out_id);
    }

    fn compile_switch(&mut self, var: Variable, default: Scope, branches: Vec<(Variable, Scope)>) {
        let var = self.read(&var);

        let branch_labels = branches
            .into_iter()
            .map(|(val, branch)| {
                let value = val.as_const().expect("Switch case must be const").as_u32();
                (value, self.id(), branch)
            })
            .collect::<Vec<_>>();
        let default_block = self.id();
        let next = self.id();

        let targets = branch_labels
            .iter()
            .map(|(val, id, _)| (Operand::LiteralBit32(*val), *id))
            .collect::<Vec<_>>();

        self.selection_merge(next, SelectionControl::NONE).unwrap();
        self.switch(var, default_block, targets).unwrap();
        for (_, id, branch) in branch_labels {
            self.compile_scope(branch, Some(id));
            if self.selected_block().is_some() {
                self.branch(next).unwrap();
            }
        }
        self.compile_scope(default, Some(default_block));
        if self.selected_block().is_some() {
            self.branch(next).unwrap();
        }

        self.begin_block(Some(next)).unwrap();
    }
}
