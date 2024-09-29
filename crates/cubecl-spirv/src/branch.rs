use cubecl_core::ir::{self as core, Loop, RangeLoop};
use cubecl_core::ir::{Branch, If, Scope};
use rspirv::spirv::{LoopControl, SelectionControl, StorageClass, Word};

use crate::{item::Item, variable::Variable, SpirvCompiler, SpirvTarget};

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
                let start = self.compile_variable(start);
                let end = self.compile_variable(end);
                let step = step.map(|it| self.compile_variable(it));
                let i = match i {
                    core::Variable::Local { id, item, depth } => {
                        let item = Item::Pointer(
                            StorageClass::Function,
                            Box::new(self.compile_item(item)),
                        );
                        let ty = item.id(self);
                        let var = self.declare_function_variable(ty);
                        self.state.variables.insert((id, depth), var);
                        let start_id = self.read(&start);
                        self.store(var, start_id, None, vec![]).unwrap();
                        var
                    }
                    _ => unreachable!(),
                };

                self.compile_range_loop(i, start, end, step, inclusive, scope);
            }
            b => todo!("{b:?}"),
        }
    }

    pub fn compile_range_loop(
        &mut self,
        i: Word,
        start: Variable,
        end: Variable,
        step: Option<Variable>,
        inclusive: bool,
        scope: Scope,
    ) {
        let i_ty = start.item().id(self);
        let bool = self.type_bool();

        let header = self.id();
        let break_cond = self.id();
        let body = self.id();
        let continue_target = self.id();
        let post = self.id();

        self.branch(header).unwrap();
        self.begin_block(Some(header)).unwrap();

        self.loop_merge(post, continue_target, LoopControl::NONE, vec![])
            .unwrap();
        self.branch(break_cond).unwrap();

        let end_id = self.read(&end);

        self.begin_block(Some(break_cond)).unwrap();
        let i_value = self.load(i_ty, None, i, None, vec![]).unwrap();
        let cond = match inclusive {
            true => self.s_less_than_equal(bool, None, i_value, end_id).unwrap(),
            false => self.s_less_than(bool, None, i_value, end_id).unwrap(),
        };
        self.branch_conditional(cond, body, post, vec![]).unwrap();

        self.compile_scope(scope, Some(body));
        self.branch(continue_target).unwrap();

        let step_id = step
            .map(|it| self.read(&it))
            .unwrap_or_else(|| self.const_u32(1));

        self.begin_block(Some(continue_target)).unwrap();
        let i_value = self.load(i_ty, None, i, None, vec![]).unwrap();
        let inc = self.i_add(i_ty, None, i_value, step_id).unwrap();
        self.store(i, inc, None, vec![]).unwrap();
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
        self.branch(continue_target).unwrap();

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
        self.branch(next).unwrap();
        self.begin_block(Some(next)).unwrap();
    }
}
