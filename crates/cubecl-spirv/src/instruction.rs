use cubecl_core::ir::{
    self as core, BinaryOperator, Comparison, Instruction, Operation, Operator, UnaryOperator,
};
use rspirv::spirv::{Capability, Decoration, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::IndexedVariable,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, inst: Instruction) {
        // Setting source loc for non-semantic ops is pointless, they don't show up in a profiler/debugger.
        if !matches!(inst.operation, Operation::NonSemantic(_)) {
            self.set_source_loc(&inst.source_loc);
        }
        let uniform = matches!(inst.out, Some(out) if self.uniformity.is_var_uniform(out));
        match inst.operation {
            Operation::Copy(var) => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(inst.out());
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);

                self.copy_object(ty, Some(out_id), in_id).unwrap();
                self.mark_uniformity(out_id, uniform);
                self.write(&out, out_id);
            }
            Operation::Arithmetic(operator) => self.compile_arithmetic(operator, inst.out, uniform),
            Operation::Comparison(operator) => self.compile_cmp(operator, inst.out, uniform),
            Operation::Bitwise(operator) => self.compile_bitwise(operator, inst.out, uniform),
            Operation::Operator(operator) => self.compile_operator(operator, inst.out, uniform),
            Operation::Atomic(atomic) => self.compile_atomic(atomic, inst.out),
            Operation::Branch(_) => unreachable!("Branches shouldn't exist in optimized IR"),
            Operation::Metadata(meta) => self.compile_meta(meta, inst.out, uniform),
            Operation::Plane(plane) => self.compile_plane(plane, inst.out, uniform),
            Operation::Synchronization(sync) => self.compile_sync(sync),
            Operation::CoopMma(cmma) => self.compile_cmma(cmma, inst.out),
            Operation::NonSemantic(debug) => self.compile_debug(debug),
            Operation::Barrier(_) => panic!("Barrier not supported in SPIR-V"),
            Operation::Tma(_) => panic!("TMA not supported in SPIR-V"),
        }
    }

    pub fn compile_cmp(&mut self, op: Comparison, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match op {
            Comparison::Equal(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_equal(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::NotEqual(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_not_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::Lower(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_less_than(ty, Some(out), lhs, rhs)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::LowerEqual(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_less_than_equal(ty, Some(out), lhs, rhs)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::Greater(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_greater_than(ty, Some(out), lhs, rhs)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::GreaterEqual(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
        }
    }

    pub fn compile_operator(&mut self, op: Operator, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match op {
            Operator::Index(op) | Operator::UncheckedIndex(op) => {
                let is_atomic = op.list.item.elem.is_atomic();
                let value = self.compile_variable(op.list);
                let index = self.compile_variable(op.index);
                let out = self.compile_variable(out);

                if is_atomic {
                    let ptr = match self.index(&value, &index, true) {
                        IndexedVariable::Pointer(ptr, _) => ptr,
                        _ => unreachable!("Atomic is always pointer"),
                    };
                    let out_id = out.as_binding().unwrap();

                    // This isn't great but atomics can't currently be constructed so should be fine
                    self.merge_binding(out_id, ptr);
                } else {
                    let out_id = self.read_indexed(&out, &value, &index);
                    self.mark_uniformity(out_id, uniform);
                    self.write(&out, out_id);
                }
            }
            Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op) => {
                let index = self.compile_variable(op.index);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed(&out, &index, value_id);
            }
            Operator::Cast(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);

                if let Some(as_const) = input.as_const() {
                    let cast = self.static_cast(as_const, &input.elem(), &out.item());
                    self.copy_object(ty, Some(out_id), cast).unwrap();
                } else {
                    input.item().cast_to(self, Some(out_id), in_id, &out.item());
                }

                self.write(&out, out_id);
            }
            Operator::And(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.logical_and(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Or(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.logical_or(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Not(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, _, ty, input, out| {
                    b.logical_not(ty, Some(out), input).unwrap();
                });
            }
            Operator::Reinterpret(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.bitcast(ty, Some(out), input).unwrap();
                })
            }
            Operator::InitLine(op) => {
                let values = op
                    .inputs
                    .into_iter()
                    .map(|input| self.compile_variable(input))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|it| self.read(&it))
                    .collect::<Vec<_>>();
                let item = self.compile_item(out.item);
                let out = self.compile_variable(out);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);
                let ty = item.id(self);
                self.composite_construct(ty, Some(out_id), values).unwrap();
                self.write(&out, out_id);
            }
            Operator::CopyMemory(op) => {
                let input = self.compile_variable(op.input);
                let in_index = self.compile_variable(op.in_index);
                let out = self.compile_variable(out);
                let out_index = self.compile_variable(op.out_index);

                let in_ptr = self.index_ptr(&input, &in_index);
                let out_ptr = self.index_ptr(&out, &out_index);
                self.copy_memory(out_ptr, in_ptr, None, None, vec![])
                    .unwrap();
            }
            Operator::CopyMemoryBulk(op) => {
                self.capabilities.insert(Capability::Addresses);
                let input = self.compile_variable(op.input);
                let in_index = self.compile_variable(op.in_index);
                let out = self.compile_variable(out);
                let out_index = self.compile_variable(op.out_index);
                let len = op.len.as_const().unwrap().as_u32();

                let source = self.index_ptr(&input, &in_index);
                let target = self.index_ptr(&out, &out_index);
                let size = self.const_u32(len * out.item().size());
                self.copy_memory_sized(target, source, size, None, None, vec![])
                    .unwrap();
            }
            Operator::Select(op) => self.compile_select(op.cond, op.then, op.or_else, out, uniform),
        }
    }

    pub fn compile_unary_op_cast(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read_as(&input, &out_ty);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op_bool(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let in_ty = input.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out.item().id(self);

        exec(self, in_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read_as(&lhs, &out_ty);
        let rhs_id = self.read_as(&rhs, &out_ty);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_no_cast(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read(&rhs);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_bool(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let lhs_ty = lhs.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read_as(&rhs, &lhs_ty);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out.item().id(self);

        exec(self, lhs_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_select(
        &mut self,
        cond: core::Variable,
        then: core::Variable,
        or_else: core::Variable,
        out: core::Variable,
        uniform: bool,
    ) {
        let cond = self.compile_variable(cond);
        let then = self.compile_variable(then);
        let or_else = self.compile_variable(or_else);
        let out = self.compile_variable(out);

        let out_ty = out.item();
        let ty = out_ty.id(self);

        let cond_id = self.read(&cond);
        let then = self.read_as(&then, &out_ty);
        let or_else = self.read_as(&or_else, &out_ty);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        self.select(ty, Some(out_id), cond_id, then, or_else)
            .unwrap();
        self.write(&out, out_id);
    }

    pub fn mark_uniformity(&mut self, id: Word, uniform: bool) {
        if uniform {
            self.decorate(id, Decoration::Uniform, []);
        }
    }
}
