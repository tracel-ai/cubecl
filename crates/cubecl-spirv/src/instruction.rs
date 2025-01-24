use cubecl_common::ExecutionMode;
use cubecl_core::ir::{
    self as core, BinaryOperator, Comparison, Instruction, Operation, Operator, UnaryOperator,
};
use rspirv::spirv::{Capability, Decoration, Word};

use crate::{
    item::{Elem, Item},
    lookups::Slice,
    variable::IndexedVariable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, inst: Instruction) {
        match inst.operation {
            Operation::Copy(var) => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(inst.out());
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);

                self.copy_object(ty, Some(out_id), in_id).unwrap();
                self.write(&out, out_id);
            }
            Operation::Arithmetic(operator) => self.compile_arithmetic(operator, inst.out),
            Operation::Comparison(operator) => self.compile_cmp(operator, inst.out),
            Operation::Bitwise(operator) => self.compile_bitwise(operator, inst.out),
            Operation::Operator(operator) => self.compile_operator(operator, inst.out),
            Operation::Atomic(atomic) => self.compile_atomic(atomic, inst.out),
            Operation::Branch(_) => unreachable!("Branches shouldn't exist in optimized IR"),
            Operation::Metadata(meta) => self.compile_meta(meta, inst.out),
            Operation::Plane(plane) => self.compile_plane(plane, inst.out),
            Operation::Synchronization(sync) => self.compile_sync(sync),
            Operation::CoopMma(cmma) => self.compile_cmma(cmma, inst.out),
            Operation::NonSemantic(debug) => self.compile_debug(debug),
            Operation::Pipeline(_) => panic!("Pipeline not supported in SPIR-V"),
        }
    }

    pub fn compile_cmp(&mut self, op: Comparison, out: Option<core::Variable>) {
        let out = out.unwrap();
        match op {
            Comparison::Equal(op) => {
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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
                self.compile_binary_op_bool(op, out, |b, lhs_ty, ty, lhs, rhs, out| {
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

    pub fn compile_operator(&mut self, op: Operator, out: Option<core::Variable>) {
        let out = out.unwrap();
        match op {
            Operator::Index(op) => {
                let is_atomic = op.lhs.item.elem.is_atomic();
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);

                if is_atomic {
                    let checked = matches!(self.mode, ExecutionMode::Checked) && value.has_len();
                    let ptr = match self.index(&value, &index, !checked) {
                        IndexedVariable::Pointer(ptr, _) => ptr,
                        _ => unreachable!("Atomic is always pointer"),
                    };
                    let out_id = out.as_binding().unwrap();

                    // This isn't great but atomics can't currently be constructed so should be fine
                    self.merge_binding(out_id, ptr);
                } else {
                    let out_id = self.read_indexed(&out, &value, &index);
                    self.write(&out, out_id);
                }
            }
            Operator::IndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed(&out, &index, value_id);
            }
            Operator::UncheckedIndex(op) => {
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);

                let out_id = self.read_indexed_unchecked(&out, &value, &index);
                self.write(&out, out_id);
            }
            Operator::UncheckedIndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let value_id = self.read_as(&value, &out.indexed_item());

                self.write_indexed_unchecked(&out, &index, value_id);
            }
            Operator::Slice(op) => {
                let item = self.compile_item(op.input.item);
                let input = self.compile_variable(op.input);
                let start = self.compile_variable(op.start);
                let end = self.compile_variable(op.end);
                let out = match out.kind {
                    core::VariableKind::Slice { id } => id,
                    _ => unreachable!(),
                };

                let start_id = self.read(&start);
                let end_id = self.read(&end);
                let const_len = match (start.as_const(), end.as_const()) {
                    (Some(start), Some(end)) => {
                        let len = end.as_u32() - start.as_u32();
                        Some(len)
                    }
                    _ => None,
                };

                self.state.slices.insert(
                    out,
                    Slice {
                        ptr: input,
                        offset: start_id,
                        end: end_id,
                        const_len,
                        item,
                    },
                );
            }
            Operator::Cast(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);

                if let Some(as_const) = input.as_const() {
                    let cast = self.static_cast(as_const, &input.elem(), &out.item());
                    self.copy_object(ty, Some(out_id), cast).unwrap();
                } else {
                    input.item().cast_to(self, Some(out_id), in_id, &out.item());
                }

                self.write(&out, out_id);
            }
            Operator::And(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.logical_and(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Or(op) => {
                self.compile_binary_op(op, out, |b, _, ty, lhs, rhs, out| {
                    b.logical_or(ty, Some(out), lhs, rhs).unwrap();
                });
            }
            Operator::Not(op) => {
                self.compile_unary_op_cast(op, out, |b, _, ty, input, out| {
                    b.logical_not(ty, Some(out), input).unwrap();
                });
            }
            Operator::Bitcast(op) => self.compile_unary_op(op, out, |b, _, ty, input, out| {
                b.bitcast(ty, Some(out), input).unwrap();
            }),
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
                let checked =
                    matches!(self.mode, ExecutionMode::Checked) && input.has_len() && out.has_len();
                if checked {
                    let in_index = self.read(&in_index);
                    let out_index = self.read(&out_index);
                    self.compile_copy_bound(&input, &out, in_index, out_index, None, |b| {
                        b.copy_memory(out_ptr, in_ptr, None, None, vec![]).unwrap();
                    });
                } else {
                    self.copy_memory(out_ptr, in_ptr, None, None, vec![])
                        .unwrap();
                }
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
                let checked =
                    matches!(self.mode, ExecutionMode::Checked) && input.has_len() && out.has_len();
                if checked {
                    let in_index = self.read(&in_index);
                    let out_index = self.read(&out_index);
                    self.compile_copy_bound(&input, &out, in_index, out_index, Some(size), |b| {
                        b.copy_memory_sized(target, source, size, None, None, vec![])
                            .unwrap();
                    });
                } else {
                    self.copy_memory_sized(target, source, size, None, None, vec![])
                        .unwrap();
                }
            }
            Operator::Select(op) => self.compile_select(op.cond, op.then, op.or_else, out),
        }
    }

    pub fn compile_unary_op_cast(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read_as(&input, &out_ty);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_unary_op_bool(
        &mut self,
        op: UnaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(out);
        let in_ty = input.item();

        let input_id = self.read(&input);
        let out_id = self.write_id(&out);

        let ty = out.item().id(self);

        exec(self, in_ty, ty, input_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read_as(&lhs, &out_ty);
        let rhs_id = self.read_as(&rhs, &out_ty);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_no_cast(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let out_ty = out.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read(&rhs);
        let out_id = self.write_id(&out);

        let ty = out_ty.id(self);

        exec(self, out_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    pub fn compile_binary_op_bool(
        &mut self,
        op: BinaryOperator,
        out: core::Variable,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);
        let lhs_ty = lhs.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read_as(&rhs, &lhs_ty);
        let out_id = self.write_id(&out);

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

        self.select(ty, Some(out_id), cond_id, then, or_else)
            .unwrap();
        self.write(&out, out_id);
    }
}
