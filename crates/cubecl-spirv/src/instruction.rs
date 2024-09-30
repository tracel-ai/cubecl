use cubecl_core::ir::{self as core, BinaryOperator, Metadata, UnaryOperator};
use cubecl_core::ir::{Operation, Operator};
use rspirv::spirv::{Capability, Word};

use crate::{
    containers::Slice,
    item::{Elem, Item},
    variable::Variable,
    SpirvCompiler, SpirvTarget,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, op: Operation) {
        match op {
            Operation::Operator(operator) => self.compile_operator(operator),
            Operation::Branch(branch) => self.compile_branch(branch),
            Operation::Metadata(meta) => self.compile_meta(meta),
            Operation::Subcube(subcube) => self.compile_subcube(subcube),
            op => todo!("{op:?}"),
        }
    }

    pub fn compile_operator(&mut self, op: Operator) {
        match op {
            Operator::Index(op) => {
                let value = self.compile_variable(op.lhs);
                let index = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);

                self.read_indexed(out_id, &value, &index);
                self.write(&out, out_id);
            }
            Operator::IndexAssign(op) => {
                let index = self.compile_variable(op.lhs);
                let value = self.compile_variable(op.rhs);
                let out = self.compile_variable(op.out);
                let value_id = self.read(&value);

                self.write_indexed(&out, &index, value_id);
            }
            Operator::Slice(op) => {
                let item = self.compile_item(op.input.item());
                let input = self.compile_variable(op.input);
                let start = self.compile_variable(op.start);
                let end = self.compile_variable(op.end);
                let out = match op.out {
                    core::Variable::Slice { id, depth, .. } => (id, depth),
                    _ => unreachable!(),
                };

                let start_id = self.read(&start);
                let (len, const_len) = match (start.as_const(), end.as_const()) {
                    (Some(start), Some(end)) => {
                        let len = (end - start) as u32;
                        let len_id = self.const_u32(len);
                        (len_id, Some(len))
                    }
                    _ => {
                        let end_id = self.read(&end);
                        let len_ty = Elem::Int(32, false).id(self);
                        (self.i_sub(len_ty, None, end_id, start_id).unwrap(), None)
                    }
                };

                self.state.slices.insert(
                    out,
                    Slice {
                        ptr: input,
                        offset: start_id,
                        len,
                        const_len,
                        item,
                    },
                );
            }
            Operator::Assign(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(op.out);
                let input_id = self.read_as(&input, &out.item());
                let out_id = self.write_id(&out);
                let out_ty = out.item().id(self);
                self.copy_object(out_ty, Some(out_id), input_id).unwrap();
                self.write(&out, out_id);
            }
            Operator::Equal(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_equal(ty, Some(out), lhs, rhs),
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::NotEqual(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_not_equal(ty, Some(out), lhs, rhs),
                        Elem::Void => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Lower(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::LowerEqual(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_less_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_less_than_equal(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Greater(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::GreaterEqual(op) => {
                self.compile_binary_op_bool(op, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Int(_, false) => b.u_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, true) => b.s_greater_than_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(_) => b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Operator::Add(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_add(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Sub(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_add(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_add(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Mul(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Div(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_div(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_div(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Remainder(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_rem(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_rem(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Modulo(op) => {
                self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.u_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Int(_, true) => b.s_mod(ty, Some(out), lhs, rhs).unwrap(),
                        Elem::Float(_) => b.f_mod(ty, Some(out), lhs, rhs).unwrap(),
                        _ => unreachable!(),
                    };
                });
            }
            Operator::Normalize(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    T::normalize(b, ty, input, out);
                });
            }
            Operator::Magnitude(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    T::magnitude(b, ty, input, out);
                });
            }
            Operator::Dot(op) => {
                if op.lhs.item().vectorization.map(|it| it.get()).unwrap_or(1) == 1 {
                    self.compile_binary_op(op, |b, out_ty, ty, lhs, rhs, out| {
                        match out_ty.elem() {
                            Elem::Int(_, _) => b.i_mul(ty, Some(out), lhs, rhs).unwrap(),
                            Elem::Float(_) => b.f_mul(ty, Some(out), lhs, rhs).unwrap(),
                            _ => unreachable!(),
                        };
                    });
                } else {
                    let lhs = self.compile_variable(op.lhs);
                    let rhs = self.compile_variable(op.rhs);
                    let out = self.compile_variable(op.out);
                    let ty = out.item().id(self);

                    let lhs_id = self.read(&lhs);
                    let rhs_id = self.read(&rhs);
                    let out_id = self.write_id(&out);

                    if matches!(lhs.elem(), Elem::Int(_, _)) {
                        self.capabilities.insert(Capability::DotProduct);
                    }

                    match (lhs.elem(), rhs.elem()) {
                        (Elem::Int(_, false), Elem::Int(_, false)) => {
                            self.u_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Int(_, true), Elem::Int(_, false)) => {
                            self.su_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Int(_, false), Elem::Int(_, true)) => {
                            self.su_dot(ty, Some(out_id), rhs_id, lhs_id, None)
                        }
                        (Elem::Int(_, true), Elem::Int(_, true)) => {
                            self.s_dot(ty, Some(out_id), lhs_id, rhs_id, None)
                        }
                        (Elem::Float(_), Elem::Float(_)) => {
                            self.dot(ty, Some(out_id), lhs_id, rhs_id)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                    self.write(&out, out_id);
                }
            }
            op => todo!("{op:?}"),
        }
    }

    pub fn compile_unary_op(
        &mut self,
        op: UnaryOperator,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word),
    ) {
        let input = self.compile_variable(op.input);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
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
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(op.out);
        let lhs_ty = lhs.item();

        let lhs_id = self.read(&lhs);
        let rhs_id = self.read_as(&rhs, &lhs_ty);
        let out_id = self.write_id(&out);

        let ty = out.item().id(self);

        exec(self, lhs_ty, ty, lhs_id, rhs_id, out_id);
        self.write(&out, out_id);
    }

    fn compile_meta(&mut self, meta: Metadata) {
        match meta {
            Metadata::Length { var, out } => {
                let var = self.compile_variable(var);
                let out = self.compile_variable(out);
                self.length(&var, Some(&out));
            }
            meta => todo!("{meta:?}"),
        }
    }

    pub fn length(&mut self, var: &Variable, out: Option<&Variable>) -> Word {
        let (out_id, out_ty) = if let Some(out) = out {
            let out_id = self.write_id(out);
            let out_ty = out.elem().id(self);
            (Some(out_id), out_ty)
        } else {
            (None, self.type_int(32, 0))
        };

        let id = match var {
            Variable::GlobalInputArray(ptr, _) | Variable::GlobalOutputArray(ptr, _) => {
                self.array_length(out_ty, out_id, *ptr, 0).unwrap()
            }
            Variable::Slice { len, .. } => {
                if out.is_some() {
                    self.copy_object(out_ty, out_id, *len).unwrap()
                } else {
                    *len
                }
            }
            Variable::SharedMemory(_, _, len)
            | Variable::ConstantArray(_, _, len)
            | Variable::LocalArray(_, _, len) => self.const_u32(*len),
            var => unimplemented!("Var {var:?} doesn't have length"),
        };
        if let Some(out) = out {
            self.write(out, id);
        }
        id
    }
}
