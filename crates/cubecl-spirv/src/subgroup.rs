use cubecl_core::ir::Subcube;
use rspirv::spirv::{Capability, GroupOperation, Scope, Word};

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_subcube(&mut self, subcube: Subcube) {
        self.capabilities
            .insert(Capability::GroupNonUniformArithmetic);
        let subgroup = self.subgroup();
        match subcube {
            Subcube::Elect(op) => {
                let out = self.compile_variable(op.out);
                let out_id = self.write_id(&out);
                let bool = self.type_bool();
                self.group_non_uniform_elect(bool, Some(out_id), subgroup)
                    .unwrap();
                self.write(&out, out_id);
            }
            Subcube::All(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    b.group_non_uniform_all(ty, Some(out), subgroup, input)
                        .unwrap();
                });
            }
            Subcube::Any(op) => {
                self.compile_unary_op(op, |b, _, ty, input, out| {
                    b.group_non_uniform_any(ty, Some(out), subgroup, input)
                        .unwrap();
                });
            }
            Subcube::Broadcast(op) => {
                self.compile_binary_op_no_cast(op, |b, _, ty, lhs, rhs, out| {
                    b.group_non_uniform_broadcast(ty, Some(out), subgroup, lhs, rhs)
                        .unwrap();
                });
            }
            Subcube::Sum(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        crate::item::Elem::Int(_, false) => b.group_non_uniform_i_add(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Float(_) => b.group_non_uniform_f_add(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Subcube::Prod(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        crate::item::Elem::Int(_, _) => b.group_non_uniform_i_mul(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Float(_) => b.group_non_uniform_f_mul(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Subcube::Min(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        crate::item::Elem::Int(_, false) => b.group_non_uniform_u_min(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Int(_, true) => b.group_non_uniform_s_min(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Float(_) => b.group_non_uniform_f_min(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Subcube::Max(op) => {
                self.compile_unary_op(op, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        crate::item::Elem::Int(_, false) => b.group_non_uniform_u_max(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Int(_, true) => b.group_non_uniform_s_max(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        crate::item::Elem::Float(_) => b.group_non_uniform_f_max(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
        }
    }

    fn subgroup(&mut self) -> Word {
        self.const_u32(Scope::Subgroup as u32)
    }
}
