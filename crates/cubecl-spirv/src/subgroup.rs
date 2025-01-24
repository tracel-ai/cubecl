use cubecl_core::ir::{Plane, Variable};
use rspirv::spirv::{Capability, GroupOperation, Scope, Word};

use crate::{item::Elem, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_plane(&mut self, plane: Plane, out: Option<Variable>, uniform: bool) {
        self.capabilities
            .insert(Capability::GroupNonUniformArithmetic);
        let subgroup = self.subgroup();
        let out = out.unwrap();
        match plane {
            Plane::Elect => {
                let out = self.compile_variable(out);
                let out_id = self.write_id(&out);
                let bool = self.type_bool();
                self.group_non_uniform_elect(bool, Some(out_id), subgroup)
                    .unwrap();
                self.write(&out, out_id);
            }
            Plane::All(op) => {
                self.capabilities.insert(Capability::GroupNonUniformVote);
                match out.vectorization_factor() {
                    1 => {
                        self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                            b.group_non_uniform_all(ty, Some(out), subgroup, input)
                                .unwrap();
                        });
                    }
                    vec => {
                        let elem_ty = self.compile_item(op.input.item).elem().id(self);
                        let bool_ty = self.type_bool();

                        self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                            let ids = (0..vec)
                                .map(|i| {
                                    let elem_i = b
                                        .composite_extract(elem_ty, None, input, vec![i as u32])
                                        .unwrap();
                                    b.group_non_uniform_all(bool_ty, None, subgroup, elem_i)
                                        .unwrap()
                                })
                                .collect::<Vec<_>>();
                            b.composite_construct(ty, Some(out), ids).unwrap();
                        });
                    }
                };
            }
            Plane::Any(op) => {
                self.capabilities.insert(Capability::GroupNonUniformVote);
                match out.vectorization_factor() {
                    1 => {
                        self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                            b.group_non_uniform_any(ty, Some(out), subgroup, input)
                                .unwrap();
                        });
                    }
                    vec => {
                        let elem_ty = self.compile_item(op.input.item).elem().id(self);
                        let bool_ty = self.type_bool();

                        self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                            let ids = (0..vec)
                                .map(|i| {
                                    let elem_i = b
                                        .composite_extract(elem_ty, None, input, vec![i as u32])
                                        .unwrap();
                                    b.group_non_uniform_any(bool_ty, None, subgroup, elem_i)
                                        .unwrap()
                                })
                                .collect::<Vec<_>>();
                            b.composite_construct(ty, Some(out), ids).unwrap();
                        });
                    }
                };
            }
            Plane::Broadcast(op) => {
                let is_broadcast = self.uniformity.is_var_uniform(op.rhs);
                self.compile_binary_op_no_cast(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    match is_broadcast {
                        true => {
                            b.capabilities.insert(Capability::GroupNonUniformBallot);
                            b.group_non_uniform_broadcast(ty, Some(out), subgroup, lhs, rhs)
                                .unwrap();
                        }
                        false => {
                            b.capabilities.insert(Capability::GroupNonUniformShuffle);
                            b.group_non_uniform_shuffle(ty, Some(out), subgroup, lhs, rhs)
                                .unwrap();
                        }
                    }
                });
            }
            Plane::Sum(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.group_non_uniform_i_add(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Float(_) | Elem::Relaxed => b.group_non_uniform_f_add(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        elem => unreachable!("{elem}"),
                    }
                    .unwrap();
                });
            }
            Plane::Prod(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, _) => b.group_non_uniform_i_mul(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Float(_) | Elem::Relaxed => b.group_non_uniform_f_mul(
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
            Plane::Min(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.group_non_uniform_u_min(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Int(_, true) => b.group_non_uniform_s_min(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Float(_) | Elem::Relaxed => b.group_non_uniform_f_min(
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
            Plane::Max(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    match out_ty.elem() {
                        Elem::Int(_, false) => b.group_non_uniform_u_max(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Int(_, true) => b.group_non_uniform_s_max(
                            ty,
                            Some(out),
                            subgroup,
                            GroupOperation::Reduce,
                            input,
                            None,
                        ),
                        Elem::Float(_) | Elem::Relaxed => b.group_non_uniform_f_max(
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
