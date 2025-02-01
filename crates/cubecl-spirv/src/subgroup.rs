use cubecl_core::ir::{Plane, UnaryOperator, Variable};
use rspirv::spirv::{Capability, GroupOperation, Scope, Word};

use crate::{item::Elem, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_plane(&mut self, plane: Plane, out: Option<Variable>) {
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
                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
                            b.group_non_uniform_all(ty, Some(out), subgroup, input)
                                .unwrap();
                        });
                    }
                    vec => {
                        let elem_ty = self.compile_item(op.input.item).elem().id(self);
                        let bool_ty = self.type_bool();

                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
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
                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
                            b.group_non_uniform_any(ty, Some(out), subgroup, input)
                                .unwrap();
                        });
                    }
                    vec => {
                        let elem_ty = self.compile_item(op.input.item).elem().id(self);
                        let bool_ty = self.type_bool();

                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
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
            Plane::Ballot(op) => {
                self.capabilities.insert(Capability::GroupNonUniformBallot);
                assert_eq!(
                    op.input.vectorization_factor(),
                    1,
                    "plane_ballot can't work with vectorized values"
                );
                self.compile_unary_op(op, out, |b, _, ty, input, out| {
                    b.group_non_uniform_ballot(ty, Some(out), subgroup, input)
                        .unwrap();
                });
            }
            Plane::Broadcast(op) => {
                if op.rhs.as_const().is_some() {
                    self.capabilities.insert(Capability::GroupNonUniformBallot);
                    self.compile_binary_op_no_cast(op, out, |b, _, ty, lhs, rhs, out| {
                        b.group_non_uniform_broadcast(ty, Some(out), subgroup, lhs, rhs)
                            .unwrap();
                    });
                } else {
                    self.capabilities.insert(Capability::GroupNonUniformShuffle);
                    self.compile_binary_op_no_cast(op, out, |b, _, ty, lhs, rhs, out| {
                        b.group_non_uniform_shuffle(ty, Some(out), subgroup, lhs, rhs)
                            .unwrap();
                    });
                }
            }
            Plane::Sum(op) => {
                self.plane_sum(op, out, GroupOperation::Reduce);
            }
            Plane::ExclusiveSum(op) => {
                self.plane_sum(op, out, GroupOperation::ExclusiveScan);
            }
            Plane::InclusiveSum(op) => {
                self.plane_sum(op, out, GroupOperation::InclusiveScan);
            }
            Plane::Prod(op) => {
                self.plane_prod(op, out, GroupOperation::Reduce);
            }
            Plane::ExclusiveProd(op) => {
                self.plane_prod(op, out, GroupOperation::ExclusiveScan);
            }
            Plane::InclusiveProd(op) => {
                self.plane_prod(op, out, GroupOperation::InclusiveScan);
            }
            Plane::Min(op) => {
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
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
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
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

    fn plane_sum(&mut self, op: UnaryOperator, out: Variable, action: GroupOperation) {
        let subgroup = self.subgroup();
        self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
            match out_ty.elem() {
                Elem::Int(_, _) => {
                    b.group_non_uniform_i_add(ty, Some(out), subgroup, action, input, None)
                }
                Elem::Float(_) | Elem::Relaxed => {
                    b.group_non_uniform_f_add(ty, Some(out), subgroup, action, input, None)
                }
                elem => unreachable!("{elem}"),
            }
            .unwrap();
        });
    }

    fn plane_prod(&mut self, op: UnaryOperator, out: Variable, action: GroupOperation) {
        let subgroup = self.subgroup();
        self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
            match out_ty.elem() {
                Elem::Int(_, _) => {
                    b.group_non_uniform_i_mul(ty, Some(out), subgroup, action, input, None)
                }
                Elem::Float(_) | Elem::Relaxed => {
                    b.group_non_uniform_f_mul(ty, Some(out), subgroup, action, input, None)
                }
                _ => unreachable!(),
            }
            .unwrap();
        });
    }

    fn subgroup(&mut self) -> Word {
        self.const_u32(Scope::Subgroup as u32)
    }
}
