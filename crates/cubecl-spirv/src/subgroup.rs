use cubecl_core::ir::{Plane, Variable};
use cubecl_opt::Uniformity;
use rspirv::spirv::{Capability, GroupOperation, Scope, Word};

use crate::{item::Elem, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_plane(&mut self, plane: Plane, out: Option<Variable>) {
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
                let uniform = self.block_uniform();
                let execute = |b: &mut Self, ty, out, input| match uniform {
                    true => {
                        b.capabilities.insert(Capability::Groups);
                        b.group_all(ty, out, subgroup, input).unwrap()
                    }
                    false => {
                        b.capabilities.insert(Capability::GroupNonUniformVote);
                        b.group_non_uniform_all(ty, out, subgroup, input).unwrap()
                    }
                };
                match out.vectorization_factor() {
                    1 => {
                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
                            execute(b, ty, Some(out), input);
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
                                    execute(b, bool_ty, None, elem_i)
                                })
                                .collect::<Vec<_>>();
                            b.composite_construct(ty, Some(out), ids).unwrap();
                        });
                    }
                };
            }
            Plane::Any(op) => {
                let uniform = self.block_uniform();
                let execute = |b: &mut Self, ty, out, input| match uniform {
                    true => {
                        b.capabilities.insert(Capability::Groups);
                        b.group_any(ty, out, subgroup, input).unwrap()
                    }
                    false => {
                        b.capabilities.insert(Capability::GroupNonUniformVote);
                        b.group_non_uniform_any(ty, out, subgroup, input).unwrap()
                    }
                };
                match out.vectorization_factor() {
                    1 => {
                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
                            execute(b, ty, Some(out), input);
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
                                    execute(b, bool_ty, None, elem_i)
                                })
                                .collect::<Vec<_>>();
                            b.composite_construct(ty, Some(out), ids).unwrap();
                        });
                    }
                };
            }
            Plane::Broadcast(op) => {
                let uniform = self.block_uniform();
                let broadcast = self.var_uniform(op.rhs);

                if uniform && broadcast {
                    self.capabilities.insert(Capability::Groups);
                }

                self.compile_binary_op_no_cast(op, out, |b, _, ty, lhs, rhs, out| {
                    match (uniform, broadcast) {
                        (true, true) => {
                            b.capabilities.insert(Capability::Groups);
                            b.group_broadcast(ty, Some(out), subgroup, lhs, rhs)
                                .unwrap();
                        }
                        (false, true) => {
                            b.capabilities.insert(Capability::GroupNonUniformBallot);
                            b.group_non_uniform_broadcast(ty, Some(out), subgroup, lhs, rhs)
                                .unwrap();
                        }
                        (_, false) => {
                            b.capabilities.insert(Capability::GroupNonUniformShuffle);
                            b.group_non_uniform_shuffle(ty, Some(out), subgroup, lhs, rhs)
                                .unwrap();
                        }
                    }
                });
            }
            Plane::Sum(op) => {
                let uniform = self.block_uniform();
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
                    match (out_ty.elem(), uniform) {
                        (Elem::Int(_, _), true) => {
                            b.capabilities.insert(Capability::Groups);
                            b.group_i_add(ty, Some(out), subgroup, GroupOperation::Reduce, input)
                                .unwrap();
                        }
                        (Elem::Int(_, _), false) => {
                            b.capabilities.insert(Capability::GroupNonUniformArithmetic);
                            b.group_non_uniform_i_add(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                                None,
                            )
                            .unwrap();
                        }
                        (Elem::Float(_), true) | (Elem::Relaxed, true) => {
                            b.capabilities.insert(Capability::Groups);
                            b.group_f_add(ty, Some(out), subgroup, GroupOperation::Reduce, input)
                                .unwrap();
                        }
                        (Elem::Float(_), false) | (Elem::Relaxed, false) => {
                            b.capabilities.insert(Capability::GroupNonUniformArithmetic);
                            b.group_non_uniform_f_add(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                                None,
                            )
                            .unwrap();
                        }
                        (elem, _) => unreachable!("{elem}"),
                    };
                });
            }
            Plane::Prod(op) => {
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| {
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
                let uniform = self.block_uniform();
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| match uniform {
                    true => {
                        b.capabilities.insert(Capability::Groups);
                        match out_ty.elem() {
                            Elem::Int(_, false) => b.group_u_min(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            Elem::Int(_, true) => b.group_s_min(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            Elem::Float(_) | Elem::Relaxed => b.group_f_min(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                    }
                    false => {
                        b.capabilities.insert(Capability::GroupNonUniformArithmetic);
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
                    }
                });
            }
            Plane::Max(op) => {
                let uniform = self.block_uniform();
                self.compile_unary_op(op, out, |b, out_ty, ty, input, out| match uniform {
                    true => {
                        b.capabilities.insert(Capability::Groups);
                        match out_ty.elem() {
                            Elem::Int(_, false) => b.group_u_max(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            Elem::Int(_, true) => b.group_s_max(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            Elem::Float(_) | Elem::Relaxed => b.group_f_max(
                                ty,
                                Some(out),
                                subgroup,
                                GroupOperation::Reduce,
                                input,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                    }
                    false => {
                        b.capabilities.insert(Capability::GroupNonUniformArithmetic);
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
                    }
                });
            }
        }
    }

    fn subgroup(&mut self) -> Word {
        self.const_u32(Scope::Subgroup as u32)
    }

    fn block_uniform(&self) -> bool {
        let uniformity = self.opt.borrow_mut().analysis::<Uniformity>();
        uniformity.is_block_uniform(self.current_block.unwrap())
    }

    fn var_uniform(&self, var: Variable) -> bool {
        let uniformity = self.opt.borrow_mut().analysis::<Uniformity>();
        uniformity.is_var_uniform(var)
    }
}
