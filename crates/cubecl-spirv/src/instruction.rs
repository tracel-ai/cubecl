use cubecl_core::ir::{
    self as core, BinaryOperands, Comparison, Instruction, InstructionModes, Memory, Operation,
    Operator, UnaryOperands,
};
use rspirv::spirv::{Decoration, MemoryAccess, Word};

use crate::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_operation(&mut self, inst: Instruction) {
        // Setting source loc for non-semantic ops is pointless, they don't show up in a profiler/debugger.
        if !matches!(inst.operation, Operation::NonSemantic(_)) {
            self.set_source_loc(&inst.source_loc);
        }
        let uniform = inst
            .out
            .is_some_and(|out| self.uniformity.is_var_uniform(out));
        match inst.operation {
            Operation::Copy(var) => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(inst.out());
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let in_id = input.item().broadcast(self, in_id, None, &out.item());
                let out_id = self.write_id(&out);

                self.copy_object(ty, Some(out_id), in_id).unwrap();
                self.mark_uniformity(out_id, uniform);
                self.write(&out, out_id);
            }
            Operation::Memory(mem) => self.compile_memory(mem, inst.out),
            Operation::Arithmetic(operator) => {
                self.compile_arithmetic(operator, inst.out, inst.modes, uniform)
            }
            Operation::Comparison(operator) => {
                self.compile_cmp(operator, inst.out, inst.modes, uniform)
            }
            Operation::Bitwise(operator) => self.compile_bitwise(operator, inst.out, uniform),
            Operation::Operator(operator) => self.compile_operator(operator, inst.out, uniform),
            Operation::Atomic(atomic) => self.compile_atomic(atomic, inst.out, inst.modes),
            Operation::Branch(_) => unreachable!("Branches shouldn't exist in optimized IR"),
            Operation::Metadata(meta) => self.compile_meta(meta, inst.out, uniform),
            Operation::Plane(plane) => self.compile_plane(plane, inst.out, uniform),
            Operation::Synchronization(sync) => self.compile_sync(sync),
            Operation::CoopMma(cmma) => self.compile_cmma(cmma, inst.out),
            Operation::TensorIndexing(tensor) => self.compile_tensor_indexing(tensor, inst.out),
            Operation::NonSemantic(debug) => self.compile_debug(debug),
            Operation::Barrier(_) => panic!("Barrier not supported in SPIR-V"),
            Operation::Tma(_) => panic!("TMA not supported in SPIR-V"),
            Operation::Marker(_) => {}
            Operation::ConstructAggregate(..) | Operation::ExtractAggregateField(..) => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    pub fn compile_cmp(
        &mut self,
        op: Comparison,
        out: Option<core::Variable>,
        modes: InstructionModes,
        uniform: bool,
    ) {
        let out = out.unwrap();
        match op {
            Comparison::Equal(op) => {
                self.compile_binary_op_bool(op, out, uniform, |b, lhs_ty, ty, lhs, rhs, out| {
                    match lhs_ty.elem() {
                        Elem::Bool => b.logical_equal(ty, Some(out), lhs, rhs),
                        Elem::Int(_, _) => b.i_equal(ty, Some(out), lhs, rhs),
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_not_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_less_than(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_less_than_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_greater_than(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
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
                        Elem::Float(..) => {
                            b.declare_math_mode(modes, out);
                            b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs)
                        }
                        Elem::Relaxed => {
                            b.decorate(out, Decoration::RelaxedPrecision, []);
                            b.declare_math_mode(modes, out);
                            b.f_ord_greater_than_equal(ty, Some(out), lhs, rhs)
                        }
                        _ => unreachable!(),
                    }
                    .unwrap();
                });
            }
            Comparison::IsNan(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.is_nan(ty, Some(out), input).unwrap();
                });
            }
            Comparison::IsInf(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.is_inf(ty, Some(out), input).unwrap();
                });
            }
        }
    }

    pub fn compile_memory(&mut self, mem: Memory, out: Option<core::Variable>) {
        match mem {
            Memory::Index(op) => {
                let list = self.compile_variable(op.list);
                let index = self.compile_variable(op.index);
                let out = self.compile_variable(out.unwrap());

                let ptr = self.index(&list, &index, &out);

                self.write(&out, ptr);
            }
            // In SPIR-V, variables are already pointers. So for reference we just treat the out
            // var as an alias for the input.
            Memory::Reference(variable) => {
                let var = self.compile_variable(variable).id(self);
                let out_id = out.unwrap().index().unwrap();
                self.state.lookups.bindings.insert(out_id, var);
            }
            Memory::Load(variable) => {
                let ptr = self.compile_variable(variable);
                let out = self.compile_variable(out.unwrap());

                let id = self.load_aligned(&ptr, &out);
                self.write(&out, id);
            }
            Memory::Store(op) => {
                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);

                self.store_aligned(&ptr, &value);
            }
            Memory::CopyMemory(op) => {
                let source = self.compile_variable(op.source);
                let target = self.compile_variable(op.target);

                let out_ty = target.item();
                let align = source.item().size().max(target.item().size());

                let source = self.read(&source);
                let target = self.read(&target);

                if op.len == 1 {
                    self.copy_memory(
                        target,
                        source,
                        Some(MemoryAccess::ALIGNED),
                        [align.into()],
                        None,
                        [],
                    )
                    .unwrap();
                } else {
                    let size = op.len as u32 * out_ty.size();
                    let size_id = self.const_u32(size);

                    self.copy_memory_sized(
                        target,
                        source,
                        size_id,
                        Some(MemoryAccess::ALIGNED),
                        [size.into()],
                        None,
                        [],
                    )
                    .unwrap();
                }
            }
        }
    }

    pub fn compile_operator(&mut self, op: Operator, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match op {
            Operator::Cast(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let ty = out.item().id(self);
                let in_id = self.read(&input);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);

                if let Some(as_const) = input.as_const() {
                    let cast = self.static_cast(as_const, &input.elem(), &out.item()).0;
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
            Operator::InitVector(op) => {
                let values = op
                    .inputs
                    .into_iter()
                    .map(|input| self.compile_variable(input))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|it| self.read(&it))
                    .collect::<Vec<_>>();
                let item = self.compile_type(out.ty);
                let out = self.compile_variable(out);
                let out_id = self.write_id(&out);
                self.mark_uniformity(out_id, uniform);
                let ty = item.id(self);
                self.composite_construct(ty, Some(out_id), values).unwrap();
                self.write(&out, out_id);
            }
            Operator::InsertComponent(op) => {
                let vector = self.compile_variable(op.vector);
                let value = self.compile_variable(op.value);
                let output = self.compile_variable(out);

                let vector = self.read(&vector);
                let value = self.read(&value);
                let out_ty = output.item().id(self);
                let write_id = self.write_id(&output);

                if let Some(index) = op.index.as_const() {
                    let index = index.as_u32();
                    self.composite_insert(out_ty, Some(write_id), value, vector, [index])
                        .unwrap();
                } else {
                    let index = self.compile_variable(op.index);
                    let index = self.read(&index);

                    self.vector_insert_dynamic(out_ty, Some(write_id), vector, value, index)
                        .unwrap();
                }

                self.write(&output, write_id);
            }
            Operator::ExtractComponent(op) => {
                let vector = self.compile_variable(op.lhs);
                let output = self.compile_variable(out);

                let vector = self.read(&vector);
                let out_ty = output.item().id(self);
                let write_id = self.write_id(&output);

                if let Some(index) = op.rhs.as_const() {
                    let index = index.as_u32();
                    self.composite_extract(out_ty, Some(write_id), vector, [index])
                        .unwrap();
                } else {
                    let index = self.compile_variable(op.rhs);
                    let index = self.read(&index);

                    self.vector_extract_dynamic(out_ty, Some(write_id), vector, index)
                        .unwrap();
                }

                self.write(&output, write_id);
            }
            Operator::Select(op) => self.compile_select(op.cond, op.then, op.or_else, out, uniform),
        }
    }

    pub fn compile_unary_op_cast(
        &mut self,
        op: UnaryOperands,
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
        op: UnaryOperands,
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
        op: UnaryOperands,
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
        op: BinaryOperands,
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
        op: BinaryOperands,
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
        op: BinaryOperands,
        out: core::Variable,
        uniform: bool,
        exec: impl FnOnce(&mut Self, Item, Word, Word, Word, Word),
    ) {
        let lhs = self.compile_variable(op.lhs);
        let rhs = self.compile_variable(op.rhs);
        let out = self.compile_variable(out);

        let in_ty = out.item().same_vectorization(lhs.elem());

        let lhs_id = self.read_as(&lhs, &in_ty);
        let rhs_id = self.read_as(&rhs, &in_ty);
        let out_id = self.write_id(&out);
        self.mark_uniformity(out_id, uniform);

        let ty = out.item().id(self);

        exec(self, in_ty, ty, lhs_id, rhs_id, out_id);
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
