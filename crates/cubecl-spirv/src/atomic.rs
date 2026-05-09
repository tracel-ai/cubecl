use cubecl_core::ir::{AtomicOp, ElemType, InstructionModes, IntKind, UIntKind, Variable};
use rspirv::spirv::{Capability, MemorySemantics, Scope, Word};

use crate::{SpirvCompiler, SpirvTarget, item::Elem};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_atomic(
        &mut self,
        atomic: AtomicOp,
        out: Option<Variable>,
        modes: InstructionModes,
    ) {
        if let Some(out) = out
            && matches!(
                out.elem_type(),
                ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64)
            )
        {
            self.capabilities.insert(Capability::Int64Atomics);
        }

        match atomic {
            AtomicOp::Load(ptr) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(ptr);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let input_id = ptr.id(self);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_r(&ptr);

                self.atomic_load(ty, Some(out_id), input_id, memory, semantics)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Store(op) => {
                if matches!(
                    op.value.elem_type(),
                    ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64)
                ) {
                    self.capabilities.insert(Capability::Int64Atomics);
                }

                let value = self.compile_variable(op.value);
                let ptr = self.compile_variable(op.ptr);

                let value_id = self.read(&value);
                let ptr_id = ptr.id(self);

                let memory = self.scope(&ptr);
                let semantics = self.semantics_w(&ptr);

                self.atomic_store(ptr_id, memory, semantics, value_id)
                    .unwrap();
            }
            AtomicOp::Swap(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                self.atomic_exchange(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::CompareAndSwap(op) => {
                let out = out.unwrap();

                let atomic = self.compile_variable(op.ptr);
                let cmp = self.compile_variable(op.cmp);
                let val = self.compile_variable(op.val);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let atomic_id = atomic.id(self);
                let cmp_id = self.read(&cmp);
                let val_id = self.read(&val);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&atomic);
                let semantics_success = self.semantics_rw(&atomic);
                let semantics_failure = self.semantics_r(&atomic);

                assert!(
                    matches!(out_ty.elem(), Elem::Int(_, _)),
                    "compare and swap doesn't support float atomics"
                );
                self.atomic_compare_exchange(
                    ty,
                    Some(out_id),
                    atomic_id,
                    memory,
                    semantics_success,
                    semantics_failure,
                    val_id,
                    cmp_id,
                )
                .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Add(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                match out_ty.elem() {
                    Elem::Int(_, _) => self
                        .atomic_i_add(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Float(width, None) => {
                        match width {
                            16 if out_ty.vectorization() == 1 => {
                                self.capabilities.insert(Capability::AtomicFloat16AddEXT)
                            }
                            16 => self.capabilities.insert(Capability::AtomicFloat16VectorNV),
                            32 => self.capabilities.insert(Capability::AtomicFloat32AddEXT),
                            64 => self.capabilities.insert(Capability::AtomicFloat64AddEXT),
                            _ => unreachable!(),
                        };
                        self.atomic_f_add_ext(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                            .unwrap()
                    }
                    _ => unreachable!(),
                };

                self.write(&out, out_id);
            }
            AtomicOp::Sub(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                assert!(
                    matches!(out_ty.elem(), Elem::Int(_, _)),
                    "sub doesn't support float atomics"
                );
                match out_ty.elem() {
                    Elem::Int(_, _) => self
                        .atomic_i_sub(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Float(width, None) => {
                        match width {
                            16 if out_ty.vectorization() == 1 => {
                                self.capabilities.insert(Capability::AtomicFloat16AddEXT)
                            }
                            16 => self.capabilities.insert(Capability::AtomicFloat16VectorNV),
                            32 => self.capabilities.insert(Capability::AtomicFloat32AddEXT),
                            64 => self.capabilities.insert(Capability::AtomicFloat64AddEXT),
                            _ => unreachable!(),
                        };
                        let negated = self.f_negate(ty, None, value_id).unwrap();
                        self.declare_math_mode(modes, negated);
                        self.atomic_f_add_ext(ty, Some(out_id), ptr_id, memory, semantics, negated)
                            .unwrap()
                    }
                    _ => unreachable!(),
                };
                self.atomic_i_sub(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Max(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_max(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_max(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Float(width, None) => {
                        match width {
                            16 if out_ty.vectorization() == 1 => {
                                self.capabilities.insert(Capability::AtomicFloat16MinMaxEXT)
                            }
                            16 => self.capabilities.insert(Capability::AtomicFloat16VectorNV),
                            32 => self.capabilities.insert(Capability::AtomicFloat32MinMaxEXT),
                            64 => self.capabilities.insert(Capability::AtomicFloat64MinMaxEXT),
                            _ => unreachable!(),
                        };
                        self.atomic_f_max_ext(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                            .unwrap()
                    }
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            AtomicOp::Min(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_min(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_min(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                        .unwrap(),
                    Elem::Float(width, None) => {
                        match width {
                            16 if out_ty.vectorization() == 1 => {
                                self.capabilities.insert(Capability::AtomicFloat16MinMaxEXT)
                            }
                            16 => self.capabilities.insert(Capability::AtomicFloat16VectorNV),
                            32 => self.capabilities.insert(Capability::AtomicFloat32MinMaxEXT),
                            64 => self.capabilities.insert(Capability::AtomicFloat64MinMaxEXT),
                            _ => unreachable!(),
                        };
                        self.atomic_f_min_ext(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                            .unwrap()
                    }
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            AtomicOp::And(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                assert!(
                    matches!(out_ty.elem(), Elem::Int(_, _)),
                    "and doesn't support float atomics"
                );
                self.atomic_and(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Or(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                assert!(
                    matches!(out_ty.elem(), Elem::Int(_, _)),
                    "or doesn't support float atomics"
                );
                self.atomic_or(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Xor(op) => {
                let out = out.unwrap();

                let ptr = self.compile_variable(op.ptr);
                let value = self.compile_variable(op.value);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let ptr_id = ptr.id(self);
                let value_id = self.read(&value);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.scope(&ptr);
                let semantics = self.semantics_rw(&ptr);

                assert!(
                    matches!(out_ty.elem(), Elem::Int(_, _)),
                    "xor doesn't support float atomics"
                );
                self.atomic_xor(ty, Some(out_id), ptr_id, memory, semantics, value_id)
                    .unwrap();
                self.write(&out, out_id);
            }
        }
    }

    fn scope(&mut self, var: &crate::variable::Variable) -> Word {
        let value = var.scope() as u32;
        self.const_u32(value)
    }

    fn semantics_r(&mut self, var: &crate::variable::Variable) -> Word {
        let value = self.semantics_of(var) | MemorySemantics::ACQUIRE;
        self.const_u32(value.bits())
    }

    fn semantics_w(&mut self, var: &crate::variable::Variable) -> Word {
        let value = self.semantics_of(var) | MemorySemantics::RELEASE;
        self.const_u32(value.bits())
    }

    fn semantics_rw(&mut self, var: &crate::variable::Variable) -> Word {
        let value = self.semantics_of(var) | MemorySemantics::ACQUIRE_RELEASE;
        self.const_u32(value.bits())
    }

    fn semantics_of(&mut self, var: &crate::variable::Variable) -> MemorySemantics {
        match var.scope() {
            Scope::Device => MemorySemantics::UNIFORM_MEMORY,
            Scope::Workgroup => MemorySemantics::WORKGROUP_MEMORY,
            Scope::Subgroup => MemorySemantics::SUBGROUP_MEMORY,
            other => unreachable!("Invalid scope for atomic operation, {other:?}"),
        }
    }
}
