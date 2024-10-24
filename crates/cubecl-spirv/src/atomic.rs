use cubecl_core::ir::{AtomicOp, Variable};
use rspirv::spirv::{MemorySemantics, Scope};

use crate::{item::Elem, SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_atomic(&mut self, atomic: AtomicOp, out: Option<Variable>) {
        let out = out.unwrap();
        match atomic {
            AtomicOp::Load(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let input_id = input.id(self);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_load(ty, Some(out_id), input_id, memory, semantics)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Store(op) => {
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);

                let input_id = self.read(&input);
                let out_id = out.id(self);

                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_store(out_id, memory, semantics, input_id)
                    .unwrap();
            }
            AtomicOp::Swap(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_exchange(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::CompareAndSwap(op) => {
                let atomic = self.compile_variable(op.input);
                let cmp = self.compile_variable(op.cmp);
                let val = self.compile_variable(op.val);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let atomic_id = atomic.id(self);
                let cmp_id = self.read(&cmp);
                let val_id = self.read(&val);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics_success = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());
                let semantics_failure = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

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
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_i_add(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Sub(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_i_sub(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Max(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_max(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_max(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            AtomicOp::Min(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                match out_ty.elem() {
                    Elem::Int(_, false) => self
                        .atomic_u_min(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    Elem::Int(_, true) => self
                        .atomic_s_min(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                        .unwrap(),
                    _ => unreachable!(),
                };
                self.write(&out, out_id);
            }
            AtomicOp::And(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_and(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Or(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_or(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
            AtomicOp::Xor(op) => {
                let lhs = self.compile_variable(op.lhs);
                let rhs = self.compile_variable(op.rhs);
                let out = self.compile_variable(out);
                let out_ty = out.item();

                let lhs_id = lhs.id(self);
                let rhs_id = self.read(&rhs);
                let out_id = self.write_id(&out);

                let ty = out_ty.id(self);
                let memory = self.const_u32(Scope::Device as u32);
                let semantics = self.const_u32(MemorySemantics::UNIFORM_MEMORY.bits());

                self.atomic_xor(ty, Some(out_id), lhs_id, memory, semantics, rhs_id)
                    .unwrap();
                self.write(&out, out_id);
            }
        }
    }
}
