//! Experimental FLOP profiling.
//!
//! When profiling is enabled on a [`Scope`], an extra global atomic counter buffer is appended to
//! the kernel and an atomic increment is inserted after each profiled arithmetic operation. This
//! gives a *dynamic* operation count that respects control flow and loops, since the increment
//! lives next to the operation in the same basic block.
//!
//! The processor is applied automatically by [`Scope::process`] for any scope whose
//! [`crate::ProfileInfo`] is enabled, so every backend supports it without backend-specific code.
//!
//! This is a first slice: only [`Arithmetic::Add`] is counted, and the counter increments by one
//! per executed instruction (vectorization width is ignored for now).

use alloc::vec::Vec;

use crate::{
    Arithmetic, AtomicBinaryOperands, AtomicOp, ElemType, IndexOperands, Instruction, Memory,
    Operation, Processor, Scope, ScopeProcessing, Type, UIntKind, Value,
};

/// Inserts an atomic increment of `counter` after each profiled arithmetic operation.
#[derive(Debug)]
pub struct FlopCountProcessor {
    /// The global counter buffer (a `&mut [atomic<u32>]`).
    counter: Value,
}

impl FlopCountProcessor {
    /// Create a processor that increments the given global atomic `counter`.
    pub fn new(counter: Value) -> Self {
        Self { counter }
    }

    /// The element type of the FLOP counter buffer: a single atomic `u32`.
    pub fn flop_counter_type() -> Type {
        Type::atomic(Type::scalar(ElemType::UInt(UIntKind::U32)))
    }

    /// Append `counter[0] += 1` to the processing instruction stream.
    fn emit_increment(&self, processing: &mut ScopeProcessing, vector_size: usize) {
        // Build the increment in a throwaway scope that shares the global state (allocator, type
        // maps, ...) so the freshly created values don't collide with existing ones. Profiling is
        // disabled on this scope so its own `process` call doesn't recurse into FLOP counting.
        let scope = Scope::root(false, false).with_global_state(processing.global_state.clone());

        let u32_ty = Type::scalar(ElemType::UInt(UIntKind::U32));

        // `&counter[0]` -> pointer to the atomic element.
        let elem_ptr = scope.create_value(Type::pointer(
            self.counter.value_type(),
            self.counter.address_space(),
        ));
        scope.register(Instruction::new(
            Memory::Index(IndexOperands {
                list: self.counter,
                index: Value::constant(0u32.into(), u32_ty),
                unroll_factor: 1,
                checked: false,
            }),
            elem_ptr,
        ));

        // `atomicAdd(&counter[0], 1)`. The returned old value is unused.
        let old = scope.create_value(u32_ty);
        scope.register(Instruction::new(
            AtomicOp::Add(AtomicBinaryOperands {
                ptr: elem_ptr,
                value: Value::constant(vector_size.into(), u32_ty),
            }),
            old,
        ));

        let tmp = scope.process([]);
        processing.instructions.extend(tmp.instructions);
    }
}

impl Processor for FlopCountProcessor {
    fn transform(&self, mut processing: ScopeProcessing) -> ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            let (is_profiled, vector_size) = match &instruction.operation {
                Operation::Arithmetic(Arithmetic::Add(ops)) => (true, ops.lhs.vector_size()),
                _ => (false, 1),
            };

            processing.instructions.push(instruction);

            if is_profiled {
                self.emit_increment(&mut processing, vector_size);
            }
        }

        processing
    }
}
