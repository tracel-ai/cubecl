use crate::{Branch, CountInfo, Operation, OpsCounter, Value};
use alloc::vec::Vec;
use hashbrown::HashSet;

/// A reusable profiler component that tracks used flop slots during backend compilation.
#[derive(Default, Clone, Debug)]
pub struct CompilerProfiler<C: ProfileCompiler> {
    flops_used: HashSet<u32>,
    compiler: C,
}

pub trait ProfileCompiler {
    type I;

    fn increment_instruction(&self, slot: u32, amount: u32) -> Self::I;
    fn prefix_instruction(&self, slot: u32) -> Self::I;
    fn suffix_instruction(&self, slot: u32) -> Self::I;
}

impl<C: ProfileCompiler> CompilerProfiler<C> {
    pub fn new(compiler: C) -> Self {
        Self {
            flops_used: HashSet::default(),
            compiler,
        }
    }

    /// Mutable access to the backend compiler, e.g. to configure it before emitting prefix/suffix.
    pub fn compiler_mut(&mut self) -> &mut C {
        &mut self.compiler
    }

    /// Intercepts an operation, tracking any used counters (flops, etc).
    /// The backend provides a formatting closure to emit its specific increment instruction.
    pub fn profile_operation(
        &mut self,
        operation: &Operation,
        out: Option<&Value>,
        instructions: &mut Vec<<C as ProfileCompiler>::I>,
    ) {
        if matches!(operation, Operation::Branch(Branch::Return)) {}

        if let Some(CountInfo { slot, amount }) = OpsCounter::count(operation, out) {
            self.flops_used.insert(slot);
            instructions.push(self.compiler.increment_instruction(slot, amount));
        }
    }

    /// Iterates over all used profiling slots and generates initialization (prefix)
    /// and aggregation (suffix) instructions via the provided formatting closures.
    pub fn profile(&self, instructions: &mut Vec<<C as ProfileCompiler>::I>) {
        let mut prefix = Vec::new();
        for slot in self.used_slots() {
            prefix.push(self.compiler.prefix_instruction(*slot));
        }

        let mut old_instructions = core::mem::take(instructions);

        instructions.extend(prefix);
        instructions.append(&mut old_instructions);

        for slot in self.used_slots() {
            instructions.push(self.compiler.suffix_instruction(*slot));
        }
    }

    /// Returns the flop slots that were used during compilation.
    /// The backend can use this to emit initialization and flush instructions.
    pub fn used_slots(&self) -> &HashSet<u32> {
        &self.flops_used
    }
}
