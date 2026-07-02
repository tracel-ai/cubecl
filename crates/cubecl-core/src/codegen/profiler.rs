use alloc::{fmt::Display, vec::Vec};
use cubecl_ir::{Branch, CountInfo, Operation, OpsCounter, Value};
use hashbrown::HashSet;

/// A reusable profiler component that tracks used flop slots during backend compilation.
#[derive(Clone, Debug)]
pub struct CompilerProfiler<C: CompilerProfilerInstructions> {
    flops_used: HashSet<u32>,
    compiler: C,
    counter: Option<C::CounterVariable>,
}

impl<C: CompilerProfilerInstructions> Default for CompilerProfiler<C> {
    fn default() -> Self {
        Self {
            flops_used: HashSet::default(),
            compiler: C::default(),
            counter: Option::default(),
        }
    }
}

pub trait CompilerProfilerInstructions: Default {
    type Instruction;
    type CounterVariable: Display;

    /// Returns an instruction that increments the value of the given slot by the given amount.
    fn increment_instruction(&self, slot: u32, amount: u32) -> Self::Instruction;
    /// Returns an instruction that prefixes the given slot, e.g. to initialize it to zero.
    fn declare_instruction(&self, slot: u32) -> Self::Instruction;
    /// Returns an instruction that suffixes the given slot, e.g. to aggregate its value.
    fn flush_instruction(&self, slot: u32, counter: &Self::CounterVariable) -> Self::Instruction;
}

impl<C: CompilerProfilerInstructions> CompilerProfiler<C> {
    pub fn new(compiler: C) -> Self {
        Self {
            flops_used: HashSet::default(),
            compiler,
            counter: None,
        }
    }

    pub fn set_counter(&mut self, counter: C::CounterVariable) {
        self.counter = Some(counter)
    }

    /// Intercepts an operation, tracking any used counters (flops, etc).
    /// The backend provides a formatting closure to emit its specific increment instruction.
    pub fn profile_operation(
        &mut self,
        operation: &Operation,
        out: Option<&Value>,
        instructions: &mut Vec<<C as CompilerProfilerInstructions>::Instruction>,
    ) {
        if let Some(counter) = &self.counter {
            if matches!(operation, Operation::Branch(Branch::Return)) {
                for slot in &self.flops_used {
                    instructions.push(self.compiler.flush_instruction(*slot, counter));
                }
                return;
            }

            if let Some(CountInfo { slot, amount }) = OpsCounter::count(operation, out) {
                self.flops_used.insert(slot);
                instructions.push(self.compiler.increment_instruction(slot, amount));
            }
        }
    }

    /// Iterates over all used profiling slots and generates initialization (prefix)
    /// and aggregation (suffix) instructions via the provided formatting closures.
    pub fn profile(
        &self,
        instructions: &mut Vec<<C as CompilerProfilerInstructions>::Instruction>,
    ) {
        if let Some(counter) = &self.counter {
            let mut declares = Vec::new();
            for slot in &self.flops_used {
                declares.push(self.compiler.declare_instruction(*slot));
            }

            let mut old_instructions = core::mem::take(instructions);

            instructions.extend(declares);
            instructions.append(&mut old_instructions);

            for slot in &self.flops_used {
                instructions.push(self.compiler.flush_instruction(*slot, counter));
            }
        }
    }
}
