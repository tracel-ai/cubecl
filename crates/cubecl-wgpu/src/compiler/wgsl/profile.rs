use cubecl_core::CompilerProfilerInstructions;

use crate::compiler::wgsl;

#[derive(Clone, Default)]
pub struct WgslCompilerProfiler;

impl CompilerProfilerInstructions for WgslCompilerProfiler {
    type Instruction = wgsl::Instruction;
    type CounterVariable = wgsl::Value;

    fn increment_instruction(&self, slot: u32, amount: u32) -> Self::Instruction {
        wgsl::Instruction::Custom(format!("flop_{slot} += {amount}u;"))
    }

    fn declare_instruction(&self, slot: u32) -> Self::Instruction {
        wgsl::Instruction::Custom(format!("var flop_{slot} = 0u;"))
    }

    fn flush_instruction(&self, slot: u32, counter: &Self::CounterVariable) -> Self::Instruction {
        wgsl::Instruction::Custom(format!("atomicAdd(&{counter}[{slot}], flop_{slot});"))
    }
}
