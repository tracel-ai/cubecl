use std::{thread::sleep, time::Duration};

use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::{compiler::CompilationError, id::KernelId, storage::BytesResource};

use crate::dummy::DummyKernel;

const SLEEP_MS: u64 = 1;

#[derive(Debug)]
pub struct DummyElementwiseAdditionSlowWrong;
#[derive(Debug)]
pub struct DummyElementwiseMultiplication;
#[derive(Debug)]
pub struct DummyElementwiseMultiplicationSlowWrong;

impl DummyKernel for DummyElementwiseAdditionSlowWrong {
    fn compute(&self, inputs: &mut [&mut BytesResource]) {
        // Slow and wrong on purpose, for tests
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            sleep(Duration::from_millis(SLEEP_MS));
            out[i] = lhs[i]
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl DummyKernel for DummyElementwiseMultiplication {
    fn compute(&self, inputs: &mut [&mut BytesResource]) {
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            out[i] = lhs[i] * rhs[i];
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl DummyKernel for DummyElementwiseMultiplicationSlowWrong {
    fn compute(&self, inputs: &mut [&mut BytesResource]) {
        // Slow and wrong on purpose, for tests
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            sleep(Duration::from_millis(SLEEP_MS));
            out[i] = lhs[i];
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// A kernel whose compilation always fails, standing in for a kernel the backend refuses
/// to compile. `compute` is unreachable: the launch records the error and skips execution.
#[derive(Debug)]
pub struct DummyElementwiseAdditionBrokenCompilation;

impl DummyKernel for DummyElementwiseAdditionBrokenCompilation {
    fn compute(&self, _inputs: &mut [&mut BytesResource]) {
        unreachable!("a kernel that fails compilation never runs");
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }

    fn compilation_error(&self) -> Option<CompilationError> {
        Some(CompilationError::UnsupportedInstruction {
            reason: "this dummy kernel never compiles".to_string(),
            backtrace: BackTrace::capture(),
        })
    }
}
