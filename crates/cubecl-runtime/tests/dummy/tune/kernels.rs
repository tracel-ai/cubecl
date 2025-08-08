use std::{thread::sleep, time::Duration};

use cubecl_runtime::{id::KernelId, storage::BytesResource};

use crate::dummy::DummyKernel;

const SLEEP_MS: u64 = 1;

#[derive(Debug)]
pub struct DummyElementwiseAdditionSlowWrong;
#[derive(Debug)]
pub struct DummyElementwiseMultiplication;
#[derive(Debug)]
pub struct DummyElementwiseMultiplicationSlowWrong;

impl DummyKernel for DummyElementwiseAdditionSlowWrong {
    fn compute(&self, inputs: &mut [&BytesResource]) {
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
    fn compute(&self, inputs: &mut [&BytesResource]) {
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
    fn compute(&self, inputs: &mut [&BytesResource]) {
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
