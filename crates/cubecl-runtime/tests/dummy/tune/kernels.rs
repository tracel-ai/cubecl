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

#[derive(Debug)]
pub struct CacheTestFastOn3;
#[derive(Debug)]
pub struct CacheTestSlowOn3;
#[derive(Debug)]
pub struct ParameteredKernel;

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

impl DummyKernel for CacheTestFastOn3 {
    fn compute(&self, inputs: &mut [&BytesResource]) {
        // This is an artificial kernel designed for testing cache only
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();
        if size == 3 {
            out[..size].copy_from_slice(&lhs[..size]);
        } else {
            for i in 0..size {
                sleep(Duration::from_millis(SLEEP_MS));
                out[i] = lhs[i];
            }
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl DummyKernel for CacheTestSlowOn3 {
    fn compute(&self, inputs: &mut [&BytesResource]) {
        // This is an artificial kernel designed for testing cache only
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();
        if size == 3 {
            for i in 0..size {
                sleep(Duration::from_millis(SLEEP_MS));
                out[i] = rhs[i];
            }
        } else {
            out[..size].copy_from_slice(&rhs[..size]);
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

impl DummyKernel for ParameteredKernel {
    fn compute(&self, inputs: &mut [&BytesResource]) {
        // This is an artificial kernel designed for info buffer
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();
        let info = &inputs[3].read();

        for i in 0..lhs.len() {
            out[i] = lhs[i] + rhs[i] + info[0];
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}
