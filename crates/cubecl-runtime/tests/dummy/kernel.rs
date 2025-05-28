use cubecl_runtime::{id::KernelId, storage::BytesResource};

/// The DummyKernel trait should be implemented for every supported operation
pub trait DummyKernel: Sync + Send + 'static + core::fmt::Debug {
    fn compute(&self, resources: &mut [&BytesResource]);

    fn id(&self) -> KernelId;

    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }
}

/// Contains the algorithm for element-wise addition
#[derive(Debug)]
pub struct DummyElementwiseAddition;

impl DummyKernel for DummyElementwiseAddition {
    fn compute(&self, inputs: &mut [&BytesResource]) {
        // Notice how the kernel is responsible for determining which inputs
        // are read-only and which are writable.
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            out[i] = lhs[i] + rhs[i];
        }
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}
