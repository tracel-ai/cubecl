use std::sync::Arc;

use cubecl_runtime::{
    client::ComputeClient,
    server::{Binding, CubeCount},
    tune::AutotuneOperation,
};
use derive_new::new;

use crate::dummy::{DummyChannel, DummyKernel, DummyServer};

#[derive(new, Debug)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: Arc<dyn DummyKernel>,
    client: ComputeClient<DummyServer, DummyChannel>,
    shapes: Vec<Vec<usize>>,
    bindings: Vec<Binding>,
}

impl AutotuneOperation for OneKernelAutotuneOperation {
    /// Executes the operation on given bindings and server, with the additional parameters
    fn execute(self: Box<Self>) {
        self.client.execute(
            self.kernel.clone(),
            CubeCount::Static(1, 1, 1),
            self.bindings,
        );
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            client: self.client.clone(),
            shapes: self.shapes.clone(),
            bindings: self.bindings.clone(),
        })
    }
}
