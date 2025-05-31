use cubecl_runtime::{
    client::ComputeClient,
    server::{Binding, Bindings, CubeCount},
    tune::{AutotuneError, Tunable},
};
use derive_new::new;

use crate::dummy::{DummyChannel, DummyServer, KernelTask};

#[derive(new, Debug, Clone)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: KernelTask,
    client: ComputeClient<DummyServer, DummyChannel>,
}

impl Tunable for OneKernelAutotuneOperation {
    type Inputs = Vec<Binding>;
    type Output = ();

    fn execute(&self, inputs: Vec<Binding>) -> Result<(), AutotuneError> {
        self.client.execute(
            self.kernel.clone(),
            CubeCount::Static(1, 1, 1),
            Bindings::new().with_buffers(inputs),
        );

        Ok(())
    }
}
