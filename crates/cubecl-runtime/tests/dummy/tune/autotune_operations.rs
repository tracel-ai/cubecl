use cubecl_runtime::{
    client::ComputeClient,
    server::{Bindings, CubeCount, Handle},
    tune::{AutotuneError, TuneFn},
};
use derive_new::new;

use crate::dummy::{DummyRuntime, KernelTask};

#[derive(new, Clone)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct OneKernelAutotuneOperation {
    kernel: KernelTask,
    client: ComputeClient<DummyRuntime>,
}

impl TuneFn for OneKernelAutotuneOperation {
    type Inputs = Vec<Handle>;
    type Output = ();

    fn execute(&self, inputs: Vec<Handle>) -> Result<(), AutotuneError> {
        let result = self.client.launch(
            Box::new(self.kernel.clone()),
            CubeCount::Static(1, 1, 1),
            Bindings::new().with_buffers(inputs),
        );

        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(AutotuneError::Launch(err)),
        }
    }

    fn name(&self) -> &str {
        "OneKernelAutotuneOperation"
    }
}
