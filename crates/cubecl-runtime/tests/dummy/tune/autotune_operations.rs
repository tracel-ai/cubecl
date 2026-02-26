use cubecl_runtime::{
    client::ComputeClient,
    server::{KernelArguments, CubeCount, Handle},
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
    type Inputs = Vec<Handle<DummyRuntime>>;
    type Output = ();

    fn execute(&self, inputs: Vec<Handle<DummyRuntime>>) -> Result<(), AutotuneError> {
        self.client.launch(
            Box::new(self.kernel.clone()),
            CubeCount::Static(1, 1, 1),
            KernelArguments::new().with_buffers(inputs.into_iter().map(|h| h.binding()).collect()),
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "OneKernelAutotuneOperation"
    }
}
