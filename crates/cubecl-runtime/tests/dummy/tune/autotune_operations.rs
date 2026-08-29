use cubecl_runtime::{
    client::ComputeClient,
    kernel::BufferIOAttr,
    server::{CubeCount, Handle, KernelArguments},
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

impl OneKernelAutotuneOperation {
    pub fn run(&self, inputs: Vec<Handle>) -> Result<(), String> {
        // Every dummy kernel reads its leading handles and writes the last
        // one, so the launch declares exactly that — what a generated launch
        // function would declare from `&Tensor` versus `&mut Tensor`. The
        // declaration is what keeps a candidate that fails to compile from
        // tainting the inputs every other candidate still has to read.
        let last = inputs.len().saturating_sub(1);
        let mut args = KernelArguments::new();
        for (position, handle) in inputs.into_iter().enumerate() {
            let io = match position == last {
                true => BufferIOAttr::WriteOnly,
                false => BufferIOAttr::ReadOnly,
            };
            args = args.with_buffer_io(handle.binding(), io);
        }
        self.client.launch(
            Box::new(self.kernel.clone()),
            CubeCount::Static(1, 1, 1),
            args,
        );
        Ok(())
    }
}
