use cubecl::{
    client::ComputeClient,
    linalg::{matmul::Strategy, tensor::TensorHandle},
    prelude::Float,
    Runtime,
};

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, half::f16>(Strategy::Accelerated);
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Strategy::PlaneMma);
}

pub fn run<R: Runtime, E: Float>(strategy: Strategy) {
    let profiling = Profiling::<R, E>::new(strategy, 1, 4096, 4096, 4096);
    profiling.run();
}

struct Profiling<R: Runtime, E: Float> {
    lhs: TensorHandle<R, E>,
    rhs: TensorHandle<R, E>,
    out: TensorHandle<R, E>,
    strategy: Strategy,
    client: ComputeClient<R::Server, R::Channel>,
}

impl<R: Runtime, E: Float> Profiling<R, E> {
    pub fn new(strategy: Strategy, b: usize, m: usize, n: usize, k: usize) -> Self {
        let client = R::client(&Default::default());

        Self {
            lhs: TensorHandle::zeros(&client, vec![b, m, k]),
            rhs: TensorHandle::zeros(&client, vec![b, k, n]),
            out: TensorHandle::zeros(&client, vec![b, m, n]),
            strategy,
            client,
        }
    }

    pub fn run(&self) {
        println!("Running on {} ...", R::name());
        let time = std::time::Instant::now();
        cubecl::linalg::matmul::launch_ref::<R, E>(
            &self.strategy,
            &self.client,
            self.lhs.as_ref(),
            self.rhs.as_ref(),
            self.out.as_ref(),
        );
        cubecl::future::block_on(self.client.sync());
        println!("Done {:?}", time.elapsed());
    }
}
