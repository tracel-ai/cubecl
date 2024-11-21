use cubecl::{
    client::ComputeClient,
    linalg::{matmul::Strategy, tensor::TensorHandle},
    prelude::Float,
    Runtime,
};

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl_linalg::matmul::kernels::cmma_old::PredefinedCmmaConfig;

        let args = parse();
        run::<cubecl::cuda::CudaRuntime, half::f16>(Strategy::Accelerated, args);
        run::<cubecl::cuda::CudaRuntime, half::f16>(
            Strategy::CmmaOld(PredefinedCmmaConfig::M128K16.into()),
            args,
        );
    }
    #[cfg(feature = "wgpu-spirv")]
    {
        use cubecl_linalg::matmul::kernels::cmma_old::PredefinedCmmaConfig;
        println!("A");

        type C = cubecl::wgpu::spirv::SpirvCompiler;
        type R = cubecl::wgpu::WgpuRuntime<C>;

        let args = parse();
        run::<R, half::f16>(Strategy::Accelerated, args);
        run::<R, half::f16>(
            Strategy::CmmaOld(PredefinedCmmaConfig::M128K16.into()),
            args,
        );
    }
}

pub fn parse() -> [usize; 4] {
    use std::env;

    const USAGE: &str = "Usage: <batch> <m> <n> <k>";

    // Collect command-line arguments
    let args: Vec<usize> = env::args()
        .skip(1)
        .map(|arg| str::parse::<usize>(&arg).expect(USAGE))
        .collect();

    args.try_into().expect(USAGE)
}

pub fn run<R: Runtime, E: Float>(strategy: Strategy, args: [usize; 4]) {
    let profiling = Profiling::<R, E>::new(strategy, args[0], args[1], args[2], args[3]);
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
