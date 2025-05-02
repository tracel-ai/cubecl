use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, TimingMethod},
    client::ComputeClient,
    future,
};
use cubecl_linalg::{
    convolution::{self, ConvolutionArgs, algorithm::simple::SimpleConvAlgorithm},
    matmul::components::{MatmulPrecision, tile::accelerated::Accelerated},
    tensor::TensorHandle,
};

use cubecl::prelude::*;

impl<R: Runtime, MP: MatmulPrecision> Benchmark for Conv2dBench<R, MP> {
    type Args = (
        TensorHandle<R, MP::EI>,
        TensorHandle<R, MP::EI>,
        TensorHandle<R, MP::EI>,
    );

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        let input = TensorHandle::zeros(&client, self.input_shape.to_vec());
        let weight = TensorHandle::zeros(&client, self.weight_shape.to_vec());
        let bias = TensorHandle::zeros(&client, vec![self.bias_shape]);

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Args) {
        let client = R::client(&self.device);
        let [n, _, h_in, w_in] = self.input_shape;
        let [c_out, _, k_h, k_w] = self.weight_shape;
        let [s_h, s_w] = self.args.stride;
        let [p_h, p_w] = self.args.padding;
        let [d_h, d_w] = self.args.dilation;

        let h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
        let w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;

        let out: TensorHandle<R, MP::EO> =
            TensorHandle::empty(&client, vec![n, c_out, h_out, w_out]);

        convolution::launch_conv::<R, MP, SimpleConvAlgorithm<Accelerated>, 2>(
            &self.client,
            &input.as_ref(),
            &weight.as_ref(),
            &Some(bias.as_ref()),
            &out.as_ref(),
            self.args.clone(),
        )
        .unwrap();
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "{}-conv2d-{}-{}-{}-{}",
            R::name(&client),
            MP::EI::as_elem_native_unchecked(),
            MP::ES::as_elem_native_unchecked(),
            MP::EA::as_elem_native_unchecked(),
            MP::EO::as_elem_native_unchecked(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Args) -> cubecl::benchmark::ProfileDuration {
        self.client.profile(|| self.execute(args))
    }
}

#[allow(dead_code)]
pub struct Conv2dBench<R: Runtime, MP> {
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    bias_shape: usize,
    args: ConvolutionArgs<2>,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _phantom: PhantomData<MP>,
}

#[allow(dead_code)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device) {
    let client = R::client(&device);
    let batch_size = 16;

    let bench1 = Conv2dBench::<R, MP> {
        input_shape: [batch_size, 3, 227, 227],
        weight_shape: [96, 3, 11, 11],
        bias_shape: 96,
        args: ConvolutionArgs {
            stride: [4, 4],
            padding: [0, 0],
            dilation: [1, 1],
        },
        client: client.clone(),
        device: device.clone(),
        _phantom: PhantomData,
    };

    let bench2 = Conv2dBench::<R, MP> {
        input_shape: [batch_size, 4, 256, 256],
        weight_shape: [64, 4, 8, 8],
        bias_shape: 64,
        args: ConvolutionArgs {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        },
        client: client.clone(),
        device: device.clone(),
        _phantom: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!(
            "input: {:?} weight: {:?}",
            bench.input_shape, bench.weight_shape
        );
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Full));
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    run::<R, MP>(Default::default());
}

fn main() {
    #[cfg(feature = "wgpu")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, f32>();
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }

    #[cfg(all(feature = "hip", target_os = "linux"))]
    {
        run_benches::<cubecl::hip::HipRuntime, half::f16>();
    }

    #[cfg(feature = "cuda")]
    {
        run_benches::<cubecl::cuda::CudaRuntime, half::f16>();
    }

    #[cfg(feature = "wgpu-msl")]
    {
        run_benches::<cubecl::wgpu::WgpuRuntime, half::f16>();
    }
}
