use std::marker::PhantomData;

use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
};
use cubecl_convolution::ConvolutionArgs;
use cubecl_convolution::{
    self as convolution, kernels::layered::algorithm::simple::SimpleConvAlgorithm,
};
use cubecl_matmul::{
    MatmulInputHandleRef,
    components::{
        AccG, AccR, LhsG, LhsS, MatmulPrecision, RhsG,
        tile::{cmma::CmmaMatmul, io::Strided},
    },
};
use cubecl_std::{CubeOption, tensor::TensorHandle};

use cubecl_random::random_uniform;

use cubecl::prelude::*;

impl<R: Runtime, MP: MatmulPrecision> Benchmark for Conv2dBench<R, MP> {
    type Input = (
        TensorHandle<R, LhsG<MP>>,
        TensorHandle<R, RhsG<MP>>,
        TensorHandle<R, AccG<MP>>,
    );
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let input = TensorHandle::<R, LhsG<MP>>::empty(&client, self.input_shape.to_vec());
        random_uniform::<R, LhsG<MP>>(
            &client,
            LhsG::<MP>::from_int(0),
            LhsG::<MP>::from_int(1),
            input.as_ref(),
        );
        let weight = TensorHandle::<R, RhsG<MP>>::empty(&client, self.weight_shape.to_vec());
        random_uniform::<R, RhsG<MP>>(
            &client,
            RhsG::<MP>::from_int(0),
            RhsG::<MP>::from_int(1),
            weight.as_ref(),
        );
        let bias = TensorHandle::<R, AccG<MP>>::empty(&client, vec![self.bias_shape]);
        random_uniform::<R, AccG<MP>>(
            &client,
            AccG::<MP>::from_int(0),
            AccG::<MP>::from_int(1),
            bias.as_ref(),
        );

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Input) -> Result<(), String> {
        let client = R::client(&self.device);
        let [n, _, h_in, w_in] = self.input_shape;
        let [c_out, _, k_h, k_w] = self.weight_shape;
        let [s_h, s_w] = self.args.stride;
        let [p_h, p_w] = self.args.padding;
        let [d_h, d_w] = self.args.dilation;

        let h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1;
        let w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1;

        let out: TensorHandle<R, AccG<MP>> =
            TensorHandle::empty(&client, vec![n, c_out, h_out, w_out]);

        convolution::launch_conv::<R, MP, SimpleConvAlgorithm<CmmaMatmul<CubeOption<Strided>>>, 2>(
            &self.client,
            &MatmulInputHandleRef::Normal(input.as_ref()),
            &MatmulInputHandleRef::Normal(weight.as_ref()),
            &Some(bias.as_ref()),
            &out.as_ref(),
            self.args.clone(),
        )
        .map_err(|it| format!("{it:?}"))?;
        Ok(())
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        format!(
            "{}-conv2d-{}-{}-{}-{}",
            R::name(&client),
            LhsG::<MP>::as_type_native_unchecked(),
            LhsS::<MP>::as_type_native_unchecked(),
            AccR::<MP>::as_type_native_unchecked(),
            AccG::<MP>::as_type_native_unchecked(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "conv-bench")
            .map_err(|it| format!("{it:?}"))
    }
}

#[allow(dead_code)]
pub struct Conv2dBench<R: Runtime, MP> {
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    bias_shape: usize,
    args: ConvolutionArgs<2>,
    device: R::Device,
    client: ComputeClient<R>,
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
        println!("{}", bench.run(TimingMethod::System).unwrap());
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
