use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::{
    Runtime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
};
use cubecl_attention::components::{AttentionPrecision, AttentionProblem};
use cubecl_convolution::ConvolutionArgs;
use cubecl_convolution::{
    self as convolution, kernels::layered::algorithm::simple::SimpleConvAlgorithm,
};
use cubecl_random::random_uniform;
use cubecl_std::{CubeOption, tensor::TensorHandle};

struct AttentionInputs<R: Runtime> {
    query: TensorHandle<R, QG<AP>>,
    key: TensorHandle<R, KG<AP>>,
    value: TensorHandle<R, VG<AP>>,
    mask: TensorHandle<R, MSK<AP>>,
    out: TensorHandle<R, OG<AP>>,
}

impl<R: Runtime, AP: AttentionPrecision> Benchmark for AttentionBench<R, AP> {
    type Input = AttentionInputs<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let input = TensorHandle::<R, QG<AP>>::empty(&client, self.input_shape.to_vec());
        random_uniform::<R, QG<AP>>(
            &client,
            QG::<AP>::from_int(0),
            QG::<AP>::from_int(1),
            input.as_ref(),
        );
        let weight = TensorHandle::<R, KG<AP>>::empty(&client, self.weight_shape.to_vec());
        random_uniform::<R, KG<AP>>(
            &client,
            KG::<AP>::from_int(0),
            KG::<AP>::from_int(1),
            weight.as_ref(),
        );
        let bias = TensorHandle::<R, VG<AP>>::empty(&client, vec![self.bias_shape]);
        random_uniform::<R, VG<AP>>(
            &client,
            VG::<AP>::from_int(0),
            VG::<AP>::from_int(1),
            bias.as_ref(),
        );

        (input, weight, bias)
    }

    fn execute(&self, (input, weight, bias): Self::Input) -> Result<(), String> {
        let client = R::client(&self.device);
        let out: TensorHandle<R, OG<AP>> =
            TensorHandle::empty(&client, vec![n, c_out, h_out, w_out]);

        attention::launch_conv::<R, AP, SimpleConvAlgorithm<CmmaMatmul<CubeOption<Strided>>>, 2>(
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
            "{}-attention-{}-{}-{}-{}",
            R::name(&client),
            QG::<AP>::as_type_native_unchecked(),
            KG::<AP>::as_type_native_unchecked(),
            VG::<AP>::as_type_native_unchecked(),
            OG::<AP>::as_type_native_unchecked(),
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "attention-bench")
            .map_err(|it| format!("{it:?}"))
    }
}

#[allow(dead_code)]
pub struct AttentionBench<R: Runtime, AP> {
    problem: AttentionProblem,
    device: R::Device,
    client: ComputeClient<R::Server>,
    _phantom: PhantomData<AP>,
}

#[allow(dead_code)]
fn run<R: Runtime, AP: AttentionPrecision>(device: R::Device) {
    let client = R::client(&device);
    let batch_size = 16;

    let bert = AttentionProblem {
        batch: 8,
        num_heads: 12,
        seq_q: 128,
        seq_kv: 128,
        head_dim: 64,
        val_dim: 64,
        masked: false,
        causal: false,
    };
    let gpt2 = AttentionProblem {
        batch: 4,
        num_heads: 12,
        seq_q: 1024,
        seq_kv: 1024,
        head_dim: 64,
        val_dim: 64,
        masked: true,
        causal: true,
    };
    let llama = AttentionProblem {
        batch: 4,
        num_heads: 32,
        seq_q: 2048,
        seq_kv: 2048,
        head_dim: 128,
        val_dim: 128,
        masked: true,
        causal: true,
    };
    let long_context = AttentionProblem {
        batch: 1,
        num_heads: 16,
        seq_q: 4096,
        seq_kv: 4096,
        head_dim: 128,
        val_dim: 128,
        masked: true,
        causal: true,
    };
    let encoder_decoder = AttentionProblem {
        batch: 2,
        num_heads: 16,
        seq_q: 512,
        seq_kv: 1024,
        head_dim: 128,
        val_dim: 128,
        masked: false,
        causal: false,
    };

    for problem in [bert, gpt2, llama, long_context, encoder_decoder] {
        let bench = AttentionBench::<R, MP> {
            problem,
            client: client.clone(),
            device: device.clone(),
            _phantom: PhantomData,
        };

        println!("problem: {:?}", bench.problem);
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System).unwrap());
    }
}

#[allow(unused)]
fn run_benches<R: Runtime, AP: AttentionPrecision>() {
    let client = R::client(&Default::default());

    run::<R, AP>(Default::default());
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
