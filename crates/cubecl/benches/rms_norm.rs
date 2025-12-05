use cubecl::future;
use cubecl::prelude::*;
use cubecl_std::tensor::{TensorHandle, rms_norm};
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() {
    use cubecl::cuda::{CudaDevice, CudaRuntime};

    println!("=== SIMPLE RMSNORM BENCHMARK ===\n");

    let device = CudaDevice::default();
    let shapes = vec![
        vec![8, 1024, 4096],
        vec![16, 1024, 4096],
        vec![32, 2048, 4096],
        vec![64, 4096, 4096],
    ];

    for shape in &shapes {
        let total_elems = shape.iter().product::<usize>();
        println!(
            "Shape: {:?} ({:.2}M elements)",
            shape,
            total_elems as f64 / 1e6
        );
        println!("Type |   Time(ms)  |  Throughput(Gelem/s)");
        println!("-----|--------------|----------------------");

        bench_dtype::<CudaRuntime, f32>(&device, shape, "F32");
        bench_dtype::<CudaRuntime, half::f16>(&device, shape, "F16");
        bench_dtype::<CudaRuntime, half::bf16>(&device, shape, "BF16");

        println!();
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Requires CUDA. Run with --features cuda");
}

fn bench_dtype<R: Runtime, F: cubecl::frontend::Float + 'static>(
    device: &R::Device,
    shape: &[usize],
    dtype_name: &str,
) {
    let client = R::client(device);
    let total_elems: usize = shape.iter().product();
    let cols = shape[shape.len() - 1];

    let input = TensorHandle::<R, F>::zeros(&client, shape.to_vec());
    let weight = TensorHandle::<R, F>::zeros(&client, vec![cols]);
    let output = TensorHandle::<R, F>::empty(&client, shape.to_vec());

    // Warmup
    for _ in 0..5 {
        rms_norm::launch_ref::<R, F>(
            &client,
            input.as_ref(),
            weight.as_ref(),
            None,
            output.as_ref(),
            1e-5,
        );
    }
    future::block_on(client.sync());

    // Benchmark (async launches, one sync at end)
    let iterations = 15;
    let start = Instant::now();
    for _ in 0..iterations {
        rms_norm::launch_ref::<R, F>(
            &client,
            input.as_ref(),
            weight.as_ref(),
            None,
            output.as_ref(),
            1e-5,
        );
    }
    future::block_on(client.sync());

    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = (elapsed / iterations as f64) * 1000.0;
    let throughput = (total_elems as f64 / 1e9) / (avg_ms / 1000.0);

    println!("{:4} | {:10.3} | {:20.2}", dtype_name, avg_ms, throughput);
}
