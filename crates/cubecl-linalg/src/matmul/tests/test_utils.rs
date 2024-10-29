use cubecl_core::{client::ComputeClient, server::Handle, CubeElement, Runtime};

use crate::matmul::problem::MatmulProblem;

pub(crate) fn assert_equals_approx<I: CubeElement, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read(output.binding());
    let actual = I::from_bytes(&actual);
    println!("{:?}", actual);
    println!("{:?}", expected);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = I::to_f32_value(*a);
        if (a - e).abs() >= epsilon {
            return Err(format!(
            "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
            i,
            a,
            e,
            (a - e).abs(),
            epsilon
            ));
        }
    }

    Ok(())
}

pub(crate) fn generate_random_data(num_elements: usize) -> Vec<f32> {
    fn lcg(seed: &mut u64) -> f32 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: f64 = 2u64.pow(32) as f64;

        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
        (*seed as f64 / M * 2.0 - 1.0) as f32
    }

    let mut seed = 12345;

    (0..num_elements).map(|_| lcg(&mut seed)).collect()
}

pub(crate) fn matmul_cpu_reference(lhs: &[f32], rhs: &[f32], problem: MatmulProblem) -> Vec<f32> {
    let m = problem.m as usize;
    let n = problem.n as usize;
    let k = problem.k as usize;
    let b = problem.num_batches();

    let mut out = vec![0.; m * n * b];

    for b_ in 0..b {
        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    out[(b_ * m * n) + i * n + j] +=
                        lhs[(b_ * m * k) + i * k + k_] * rhs[(b_ * k * n) + k_ * n + j];
                }
            }
        }
    }

    out
}
