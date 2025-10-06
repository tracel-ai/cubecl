#![allow(clippy::needless_range_loop)]

use std::fmt::Display;

use cubecl_core::{
    CubeElement, Runtime,
    client::ComputeClient,
    flex32,
    prelude::{CubePrimitive, Exp, Float, Numeric},
    server::{self},
    tf32,
};

use cubecl_runtime::MmaConfig;
use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{AttentionIdent, AttentionPrecision, AttentionProblem},
    tests::attention_test_launcher::{strides, tensor_size},
};

pub trait TestPrecision {
    type EG: Float + CubeElement + Display + CastInto<Self::ES> + Sampleable;
    type ES: Float + Display + CastInto<Self::EA>;
    type EA: Float + Display + CastInto<Self::EG> + Exp;
    type EM: Numeric + CubeElement + Display + Sampleable;
    type AP: AttentionPrecision;

    #[allow(clippy::too_many_arguments)]
    fn assert_result<R: Runtime>(
        query: &[Self::EG],
        key: &[Self::EG],
        value: &[Self::EG],
        mask: Option<&[Self::EM]>,
        problem: &AttentionProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        shape: &[usize],
        strides: &[usize],
    );
}

impl<EG, ES> TestPrecision for (EG, ES)
where
    EG: Float + CubeElement + Display + CastInto<ES> + Sampleable + AttentionPrecision,
    ES: Float + Display + CastInto<f32>,
    f32: CastInto<EG>,
{
    type EG = EG;
    type ES = ES;
    type EA = f32;
    type EM = u8;
    type AP = EG;

    fn assert_result<R: Runtime>(
        query: &[EG],
        key: &[EG],
        value: &[EG],
        mask: Option<&[u8]>,
        problem: &AttentionProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        shape: &[usize],
        strides: &[usize],
    ) {
        let maybe_f16 = client.properties().features.cmma.contains(&MmaConfig {
            a_type: ES::as_type_native().expect("To be a native type"),
            b_type: ES::as_type_native().expect("To be a native type"),
            cd_type: EG::as_type_native().expect("To be a native type"),
            m: 16,
            k: 16,
            n: 16,
        });
        let maybe_tf32 = client.properties().features.cmma.contains(&MmaConfig {
            a_type: ES::as_type_native().expect("To be a native type"),
            b_type: ES::as_type_native().expect("To be a native type"),
            cd_type: EG::as_type_native().expect("To be a native type"),
            m: 16,
            k: 8,
            n: 16,
        });

        // Need to compensate for the temporary conversion to f16/tf32
        let epsilon = match maybe_f16 || maybe_tf32 {
            true => 10e-3 / EG::EPSILON.to_f32().unwrap() * half::f16::EPSILON.to_f32(),
            false => 10e-3,
        };

        let expected = flash_attention_v2_cpu::<Self>(query, key, value, mask, problem)
            .into_iter()
            .collect::<Vec<EG>>();

        if let Err(e) =
            assert_equals_approx::<R, EG>(client, out, shape, strides, &expected, epsilon)
        {
            panic!("{}", e);
        }
    }
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: server::Handle,
    shape: &[usize],
    strides: &[usize],
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read_one_tensor(output.copy_descriptor(shape, strides, size_of::<F>()));
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        // println!("{:?}: {:?}, {:?}", i, a, e);
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

        if f32::is_nan(a.to_f32().unwrap())
            || f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error
        {
            return Err(format!(
                "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                i,
                *a,
                *e,
                f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
                epsilon
            ));
        }
    }

    Ok(())
    // Err("".to_string())
}

pub trait CastInto<E> {
    fn cast_into(self) -> E;
}

impl<E> CastInto<E> for E {
    fn cast_into(self) -> E {
        self
    }
}

impl CastInto<f32> for half::f16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for half::bf16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for flex32 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<half::bf16> for f32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self)
    }
}

impl CastInto<half::bf16> for half::f16 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for half::bf16 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for f32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self)
    }
}

impl CastInto<half::f16> for flex32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::bf16> for flex32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<flex32> for f32 {
    fn cast_into(self) -> flex32 {
        flex32::from_f32(self)
    }
}

impl CastInto<f32> for tf32 {
    fn cast_into(self) -> f32 {
        self.to_f32()
    }
}

impl CastInto<tf32> for f32 {
    fn cast_into(self) -> tf32 {
        tf32::from_f32(self)
    }
}

impl CastInto<u16> for u8 {
    fn cast_into(self) -> u16 {
        self as u16
    }
}

impl CastInto<i32> for u16 {
    fn cast_into(self) -> i32 {
        self as i32
    }
}

impl CastInto<u8> for i32 {
    fn cast_into(self) -> u8 {
        self as u8
    }
}

pub trait Sampleable: Sized + CubePrimitive {
    fn sample<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R, Self>;
}

macro_rules! sample_float {
    ($($t:ty),*) => {
        $(
            impl Sampleable for $t
            {
                fn sample<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>, shape: &[usize], seed: u64) -> TensorHandle::<R, Self> {
                    cubecl_random::seed(seed);
                    let output = TensorHandle::<R, Self>::empty(client, shape.to_vec());

                    cubecl_random::random_uniform::<R, Self>(&client, Self::from_int(-50), Self::from_int(50), output.as_ref());

                    output
                }
            }
        )*
    };
}

sample_float!(half::f16);
sample_float!(half::bf16);
sample_float!(f32);
sample_float!(f64);
sample_float!(u8);

impl Sampleable for flex32 {
    fn sample<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R, Self> {
        cubecl_random::seed(seed);
        let output = TensorHandle::<R, flex32>::empty(client, shape.to_vec());

        cubecl_random::random_uniform::<R, f32>(
            client,
            f32::from_int(-1),
            f32::from_int(1),
            output.as_ref(),
        );

        output
    }
}

impl Sampleable for tf32 {
    fn sample<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R, Self> {
        cubecl_random::seed(seed);
        let output = TensorHandle::<R, tf32>::empty(client, shape.to_vec());

        cubecl_random::random_uniform::<R, f32>(
            client,
            f32::from_int(-1),
            f32::from_int(1),
            output.as_ref(),
        );

        output
    }
}

impl Sampleable for bool {
    fn sample<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R, Self> {
        cubecl_random::seed(seed);
        let output = TensorHandle::<R, bool>::empty(client, shape.to_vec());

        cubecl_random::random_bernoulli::<R, f32>(client, 0.5, output.as_ref());

        output
    }
}

pub(crate) fn flash_attention_v2_cpu<P: TestPrecision>(
    query: &[P::EG],
    key: &[P::EG],
    value: &[P::EG],
    mask: Option<&[P::EM]>,
    problem: &AttentionProblem,
) -> Vec<P::EG>
where
{
    let batch = problem.batch;
    let seq_q = problem.seq_q;
    let seq_k = problem.seq_kv;
    let num_heads = problem.num_heads;
    let head_dim = problem.head_dim;
    let val_dim = problem.val_dim;
    let masked = mask.is_some();

    // Precompute strides for indexing
    let query_strides = strides(problem, AttentionIdent::Query);
    let key_strides = strides(problem, AttentionIdent::Key);
    let value_strides = strides(problem, AttentionIdent::Value);
    let mask_strides = strides(problem, AttentionIdent::Mask);
    let out_strides = strides(problem, AttentionIdent::Out);

    let out_size = tensor_size(problem, AttentionIdent::Out);
    let mut out = vec![P::EG::from_int(0); out_size];

    // scaling factor 1/sqrt(dk)
    let scale = P::EA::new((head_dim as f32).sqrt().recip());

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                // Initialize running state for query row i
                // m = -inf, l = 0, accumulator O (unnormalized numerator) = 0
                let mut m = P::EA::new(f32::NEG_INFINITY);
                let mut l = P::EA::from_int(0);
                let mut acc_row = vec![P::EA::from_int(0); val_dim];

                // For each K/V block
                let mut k_block_start = 0usize;
                while k_block_start < seq_k {
                    let k_block_end = std::cmp::min(seq_k, k_block_start + seq_k);
                    let cur_block_len = k_block_end - k_block_start;

                    // Step A: compute S_block[j'] = Q_i · K_{j'}  for j' in block
                    // store in a small Vec<P::EA>
                    let mut s_block = vec![P::EA::from_int(0); cur_block_len];
                    for (bj, j) in (k_block_start..k_block_end).enumerate() {
                        let mut dot = P::EA::from_int(0);
                        for d in 0..head_dim {
                            let q_idx = b * query_strides[0]
                                + i * query_strides[1]
                                + h * query_strides[2]
                                + d * query_strides[3];
                            let k_idx = b * key_strides[0]
                                + j * key_strides[1]
                                + h * key_strides[2]
                                + d * key_strides[3];
                            let q_val: P::ES = query[q_idx].cast_into();
                            let k_val: P::ES = key[k_idx].cast_into();

                            dot += (q_val * k_val).cast_into();
                        }
                        // apply scale (1/sqrt(dk))
                        dot *= scale;

                        // apply mask (for masked positions set -inf)
                        let s_val = if masked {
                            let m_idx = b * mask_strides[0]
                                + i * mask_strides[1]
                                + h * mask_strides[2]
                                + j * mask_strides[3];
                            let m_val = mask.unwrap()[m_idx].cast_into();
                            if m_val != P::EM::from_int(0) {
                                P::EA::new(f32::NEG_INFINITY)
                            } else {
                                dot
                            }
                        } else {
                            dot
                        };

                        s_block[bj] = s_val;
                    }

                    // Step B: compute new row max m' = max(m, rowmax(S_block))
                    let mut block_max = P::EA::new(f32::NEG_INFINITY);
                    for &v_s in &s_block {
                        if v_s > block_max {
                            block_max = v_s;
                        }
                    }
                    // m_new
                    let mut m_new = m;
                    if block_max > m_new {
                        m_new = block_max;
                    }

                    // Step C: compute Ptilde = exp(S_block - m_new)
                    // and rowsum = sum Ptilde
                    let mut rowsum = P::EA::from_int(0);
                    let mut p_tilde = vec![P::EA::from_int(0); cur_block_len];
                    for (bj, &sval) in s_block.iter().enumerate() {
                        // if sval is -inf, exp(-inf)=0 as desired
                        let e = P::EA::exp(sval - m_new);
                        p_tilde[bj] = e;
                        rowsum += e;
                    }

                    // Step D: update running l: l_new = exp(m - m_new)*l + rowsum
                    // note: exp(prev_m - m_new) where prev_m==m
                    let epm = P::EA::exp(m - m_new);
                    let l_new = epm * l + rowsum;

                    // Step E: update numerator accumulator:
                    // acc = exp(m - m_new) * acc + Ptilde @ V_block
                    // First scale old accumulator by epm
                    for d in 0..val_dim {
                        acc_row[d] = epm * acc_row[d];
                    }
                    // Add Ptilde @ V_block
                    for (bj, j) in (k_block_start..k_block_end).enumerate() {
                        let p_val = p_tilde[bj];
                        for d in 0..val_dim {
                            let v_idx = b * value_strides[0]
                                + j * value_strides[1]
                                + h * value_strides[2]
                                + d * value_strides[3];
                            // cast v to EA so multiplication is in EA
                            let v_val: P::EA = value[v_idx].cast_into().cast_into();
                            acc_row[d] += p_val * v_val;
                        }
                    }

                    // commit updated m and l for next block
                    m = m_new;
                    l = l_new;

                    // next block
                    k_block_start += cur_block_len;
                } // end while over K/V blocks

                // Step final: normalize accumulator: O_final = acc_row / l
                // write into output
                let out_base = b * out_strides[0] + i * out_strides[1] + h * out_strides[2];

                // guard against tiny l (numerical safety)
                let eps = P::EA::new(1e-20f32);
                let denom = if l > eps { l } else { eps };
                for d in 0..val_dim {
                    let out_idx = out_base + d * out_strides[3];
                    out[out_idx] = (acc_row[d] / denom).cast_into();
                }
            }
        }
    }

    out
}
