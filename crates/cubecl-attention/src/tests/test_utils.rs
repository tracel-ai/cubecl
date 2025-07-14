use std::fmt::Display;

use cubecl_core::{
    CubeElement, Feature, Runtime,
    client::ComputeClient,
    flex32,
    prelude::{CubePrimitive, Exp, Float, Numeric},
    server::{self},
    tf32,
};

use cubecl_std::tensor::TensorHandle;

use crate::{
    components::{AttentionPrecision, AttentionProblem, Ident},
    tests::attention_test_launcher::{strides, tensor_size},
};

pub trait TestPrecision {
    type EG: Float + CubeElement + Display + CastInto<Self::ES> + Sampleable;
    type ES: Float + Display + CastInto<Self::EA>;
    type EA: Float + Display + CastInto<Self::EG> + Exp;
    type EM: Numeric + CubeElement + Display + Sampleable;
    type MP: AttentionPrecision;

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
    type MP = EG;

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
        let maybe_f16 = client.properties().feature_enabled(Feature::Cmma {
            a: ES::as_elem_native().expect("To be a native type"),
            b: ES::as_elem_native().expect("To be a native type"),
            c: EG::as_elem_native().expect("To be a native type"),
            m: 16,
            k: 16,
            n: 16,
        });
        let maybe_tf32 = client.properties().feature_enabled(Feature::Cmma {
            a: ES::as_elem_native().expect("To be a native type"),
            b: ES::as_elem_native().expect("To be a native type"),
            c: EG::as_elem_native().expect("To be a native type"),
            m: 16,
            k: 8,
            n: 16,
        });

        // Need to compensate for the temporary conversion to f16/tf32
        let epsilon = match maybe_f16 || maybe_tf32 {
            true => 10e-5 / EG::EPSILON.to_f32().unwrap() * half::f16::EPSILON.to_f32(),
            false => 10e-5,
        };

        let expected = attention_cpu_reference::<Self>(query, key, value, mask, problem)
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
    let actual = client.read_one_tensor(output.binding_with_meta(
        shape.to_vec(),
        strides.to_vec(),
        size_of::<F>(),
    ));
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
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

                    cubecl_random::random_uniform::<R, Self>(&client, Self::from_int(-1), Self::from_int(1), output.as_ref());

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

pub(crate) fn attention_cpu_reference<P: TestPrecision>(
    query: &[P::EG],
    key: &[P::EG],
    value: &[P::EG],
    mask: Option<&[P::EM]>, // optional mask, same shape as mask strides if present
    problem: &AttentionProblem,
) -> Vec<P::EG>
where
{
    let batch = problem.batch;
    let seq_q = problem.seq_q;
    let seq_k = problem.seq_k;
    let num_heads = problem.num_heads;
    let head_dim = problem.head_dim;
    let masked = mask.is_some();

    // Precompute strides for indexing
    let query_strides = strides(problem, Ident::Query);
    let key_strides = strides(problem, Ident::Key);
    let value_strides = strides(problem, Ident::Value);
    let mask_strides = strides(problem, Ident::Mask);
    let out_strides = strides(problem, Ident::Out);

    let out_size = tensor_size(problem, Ident::Out);
    let mut out = vec![P::EG::from_int(0); out_size];

    // Constant for scaling
    let scale = P::EA::new((head_dim as f32).sqrt().recip());

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_q {
                // Step 1: Compute attention scores (A[i,j]) for all j in seq_k
                let mut scores = vec![P::EA::from_int(0); seq_k];
                for j in 0..seq_k {
                    // Dot product Q[b,i,h,:] · K[b,j,h,:]
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
                    dot *= scale;

                    // Apply mask if present
                    scores[j] = if masked {
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
                }

                // Step 2: Compute softmax over scores
                let mut max_score = P::EA::new(f32::NEG_INFINITY);
                for i in 0..seq_k {
                    let val = scores[i];
                    if val > max_score {
                        max_score = val;
                    }
                }
                let mut sum_exp = P::EA::from_int(0);
                let mut exp_scores = vec![P::EA::from_int(0); seq_k];
                for j in 0..seq_k {
                    let e = P::EA::exp(scores[j] - max_score);
                    exp_scores[j] = e;
                    sum_exp += e;
                }

                // Step 3: Compute context vector: sum_j softmax_j * V[b,j,h,:]
                for d in 0..head_dim {
                    let mut ctx = P::EA::from_int(0);
                    for j in 0..seq_k {
                        let v_idx = b * value_strides[0]
                            + j * value_strides[1]
                            + h * value_strides[2]
                            + d * value_strides[3];
                        let v_val = value[v_idx].cast_into().cast_into();
                        let softmax_j = exp_scores[j] / sum_exp;
                        ctx += softmax_j * v_val;
                    }
                    let out_idx = b * out_strides[0]
                        + i * out_strides[1]
                        + h * out_strides[2]
                        + d * out_strides[3];
                    out[out_idx] = ctx.cast_into();
                }
            }
        }
    }

    out
}
