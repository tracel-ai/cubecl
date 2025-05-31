use std::fmt::Display;

use cubecl_core::{
    CubeElement, Feature, Runtime,
    client::ComputeClient,
    flex32,
    prelude::{CubePrimitive, Float, Numeric},
    server::{self},
    tf32,
};

pub use cubecl_std::SymQ8;

use crate::{
    matmul::{
        components::{Ident, MatmulPrecision, MatmulProblem},
        tests::cmma_matmul::matmul_test_launcher::strides,
    },
    tensor::TensorHandle,
};

pub struct QuantizationParams<N: Numeric> {
    pub scaling: Vec<N>, // This is the bit cast of an f32 into the appropriate numbers of N.
    pub zero_offset: N,
}

pub trait TestPrecision {
    type EG: Numeric + CubeElement + Display + CastInto<Self::ES> + Sample;
    type ES: Numeric + Display + CastInto<Self::EA>;
    type EA: Numeric + Display + CastInto<Self::EG>;
    type MP: MatmulPrecision;
    const QUANTIZED: bool;

    fn quantization_params(ident: Ident) -> Option<QuantizationParams<Self::EG>>;

    #[allow(clippy::too_many_arguments)]
    fn assert_result<R: Runtime>(
        lhs: &[Self::EG],
        lhs_quant: Option<(f32, i32)>,
        rhs: &[Self::EG],
        rhs_quant: Option<(f32, i32)>,
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        out_quant: Option<(f32, i32)>,
        shape: &[usize],
        strides: &[usize],
    );
}

impl<EG, ES> TestPrecision for (EG, ES)
where
    EG: Float + CubeElement + Display + CastInto<ES> + Sample + MatmulPrecision,
    ES: Numeric + Display + CastInto<f32>,
    f32: CastInto<EG>,
{
    type EG = EG;
    type ES = ES;
    type EA = f32;
    type MP = EG;
    const QUANTIZED: bool = false;

    fn quantization_params(_: Ident) -> Option<QuantizationParams<Self::EG>> {
        None
    }

    fn assert_result<R: Runtime>(
        lhs: &[EG],
        _lhs_quant: Option<(f32, i32)>,
        rhs: &[EG],
        _rhs_quant: Option<(f32, i32)>,
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        _out_quant: Option<(f32, i32)>,
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

        let expected = matmul_cpu_reference::<Self>(lhs, rhs, problem)
            .into_iter()
            .map(|x| x.cast_into())
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

// TODO:
//   - Add different conversions from i32 to u8.
//   - Fix with proper types for precision
impl TestPrecision for SymQ8 {
    type EG = u8;
    type ES = u16;
    type EA = i32;
    type MP = SymQ8;

    const QUANTIZED: bool = true;

    fn quantization_params(ident: Ident) -> Option<QuantizationParams<Self::EG>> {
        // These are somewhat arbitrary values. I try to use f32 that are exactly representable
        // to avoid some rounding issues.
        Some(match ident {
            Ident::Lhs => QuantizationParams {
                scaling: f32::to_be_bytes(0.6).to_vec(),
                zero_offset: 15,
            },
            Ident::Rhs => QuantizationParams {
                scaling: f32::to_be_bytes(0.1).to_vec(),
                zero_offset: 20,
            },
            Ident::Out => QuantizationParams {
                scaling: f32::to_be_bytes(0.4).to_vec(),
                zero_offset: 50,
            },
        })
    }

    fn assert_result<R: Runtime>(
        lhs: &[u8],
        lhs_quant: Option<(f32, i32)>,
        rhs: &[u8],
        rhs_quant: Option<(f32, i32)>,
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        out_quant: Option<(f32, i32)>,
        shape: &[usize],
        strides: &[usize],
    ) {
        let out = client.read_one_tensor(out.binding_with_meta(
            shape.to_vec(),
            strides.to_vec(),
            size_of::<Self::EG>(),
        ));
        let out = u8::from_bytes(&out);

        let (lhs_scaling, lhs_offset) = lhs_quant.unwrap();
        let (rhs_scaling, rhs_offset) = rhs_quant.unwrap();
        let (out_scaling, out_offset) = out_quant.unwrap();

        // TODO Move to some better place and wrap into a function.
        let scaling_factor = (lhs_scaling * rhs_scaling) / out_scaling;
        let approx_scaling = ApproxScaling::from_f32(scaling_factor);

        let expected = matmul_cpu_reference_quantized(
            lhs,
            rhs,
            problem,
            lhs_offset,
            rhs_offset,
            out_offset,
            approx_scaling,
        );
        assert_eq!(out, expected);
    }
}

struct ApproxScaling {
    multiplier: i64,
    rounding: i64,
    shift: u32,
}

impl ApproxScaling {
    fn from_f32(x: f32) -> Self {
        let log = x.log2().ceil() as i32;
        let multiplier = (x * 2.0_f32.powi(31 - log)).round() as i64;
        let rounding: i64 = 1 << (30 - log as i64);
        let shift = (31 - log) as u32;
        Self {
            multiplier,
            rounding,
            shift,
        }
    }

    fn scale(&self, x: i32) -> i32 {
        if self.multiplier == i32::MIN as i64 && x == i32::MIN {
            return i32::MAX; // sature on overflow. (while multiplier is in the range of an i32 even if it is a i64)
        }
        let prod = (x as i64) * self.multiplier;
        let prod_with_rounding = prod + self.rounding;
        (prod_with_rounding >> self.shift) as i32
    }
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

pub trait Sample: Sized + CubePrimitive {
    fn sample<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        seed: u64,
    ) -> TensorHandle<R, Self>;
}

macro_rules! sample_float {
    ($($t:ty),*) => {
        $(
            impl Sample for $t
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

impl Sample for flex32 {
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

impl Sample for tf32 {
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

/// Solves a matmul problem with EG inputs, multiplied as ES and accumulated as EA.
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn matmul_cpu_reference<P: TestPrecision>(
    lhs: &[P::EG],
    rhs: &[P::EG],
    problem: &MatmulProblem,
) -> Vec<P::EA>
where
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let num_batches = problem.num_batches();

    let (b_lhs, b_rhs) = problem.batches.clone();
    assert!(
        b_lhs.len() == b_rhs.len(),
        "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning."
    );

    let lhs_strides = strides(problem, Ident::Lhs);
    let rhs_strides = strides(problem, Ident::Rhs);
    let out_strides = strides(problem, Ident::Out);

    let mut out = vec![P::EA::from_int(0); m * n * num_batches];

    for nth_batch in 0..num_batches {
        let batch_out = nth_batch * m * n;
        let mut batch_lhs = 0;
        let mut batch_rhs = 0;
        for b in 0..b_lhs.len() {
            let tmp = batch_out / out_strides[b];
            batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
            batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
        }

        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    let lhs_index = i * k + k_;
                    let rhs_index = k_ * n + j;
                    let out_index = i * n + j;

                    let l: P::ES = lhs[batch_lhs + lhs_index].cast_into();
                    let r: P::ES = rhs[batch_rhs + rhs_index].cast_into();
                    let prod = l * r;

                    out[batch_out + out_index] += prod.cast_into();
                }
            }
        }
    }

    out
}

fn matmul_cpu_reference_quantized(
    lhs: &[u8],
    rhs: &[u8],
    problem: &MatmulProblem,
    lhs_zero_offset: i32,
    rhs_zero_offset: i32,
    out_zero_offset: i32,
    approx_scaling: ApproxScaling,
) -> Vec<u8>
where
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let num_batches = problem.num_batches();

    let (b_lhs, b_rhs) = problem.batches.clone();
    assert!(
        b_lhs.len() == b_rhs.len(),
        "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning."
    );

    let lhs_strides = strides(problem, Ident::Lhs);
    let rhs_strides = strides(problem, Ident::Rhs);
    let out_strides = strides(problem, Ident::Out);

    let mut out = vec![0; m * n * num_batches];

    for nth_batch in 0..num_batches {
        let batch_out = nth_batch * m * n;
        let mut batch_lhs = 0;
        let mut batch_rhs = 0;
        for b in 0..b_lhs.len() {
            let tmp = batch_out / out_strides[b];
            batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
            batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
        }

        // Perform matmul
        for row in 0..m {
            for col in 0..n {
                let mut elem = 0;
                for middle in 0..k {
                    let lhs_index = row * k + middle;
                    let rhs_index = middle * n + col;

                    let l = lhs[batch_lhs + lhs_index] as i32 - lhs_zero_offset;
                    let r = rhs[batch_rhs + rhs_index] as i32 - rhs_zero_offset;
                    let prod = l * r;
                    elem += prod;
                }
                elem = approx_scaling.scale(elem);
                elem += out_zero_offset;
                let out_index = row * n + col;
                out[batch_out + out_index] = if elem < 0 {
                    0
                } else if elem > 255 {
                    255
                } else {
                    elem as u8
                };
            }
        }
    }

    out
}

/// Deprecated
pub(crate) struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

/// Deprecated
impl MatmulTestCase {
    pub(crate) fn matmul_cpu<R: Runtime, F: Float + CubeElement>(
        &self,
        lhs: &TensorHandle<R, F>,
        rhs: &TensorHandle<R, F>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Vec<F> {
        let lhs_binding = &client.read_one_tensor(lhs.handle.clone().binding_with_meta(
            lhs.shape.clone(),
            lhs.strides.clone(),
            size_of::<F>(),
        ));
        let rhs_binding = &client.read_one_tensor(rhs.handle.clone().binding_with_meta(
            rhs.shape.clone(),
            rhs.strides.clone(),
            size_of::<F>(),
        ));

        let lhs = F::from_bytes(lhs_binding);
        let rhs = F::from_bytes(rhs_binding);

        self.matmul_cpu_algorithm(lhs, rhs)
    }

    fn matmul_cpu_algorithm<F: Float + CubeElement>(&self, lhs: &[F], rhs: &[F]) -> Vec<F> {
        let mut out = vec![F::from_int(0); self.batch * self.m * self.n];
        let lhs_batch_offset = self.m * self.k;
        let rhs_batch_offset = self.k * self.n;
        let out_batch_offset = self.m * self.n;

        for b in 0..self.batch {
            for i in 0..self.m {
                for j in 0..self.n {
                    for k_ in 0..self.k {
                        let lhs_value = lhs[b * lhs_batch_offset + i * self.k + k_];
                        let rhs_value = rhs[b * rhs_batch_offset + j + k_ * self.n];

                        let result = lhs_value * rhs_value;

                        out[b * out_batch_offset + i * self.n + j] += result;
                    }
                }
            }
        }

        out
    }

    pub(crate) fn random_lhs<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.m, self.k])
    }

    pub(crate) fn random_rhs<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.k, self.n])
    }

    pub(crate) fn empty_out<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        TensorHandle::empty(client, vec![self.batch, self.m, self.n])
    }

    pub(crate) fn random_tensor<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        shape: Vec<usize>,
    ) -> TensorHandle<R, F> {
        F::sample::<R>(client, &shape, 999)
    }
}
