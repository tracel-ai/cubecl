//! Accuracy of the f64 transcendental functions, in ulp against the host's libm.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_ir::features::TypeUsage;
use cubecl_runtime::runtime::Runtime;

#[derive(CubeType, Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum F64Function {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Exp,
    Ln,
    Powf,
}

impl F64Function {
    /// The host reference.
    fn reference(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Asin => x.asin(),
            Self::Acos => x.acos(),
            Self::Atan => x.atan(),
            Self::Atan2 => x.atan2(y),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            // SAFETY: the C math functions take and return plain doubles.
            Self::Asinh => unsafe { c_math::asinh(x) },
            Self::Acosh => unsafe { c_math::acosh(x) },
            Self::Atanh => unsafe { c_math::atanh(x) },
            Self::Exp => x.exp(),
            Self::Ln => x.ln(),
            Self::Powf => x.powf(y),
        }
    }

    /// The bound in ulp: the bound of the emulation plus the host's own error.
    fn max_ulp(self, x: f64) -> u64 {
        match self {
            // Above 1e14 the reduction carries about 106 bits. An argument within 2^-61 of a
            // multiple of pi/2, the closest any f64 comes, loses up to 61 of them to cancellation.
            Self::Sin | Self::Cos | Self::Tan if x.abs() >= 1e14 => 16,
            Self::Exp | Self::Powf | Self::Asinh | Self::Acosh | Self::Atanh => 2,
            _ => 4,
        }
    }
}

/// Rust's `asinh`, `acosh` and `atanh` are formulas that lose accuracy near 1 and overflow for a
/// large argument, so the C library's functions are the reference.
mod c_math {
    unsafe extern "C" {
        pub fn asinh(x: f64) -> f64;
        pub fn acosh(x: f64) -> f64;
        pub fn atanh(x: f64) -> f64;
    }
}

#[cube(launch_unchecked)]
pub fn kernel_f64_function<N: Size>(
    lhs: &[Vector<f64, N>],
    rhs: &[Vector<f64, N>],
    output: &mut [Vector<f64, N>],
    #[comptime] function: F64Function,
) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let x = lhs[i];
        let y = rhs[i];
        match function {
            F64Function::Sin => output[i] = x.sin(),
            F64Function::Cos => output[i] = x.cos(),
            F64Function::Tan => output[i] = x.tan(),
            F64Function::Asin => output[i] = x.asin(),
            F64Function::Acos => output[i] = x.acos(),
            F64Function::Atan => output[i] = x.atan(),
            F64Function::Atan2 => output[i] = x.atan2(y),
            F64Function::Sinh => output[i] = x.sinh(),
            F64Function::Cosh => output[i] = x.cosh(),
            F64Function::Tanh => output[i] = x.tanh(),
            F64Function::Asinh => output[i] = x.asinh(),
            F64Function::Acosh => output[i] = x.acosh(),
            F64Function::Atanh => output[i] = x.atanh(),
            F64Function::Exp => output[i] = x.exp(),
            F64Function::Ln => output[i] = x.ln(),
            F64Function::Powf => output[i] = x.powf(y),
        }
    }
}

const SPECIAL: [f64; 14] = [
    0.0,
    -0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    5e-324,
    -5e-324,
    f64::MAX,
    f64::MIN,
    1.0,
    -1.0,
    0.5,
];

/// `count` points spread evenly over `[from, to]`.
fn linear(from: f64, to: f64, count: u32) -> impl Iterator<Item = f64> {
    (0..count).map(move |i| from + (to - from) * f64::from(i) / f64::from(count - 1))
}

/// `count` points spread evenly in magnitude over `[from, to]`, with both signs.
fn logarithmic(from: f64, to: f64, count: u32) -> impl Iterator<Item = f64> {
    let (from, to) = (from.ln(), to.ln());
    linear(from, to, count).flat_map(|e| [e.exp(), -e.exp()])
}

fn arguments(function: F64Function) -> Vec<(f64, f64)> {
    let unary = |values: Vec<f64>| values.into_iter().map(|x| (x, 1.0)).collect();
    let mut values: Vec<f64> = SPECIAL.to_vec();
    match function {
        F64Function::Sin | F64Function::Cos | F64Function::Tan => {
            values.extend(linear(-10.0, 10.0, 2001));
            values.extend(logarithmic(1e-300, 1e300, 1000));
            // The edges of each range reduction.
            values.extend([14.999, 15.0, 15.001, 1e6, 1e14, 1.0001e14, 1e22, 2e300]);
            values.extend([f64::MAX / 3.0, 6_381_956_970_095_103.0 * 2f64.powi(797)]);
            unary(values)
        }
        F64Function::Asin | F64Function::Acos | F64Function::Atanh => {
            values.extend(linear(-1.0, 1.0, 2001));
            values.extend(logarithmic(1e-300, 1.0, 400));
            values.extend([0.999_999_999_999, -0.999_999_999_999, 1.000_000_1, -2.0]);
            unary(values)
        }
        F64Function::Atan | F64Function::Asinh => {
            values.extend(linear(-10.0, 10.0, 2001));
            values.extend(logarithmic(1e-300, 1e300, 1000));
            unary(values)
        }
        F64Function::Acosh => {
            values.extend(linear(1.0, 10.0, 1001));
            values.extend(logarithmic(1.0, 1e300, 1000));
            values.extend([1.000_000_000_001, 0.999, -2.0]);
            unary(values)
        }
        F64Function::Sinh | F64Function::Cosh => {
            values.extend(linear(-720.0, 720.0, 2001));
            values.extend(logarithmic(1e-300, 1.0, 400));
            values.extend([709.0, 709.78, 710.5, -709.78]);
            unary(values)
        }
        F64Function::Tanh => {
            values.extend(linear(-30.0, 30.0, 2001));
            values.extend(logarithmic(1e-300, 1.0, 400));
            values.extend([18.714_973_875, 18.8, -19.0]);
            unary(values)
        }
        F64Function::Exp => {
            values.extend(linear(-760.0, 720.0, 4001));
            values.extend(logarithmic(1e-300, 1.0, 400));
            values.extend([
                709.78, 709.79, -708.4, -744.0, -745.13, -745.2, -1000.0, -1001.0,
            ]);
            unary(values)
        }
        F64Function::Ln => {
            values.extend(linear(0.5, 2.0, 2001));
            values.extend(logarithmic(5e-324, f64::MAX, 2000));
            values.extend([1e-310, 2.5e-320]);
            unary(values)
        }
        F64Function::Atan2 | F64Function::Powf => {
            let mut grid: Vec<f64> = SPECIAL.to_vec();
            grid.extend([
                2.0, -2.0, 3.0, -3.0, -0.5, 0.3, -2.5, 3.7, -3.7, 1e10, 1e-300, 1e300,
            ]);
            let mut pairs: Vec<(f64, f64)> = grid
                .iter()
                .flat_map(|&x| grid.iter().map(move |&y| (x, y)))
                .collect();
            let lhs = if function == F64Function::Powf {
                linear(0.01, 100.0, 61).collect::<Vec<_>>()
            } else {
                linear(-100.0, 100.0, 61).collect()
            };
            let rhs = linear(-50.0, 50.0, 61).collect::<Vec<_>>();
            pairs.extend(lhs.iter().flat_map(|&x| rhs.iter().map(move |&y| (x, y))));
            pairs.extend([
                (-8.0, 1.0 / 3.0),
                (2.0, 1023.5),
                (2.0, -1074.0),
                (0.5, 1075.0),
            ]);
            pairs
        }
    }
}

/// The distance in representable f64 values.
fn ulp_distance(a: f64, b: f64) -> u64 {
    let ordered = |x: f64| {
        let bits = x.to_bits().cast_signed();
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    ordered(a).abs_diff(ordered(b))
}

#[allow(
    clippy::float_cmp,
    reason = "an infinity or a zero must come out exactly, with its sign"
)]
fn check(
    function: F64Function,
    (x, y): (f64, f64),
    actual: f64,
) -> Result<(), alloc::string::String> {
    let expected = function.reference(x, y);
    let ok = if expected.is_nan() {
        actual.is_nan()
    } else if expected.is_infinite() || expected == 0.0 {
        actual == expected && actual.is_sign_negative() == expected.is_sign_negative()
    } else {
        ulp_distance(actual, expected) <= function.max_ulp(x)
    };
    if ok {
        Ok(())
    } else {
        Err(alloc::format!(
            "{function:?}({x:e}, {y:e}) = {actual:e}, expected {expected:e} ({} ulp)",
            ulp_distance(actual, expected)
        ))
    }
}

/// # Panics
///
/// When a result is further from the host's than the function's bound.
pub fn test_f64_function<R: Runtime>(client: &Client, function: F64Function, vector_size: usize) {
    if !f64::supported_uses(client).contains(TypeUsage::Arithmetic) {
        return;
    }

    let mut pairs = arguments(function);
    while !pairs.len().is_multiple_of(vector_size) {
        pairs.push((1.0, 1.0));
    }
    let lhs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let rhs: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let len = pairs.len();

    let lhs_handle = client.create_from_slice(f64::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(f64::as_bytes(&rhs));
    let out_handle = client.empty(len * size_of::<f64>());

    let units = len / vector_size;
    let cube_dim = CubeDim::new_1d(64);
    unsafe {
        kernel_f64_function::launch_unchecked(
            client,
            CubeCount::Static(u32::try_from(units.div_ceil(64)).unwrap(), 1, 1),
            cube_dim,
            vector_size,
            BufferArg::from_raw_parts(lhs_handle, len),
            BufferArg::from_raw_parts(rhs_handle, len),
            BufferArg::from_raw_parts(out_handle.clone(), len),
            function,
        );
    }

    let actual = client.read_one_unchecked(out_handle);
    let actual = f64::from_bytes(&actual);

    let failures: Vec<_> = pairs
        .iter()
        .zip(actual)
        .filter_map(|(&pair, &actual)| check(function, pair, actual).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {len} values are off:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_f64_math {
    () => {
        $crate::testgen_f64_math!(@functions
            Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Exp,
            Ln, Powf
        );
    };
    (@functions $($function:ident),*) => {
        ::paste::paste! {
            $(
                #[$crate::runtime_tests::test_log::test]
                fn [<test_f64_ $function:snake>]() {
                    use $crate::runtime_tests::f64_math::{F64Function, test_f64_function};

                    let client = TestRuntime::client(&Default::default());
                    test_f64_function::<TestRuntime>(&client, F64Function::$function, 1);
                    test_f64_function::<TestRuntime>(&client, F64Function::$function, 2);
                }
            )*
        }
    };
}
