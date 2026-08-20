use crate::prelude::*;
use crate::{self as cubecl};

macro_rules! test_binary_impl {
    (
        $test_name:ident,
        $primitive_type:tt,
        $cmp:ident,
        [$({
            vectorization: $vectorization:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
        }),*]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            #[cube(launch_unchecked, fast_math = FastMath::all())]
            fn test_function<N: Size>(
                lhs: &[Vector<$primitive_type, N>],
                rhs: &[Vector<$primitive_type, N>],
                output: &mut [Vector<u32, N>]
            ) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = Vector::cast_from(lhs[ABSOLUTE_POS].$cmp(&rhs[ABSOLUTE_POS]));
                }
            }

            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($lhs.len() * core::mem::size_of::<u32>());
                let lhs_handle = client.create_from_slice($primitive_type::as_bytes(lhs));
                let rhs_handle = client.create_from_slice($primitive_type::as_bytes(rhs));

                unsafe {
                    test_function::launch_unchecked(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d((lhs.len() / $vectorization as usize) as u32),
                        $vectorization,
                        BufferArg::from_raw_parts(lhs_handle, lhs.len()),
                        BufferArg::from_raw_parts(rhs_handle, rhs.len()),
                        BufferArg::from_raw_parts(output_handle.clone(), $lhs.len()),
                    )
                };


                let actual = client.read_one_unchecked(output_handle);
                let actual = u32::from_bytes(&actual);
                for i in 0..lhs.len() {
                    let l = lhs[i];
                    let r = rhs[i];
                    let result = (l.$cmp(&r)) as u32;
                    assert!(actual[i] == result, "{} {} should give {} but gave {}", l, r, result, actual[i]);
                }
            }
            )*
        }
    };
}

// 00001100

test_binary_impl!(
    test_gt,
    u32,
    gt,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

test_binary_impl!(
    test_lt,
    u32,
    lt,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

test_binary_impl!(
    test_ge,
    u32,
    ge,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

test_binary_impl!(
    test_le,
    u32,
    le,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

test_binary_impl!(
    test_eq,
    u32,
    eq,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

test_binary_impl!(
    test_ne,
    u32,
    ne,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

/// NaN comparison semantics: `<`, `<=`, `>`, `>=` and `==` are *ordered* (false whenever an
pub fn test_nan_ordering<R: Runtime>(client: ComputeClient<R>) {
    #[cube(launch_unchecked)]
    fn test_function(lhs: &[f32], rhs: &[f32], output: &mut [u32]) {
        if ABSOLUTE_POS < lhs.len() {
            let l = lhs[ABSOLUTE_POS];
            let r = rhs[ABSOLUTE_POS];
            let mut bits = 0u32;
            if l < r {
                bits += 1;
            }
            if l <= r {
                bits += 2;
            }
            if l > r {
                bits += 4;
            }
            if l >= r {
                bits += 8;
            }
            if l == r {
                bits += 16;
            }
            output[ABSOLUTE_POS] = bits;
        }
    }

    let nan = f32::NAN;
    let lhs: &[f32] = &[1.0, nan, nan, 1.0, 2.0];
    let rhs: &[f32] = &[nan, 1.0, nan, 2.0, 1.0];

    let output_handle = client.empty(lhs.len() * core::mem::size_of::<u32>());
    let lhs_handle = client.create_from_slice(f32::as_bytes(lhs));
    let rhs_handle = client.create_from_slice(f32::as_bytes(rhs));

    unsafe {
        test_function::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(lhs.len() as u32),
            BufferArg::from_raw_parts(lhs_handle, lhs.len()),
            BufferArg::from_raw_parts(rhs_handle, rhs.len()),
            BufferArg::from_raw_parts(output_handle.clone(), lhs.len()),
        )
    };

    let actual = client.read_one_unchecked(output_handle);
    let actual = u32::from_bytes(&actual);

    for i in 0..lhs.len() {
        let (l, r) = (lhs[i], rhs[i]);
        let expected = (l < r) as u32
            + ((l <= r) as u32) * 2
            + ((l > r) as u32) * 4
            + ((l >= r) as u32) * 8
            + ((l == r) as u32) * 16;
        assert_eq!(
            actual[i], expected,
            "comparing {l} with {r}: expected bits {expected:05b}, got {:05b} \
             (bit order: <, <=, >, >=, ==)",
            actual[i]
        );
    }
}

/// Comparing a vector with itself folds at compile time, and the answer is a vector of `true`,
/// not the one `true` a bare [`BoolAttr`](cubecl_ir::attributes::BoolAttr) can hold. Each of these
/// picks a different fold: two identical operands, and an operand against its type's extreme.
#[cube(launch_unchecked)]
fn kernel_folded(output: &mut [Vector<u32, Const<4>>]) {
    if ABSOLUTE_POS < output.len() {
        let value = output[ABSOLUTE_POS];

        let same = value.equal(&value);
        let below_min = value.less_than(&Vector::new(u32::MIN));

        output[ABSOLUTE_POS] =
            Vector::cast_from(same) + Vector::cast_from(below_min) * Vector::new(2u32);
    }
}

/// A vector comparison that constant folds still answers per lane.
pub fn test_folded_vector<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(u32::as_bytes(&[7u32, 0, 3, 0]));

    unsafe {
        kernel_folded::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(handle.clone(), 4),
        )
    };

    let actual = client.read_one_unchecked(handle);
    assert_eq!(
        actual.len() / size_of::<u32>(),
        4,
        "a failed launch reads back nothing"
    );
    // Equal to itself everywhere, below `u32::MIN` nowhere.
    assert_eq!(u32::from_bytes(&actual), &[1, 1, 1, 1]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_comparison {
    () => {
        mod comparison {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[$crate::runtime_tests::test_log::test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::comparison::$test_name::<TestRuntime>(client);
                    }
                };
            }

            add_test!(test_gt);
            add_test!(test_lt);
            add_test!(test_ge);
            add_test!(test_le);
            add_test!(test_eq);
            add_test!(test_ne);
            add_test!(test_nan_ordering);
            add_test!(test_folded_vector);
        }
    };
}
