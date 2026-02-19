use crate::prelude::*;
use crate::{self as cubecl};

macro_rules! test_binary_impl {
    (
        $test_name:ident,
        $primitive_type:tt,
        $cmp:tt,
        [$({
            vectorization: $vectorization:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
        }),*]) => {
        pub fn $test_name<R: Runtime>(client: ComputeClient<R>) {
            #[cube(launch_unchecked, fast_math = FastMath::all())]
            fn test_function(lhs: &Array<$primitive_type>, rhs: &Array<$primitive_type>, output: &mut Array<u32>) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = (lhs[ABSOLUTE_POS] $cmp rhs[ABSOLUTE_POS]) as u32;
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
                        ArrayArg::from_raw_parts::<$primitive_type>(&lhs_handle, lhs.len(), $vectorization),
                        ArrayArg::from_raw_parts::<$primitive_type>(&rhs_handle, rhs.len(), $vectorization),
                        ArrayArg::from_raw_parts::<u32>(&output_handle, $lhs.len(), $vectorization),
                    )
                };


                let actual = client.read_one_unchecked(output_handle);
                let actual = u32::from_bytes(&actual);
                for i in 0..lhs.len() {
                    let l = lhs[i];
                    let r = rhs[i];
                    let result = (l $cmp r) as u32;
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
    >,
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
    <,
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
    >=,
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
    <=,
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
    ==,
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
    !=,
    [
        {
            vectorization: 4,
            lhs: &[0, 1, u32::MAX, 42],
            rhs: &[0, 2, 0, 10],
        }
    ]
);

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
        }
    };
}
