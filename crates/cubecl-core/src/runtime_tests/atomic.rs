use std::{println, vec::Vec};

use crate::{self as cubecl};

use cubecl::prelude::*;
use cubecl_ir::features::AtomicUsage;

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum NumericAtomicOp {
    Load,
    Store,
    Swap,
    Add,
    Sub,
    Min,
    Max,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum IntAtomicOp {
    And,
    Or,
    Xor,
    CompareExchange,
}

fn supports_feature<R: Runtime, F: Numeric>(
    client: &ComputeClient<R>,
    feat: AtomicUsage,
    vector_size: usize,
) -> bool {
    let ty = Type::atomic(F::as_type_native_unchecked().with_vector_size(vector_size));
    client.properties().atomic_type_usage(ty).contains(feat)
}

fn require_feature<R: Runtime, F: Numeric>(
    client: &ComputeClient<R>,
    feat: AtomicUsage,
    vector_size: usize,
    operation: &str,
) -> bool {
    let ty = Atomic::<F>::as_type_native_unchecked().with_vector_size(vector_size);

    if supports_feature::<R, F>(client, feat, vector_size) {
        println!("{ty} {operation} supported - running");
        true
    } else {
        println!("{ty} {operation} not supported - skipped");
        false
    }
}

fn filled_data<F: Numeric>(vector_size: usize, value: i64) -> Vec<F> {
    (0..vector_size)
        .map(|_| F::from_int(value))
        .collect::<Vec<_>>()
}

fn int_atomic_initial_value() -> i64 {
    0b1010
}

fn assert_all_eq<F: Numeric + CubeElement>(actual: &[F], expected: i64) {
    assert!(actual.iter().all(|actual| actual == &F::from_int(expected)));
}

fn numeric_op_name(op: NumericAtomicOp) -> &'static str {
    match op {
        NumericAtomicOp::Load => "Load",
        NumericAtomicOp::Store => "Store",
        NumericAtomicOp::Swap => "Swap",
        NumericAtomicOp::Add => "Add",
        NumericAtomicOp::Sub => "Sub",
        NumericAtomicOp::Min => "Min",
        NumericAtomicOp::Max => "Max",
    }
}

fn numeric_op_feature(op: NumericAtomicOp) -> AtomicUsage {
    match op {
        NumericAtomicOp::Load | NumericAtomicOp::Store => AtomicUsage::LoadStore,
        NumericAtomicOp::Swap => AtomicUsage::Swap,
        NumericAtomicOp::Add | NumericAtomicOp::Sub => AtomicUsage::Add,
        NumericAtomicOp::Min | NumericAtomicOp::Max => AtomicUsage::MinMax,
    }
}

fn numeric_expected(op: NumericAtomicOp) -> i64 {
    match op {
        NumericAtomicOp::Load => 12,
        NumericAtomicOp::Store => 5,
        NumericAtomicOp::Swap => 5,
        NumericAtomicOp::Add => 17,
        NumericAtomicOp::Sub => 7,
        NumericAtomicOp::Min => 5,
        NumericAtomicOp::Max => 12,
    }
}

fn int_op_name(op: IntAtomicOp) -> &'static str {
    match op {
        IntAtomicOp::And => "And",
        IntAtomicOp::Or => "Or",
        IntAtomicOp::Xor => "Xor",
        IntAtomicOp::CompareExchange => "CompareExchange",
    }
}

fn int_op_feature(op: IntAtomicOp) -> AtomicUsage {
    match op {
        IntAtomicOp::And | IntAtomicOp::Or | IntAtomicOp::Xor => AtomicUsage::Bitwise,
        IntAtomicOp::CompareExchange => AtomicUsage::CompareExchange,
    }
}

fn int_expected<I: Int>(op: IntAtomicOp) -> I {
    match op {
        IntAtomicOp::And => I::from_int(0b0010),
        IntAtomicOp::Or => I::from_int(0b1110),
        IntAtomicOp::Xor => I::from_int(0b1100),
        IntAtomicOp::CompareExchange => I::from_int(int_atomic_initial_value()),
    }
}

#[cube(launch)]
pub fn kernel_atomic_numeric<I: Numeric, N: Size>(
    input: &[Vector<I, N>],
    atomics: &[Atomic<Vector<I, N>>],
    output: &mut [Vector<I, N>],
    #[comptime] op: NumericAtomicOp,
) {
    if UNIT_POS == 0 {
        match op {
            NumericAtomicOp::Load => output[0] = atomics[0].load(),
            NumericAtomicOp::Store => atomics[0].store(input[0]),
            NumericAtomicOp::Swap => {
                atomics[0].swap(Vector::from_int(5));
            }
            NumericAtomicOp::Add => {
                atomics[0].fetch_add(Vector::from_int(5));
            }
            NumericAtomicOp::Sub => {
                atomics[0].fetch_sub(Vector::from_int(5));
            }
            NumericAtomicOp::Min => {
                atomics[0].fetch_min(Vector::from_int(5));
            }
            NumericAtomicOp::Max => {
                atomics[0].fetch_max(Vector::from_int(5));
            }
        }
    }
}

pub fn test_kernel_atomic_numeric<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R>,
    vector_size: usize,
    op: NumericAtomicOp,
) {
    if !require_feature::<R, F>(
        &client,
        numeric_op_feature(op),
        vector_size,
        numeric_op_name(op),
    ) {
        return;
    }

    let input_handle = client.create_from_slice(F::as_bytes(&filled_data::<F>(vector_size, 5)));
    let atomic_handle = client.create_from_slice(F::as_bytes(&filled_data::<F>(vector_size, 12)));
    let output_handle = client.create_from_slice(F::as_bytes(&filled_data::<F>(vector_size, 0)));

    kernel_atomic_numeric::launch::<F, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        vector_size,
        unsafe { BufferArg::from_raw_parts(input_handle, 1) },
        unsafe { BufferArg::from_raw_parts(atomic_handle.clone(), vector_size) },
        unsafe { BufferArg::from_raw_parts(output_handle.clone(), 1) },
        op,
    );

    let actual = match op {
        NumericAtomicOp::Load => client.read_one_unchecked(output_handle),
        _ => client.read_one_unchecked(atomic_handle),
    };
    let actual = F::from_bytes(&actual);

    assert_all_eq(&actual, numeric_expected(op));
}

pub fn test_kernel_atomic_numeric_all_sizes<R: Runtime, F: Numeric + CubeElement>(
    client: ComputeClient<R>,
    op: NumericAtomicOp,
) {
    for vector_size in [1, 2, 4] {
        test_kernel_atomic_numeric::<R, F>(client.clone(), vector_size, op);
    }
}

#[cube(launch)]
pub fn kernel_atomic_int<I: Int>(atomics: &[Atomic<I>], #[comptime] op: IntAtomicOp) {
    if UNIT_POS == 0 {
        match op {
            IntAtomicOp::And => {
                atomics[0].fetch_and(I::from_int(0b0110));
            }
            IntAtomicOp::Or => {
                atomics[0].fetch_or(I::from_int(0b0110));
            }
            IntAtomicOp::Xor => {
                atomics[0].fetch_xor(I::from_int(0b0110));
            }
            IntAtomicOp::CompareExchange => {
                atomics[0].compare_exchange_weak(I::from_int(12), I::from_int(5));
            }
        }
    }
}

pub fn test_kernel_atomic_int<R: Runtime, I: Int + CubeElement>(
    client: ComputeClient<R>,
    op: IntAtomicOp,
) {
    if !require_feature::<R, I>(&client, int_op_feature(op), 1, int_op_name(op)) {
        return;
    }

    let handle = client.create_from_slice(I::as_bytes(&[I::from_int(int_atomic_initial_value())]));

    kernel_atomic_int::launch::<I, R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        op,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = I::from_bytes(&actual);

    assert_eq!(actual, &[int_expected::<I>(op)]);
}

#[macro_export]
macro_rules! testgen_atomic_int {
    () => {
        use super::*;

        macro_rules! test_numeric_op {
            ($name:ident, $op:expr) => {
                #[$crate::runtime_tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_core::runtime_tests::atomic::test_kernel_atomic_numeric::<
                        TestRuntime,
                        IntType,
                    >(client, 1, $op);
                }
            };
        }

        macro_rules! test_int_op {
            ($name:ident, $op:expr) => {
                #[$crate::runtime_tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_core::runtime_tests::atomic::test_kernel_atomic_int::<
                        TestRuntime,
                        IntType,
                    >(client, $op);
                }
            };
        }

        test_numeric_op!(
            test_atomic_load_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Load
        );
        test_numeric_op!(
            test_atomic_store_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Store
        );
        test_numeric_op!(
            test_atomic_swap_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Swap
        );
        test_numeric_op!(
            test_atomic_add_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Add
        );
        test_numeric_op!(
            test_atomic_sub_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Sub
        );
        test_numeric_op!(
            test_atomic_min_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Min
        );
        test_numeric_op!(
            test_atomic_max_int,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Max
        );

        test_int_op!(
            test_atomic_and_int,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::And
        );
        test_int_op!(
            test_atomic_or_int,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::Or
        );
        test_int_op!(
            test_atomic_xor_int,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::Xor
        );
        test_int_op!(
            test_atomic_compare_exchange_int,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::CompareExchange
        );
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_uint {
    () => {
        use super::*;

        macro_rules! test_numeric_op {
            ($name:ident, $op:expr) => {
                #[$crate::runtime_tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_core::runtime_tests::atomic::test_kernel_atomic_numeric::<
                        TestRuntime,
                        UintType,
                    >(client, 1, $op);
                }
            };
        }

        macro_rules! test_int_op {
            ($name:ident, $op:expr) => {
                #[$crate::runtime_tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_core::runtime_tests::atomic::test_kernel_atomic_int::<
                        TestRuntime,
                        UintType,
                    >(client, $op);
                }
            };
        }

        test_numeric_op!(
            test_atomic_load_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Load
        );
        test_numeric_op!(
            test_atomic_store_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Store
        );
        test_numeric_op!(
            test_atomic_swap_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Swap
        );
        test_numeric_op!(
            test_atomic_add_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Add
        );
        test_numeric_op!(
            test_atomic_sub_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Sub
        );
        test_numeric_op!(
            test_atomic_min_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Min
        );
        test_numeric_op!(
            test_atomic_max_uint,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Max
        );

        test_int_op!(
            test_atomic_and_uint,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::And
        );
        test_int_op!(
            test_atomic_or_uint,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::Or
        );
        test_int_op!(
            test_atomic_xor_uint,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::Xor
        );
        test_int_op!(
            test_atomic_compare_exchange_uint,
            cubecl_core::runtime_tests::atomic::IntAtomicOp::CompareExchange
        );
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_atomic_float {
    () => {
        use super::*;

        macro_rules! test_numeric_op {
            ($name:ident, $op:expr) => {
                #[$crate::runtime_tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_core::runtime_tests::atomic::test_kernel_atomic_numeric_all_sizes::<
                        TestRuntime,
                        FloatType,
                    >(client, $op);
                }
            };
        }

        test_numeric_op!(
            test_atomic_load_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Load
        );

        test_numeric_op!(
            test_atomic_store_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Store
        );

        test_numeric_op!(
            test_atomic_swap_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Swap
        );

        test_numeric_op!(
            test_atomic_add_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Add
        );

        test_numeric_op!(
            test_atomic_sub_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Sub
        );

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        test_numeric_op!(
            test_atomic_min_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Min
        );

        /// Not available on CUDA and I have no access to a GPU that supports it in SPIR-V, but
        /// here for future proofing. Requires support for `VK_EXT_shader_atomic_float2`.
        test_numeric_op!(
            test_atomic_max_float,
            cubecl_core::runtime_tests::atomic::NumericAtomicOp::Max
        );
    };
}
