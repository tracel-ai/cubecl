use crate::{self as cubecl, wgmma::Swizzle};
use cubecl::prelude::*;
use cubecl_ir::{
    features::WgmmaConfig,
    types::{MatrixIdent, MatrixLayout},
};
use cubecl_runtime::runtime::Runtime;
use num_traits::NumCast;

use alloc::{vec, vec::Vec};
use std::println;

/// Offset of `(row, col)` in a K-major tile of 8x16-byte core matrices without swizzle. The core
/// matrices along K are adjacent (a leading byte offset of 128), and each group of 8 rows follows
/// the previous one (a stride byte offset of `8 * tile_k * size`).
#[cube]
fn core_matrix_offset(
    row: usize,
    col: usize,
    #[comptime] tile_k: usize,
    #[comptime] per_row: usize,
) -> usize {
    let core = (row / 8) * comptime![tile_k / per_row] + col / per_row;
    core * comptime![8 * per_row] + (row % 8) * per_row + col % per_row
}

/// Stages a row-major `rows x (tile_k * k_tiles)` matrix in shared memory as `k_tiles` K-major
/// tiles of core matrices.
#[cube]
fn stage<E: Scalar>(
    source: &[E],
    stage: &mut [E],
    #[comptime] rows: usize,
    #[comptime] tile_k: usize,
    #[comptime] k_tiles: usize,
) {
    let size_k = comptime![tile_k * k_tiles];
    let total = comptime![rows * size_k];
    let elem_size = E::size();
    let per_row = comptime![16 / elem_size];
    #[unroll]
    for step in 0..comptime![total.div_ceil(128)] {
        let i = step * 128 + UNIT_POS as usize;
        if i < total {
            let row = i / size_k;
            let col = i % size_k;
            let tile = col / tile_k;
            let offset = core_matrix_offset(row, col % tile_k, tile_k, per_row);
            stage[tile * comptime![rows * tile_k] + offset] = source[i];
        }
    }
}

/// `out = lhs * rhs^T` for a `64 x (k * k_tiles)` `lhs` and an `n x (k * k_tiles)` `rhs`, both
/// row-major, so both operands are K-major. One warpgroup issues `k_tiles` MMAs, `A` read from
/// shared memory or from registers.
#[cube(launch)]
pub fn kernel_wgmma<A: Scalar, B: Scalar, CD: Numeric>(
    lhs: &[A],
    rhs: &[B],
    out: &mut [CD],
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] k_tiles: usize,
    #[comptime] a_in_registers: bool,
) {
    let def = wgmma::WgmmaDefinition::<A, B, CD>::new(n);
    let size_k = comptime![k * k_tiles];
    let tile_a = comptime![64 * k];
    let tile_b = comptime![n * k];

    let mut smem_a = Shared::new_aligned_slice(64 * size_k, 128usize);
    let mut smem_b = Shared::new_aligned_slice(n * size_k, 128usize);
    stage::<A>(lhs, &mut smem_a, 64usize, k, k_tiles);
    stage::<B>(rhs, &mut smem_b, n, k, k_tiles);

    // The units wrote shared memory through the generic proxy, and the MMA reads it through the
    // async proxy.
    sync_async_proxy_shared();
    sync_cube();

    let leading = 128u32;
    // `size` is the element's only outside `comptime!`.
    let a_size = A::size();
    let b_size = B::size();
    let stride_a = comptime![(8 * k * a_size) as u32];
    let stride_b = comptime![(8 * k * b_size) as u32];

    let acc_len = def.elems_per_unit(MatrixIdent::Accumulator);
    let size!(NC) = 2usize;
    let mut acc = Array::<Vector<CD, NC>>::new(comptime![acc_len / 2]);

    if a_in_registers {
        let run = def.contiguous_elems(MatrixIdent::A);
        let size!(NA) = comptime![run as usize];
        let a_len = def.elems_per_unit(MatrixIdent::A);
        let a_vectors = comptime![a_len / run as usize];
        let mut registers_a = Array::<Vector<A, NA>>::new(a_vectors);
        #[unroll]
        for t in 0..k_tiles {
            #[unroll]
            for v in 0..a_vectors {
                let mut reg = Vector::empty();
                #[unroll]
                for e in 0..comptime![run as usize] {
                    let nth = v * comptime![run as usize] + e;
                    let (row, col) = def.position_of_nth(UNIT_POS, nth as u32, MatrixIdent::A);
                    reg.insert(e, lhs[row as usize * size_k + t * k + col as usize]);
                }
                registers_a[v] = reg;
            }
            let b = wgmma::descriptor(
                &smem_b[t * tile_b..(t + 1) * tile_b],
                leading,
                stride_b,
                Swizzle::None,
            );
            // The fragment was just written, and the MMA reads it out of the same registers.
            wgmma::fence_operand(&mut registers_a);
            wgmma::fence_operand(&mut acc);
            wgmma::fence();
            def.execute_registers_a(&registers_a, b, &mut acc, t > 0, MatrixLayout::ColMajor);
            wgmma::commit_group();
            // The next tile overwrites the fragment this MMA reads.
            wgmma::wait_group(0usize);
        }
    } else {
        wgmma::fence_operand(&mut acc);
        wgmma::fence();
        #[unroll]
        for t in 0..k_tiles {
            let a = wgmma::descriptor(
                &smem_a[t * tile_a..(t + 1) * tile_a],
                leading,
                stride_a,
                Swizzle::None,
            );
            let b = wgmma::descriptor(
                &smem_b[t * tile_b..(t + 1) * tile_b],
                leading,
                stride_b,
                Swizzle::None,
            );
            def.execute(
                a,
                b,
                &mut acc,
                t > 0,
                MatrixLayout::RowMajor,
                MatrixLayout::ColMajor,
            );
        }
        wgmma::commit_group();
        wgmma::wait_group(0usize);
    }
    wgmma::fence_operand(&mut acc);

    #[unroll]
    for v in 0..comptime![acc_len / 2] {
        let reg = acc[v];
        #[unroll]
        for e in 0..2 {
            let nth = v * 2 + e;
            let (row, col) = def.position_of_nth(UNIT_POS, nth as u32, MatrixIdent::Accumulator);
            out[row as usize * n + col as usize] = reg.extract(e);
        }
    }
}

pub fn test_wgmma<
    R: Runtime,
    A: CubeElement + Scalar + NumCast,
    B: CubeElement + Scalar + NumCast,
    CD: CubeElement + Numeric,
>(
    client: Client,
    n: usize,
    k: usize,
    k_tiles: usize,
    a_in_registers: bool,
) {
    let m = 64;
    let supported = client
        .features()
        .matmul
        .wgmma
        .iter()
        .any(|config: &WgmmaConfig| {
            config.a_type == A::cube_type()
                && config.b_type == B::cube_type()
                && config.cd_type == CD::cube_type()
                && config.m as usize == m
                && config.k as usize == k
                && config.supports_n(n as u32)
        });
    if !supported {
        println!(
            "Skipping wgmma test for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            CD::cube_type()
        );
        return;
    }

    let size_k = k * k_tiles;
    // Small integers, exact in every element type and in an `f16` accumulator.
    let lhs_at = |i: usize, l: usize| ((i * 2 + l) % 5) as i64;
    let rhs_at = |l: usize, j: usize| ((l * 3 + j) % 7) as i64;

    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..size_k).map(move |l| A::from(lhs_at(i, l)).unwrap()))
        .collect();
    // Stored transposed, `n x size_k`, so `B` is K-major.
    let rhs: Vec<B> = (0..n)
        .flat_map(|j| (0..size_k).map(move |l| B::from(rhs_at(l, j)).unwrap()))
        .collect();

    let lhs = client.create_from_slice(A::as_bytes(&lhs));
    let rhs = client.create_from_slice(B::as_bytes(&rhs));
    let out = client.create_from_slice(CD::as_bytes(&vec![CD::from_int(0); m * n]));

    unsafe {
        kernel_wgmma::launch::<A, B, CD>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(128),
            BufferArg::from_raw_parts(lhs, m * size_k),
            BufferArg::from_raw_parts(rhs, n * size_k),
            BufferArg::from_raw_parts(out.clone(), m * n),
            n,
            k,
            k_tiles,
            a_in_registers,
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = CD::from_bytes(&actual);
    assert_eq!(actual.len(), m * n, "the kernel did not run");

    for i in 0..m {
        for j in 0..n {
            let expected: i64 = (0..size_k).map(|l| lhs_at(i, l) * rhs_at(l, j)).sum();
            let actual = actual[i * n + j].to_f64().unwrap();
            assert_eq!(
                actual,
                expected as f64,
                "{:?} x {:?} -> {:?}, n: {n}, k: {k}, tiles: {k_tiles}, A in registers: \
                 {a_in_registers}: wrong value at ({i}, {j})",
                A::cube_type(),
                B::cube_type(),
                CD::cube_type(),
            );
        }
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_wgmma {
    () => {
        mod wgmma {
            use super::*;
            use cubecl_common::*;
            use cubecl_core::num_traits::cast::NumCast;
            use half::{bf16, f16};

            fn test<
                A: CubeElement + Scalar + NumCast,
                B: CubeElement + Scalar + NumCast,
                CD: CubeElement + Numeric,
            >(
                n: usize,
                k: usize,
                k_tiles: usize,
                a_in_registers: bool,
            ) {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::wgmma::test_wgmma::<TestRuntime, A, B, CD>(
                    client,
                    n,
                    k,
                    k_tiles,
                    a_in_registers,
                )
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_shared() {
                test::<f16, f16, f32>(8, 16, 1, false);
                test::<f16, f16, f32>(64, 16, 1, false);
                test::<f16, f16, f32>(256, 16, 1, false);
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_accumulates_over_k() {
                test::<f16, f16, f32>(128, 16, 4, false);
                test::<f16, f16, f16>(64, 16, 4, false);
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_registers_a() {
                test::<f16, f16, f32>(64, 16, 1, true);
                test::<f16, f16, f32>(128, 16, 2, true);
            }

            #[$crate::runtime_tests::test_log::test]
            fn bf16() {
                test::<bf16, bf16, f32>(64, 16, 2, false);
                test::<bf16, bf16, f32>(64, 16, 2, true);
            }

            #[$crate::runtime_tests::test_log::test]
            fn tf32() {
                test::<tf32, tf32, f32>(8, 8, 1, false);
                test::<tf32, tf32, f32>(64, 8, 2, false);
                test::<tf32, tf32, f32>(64, 8, 2, true);
            }

            #[$crate::runtime_tests::test_log::test]
            fn fp8() {
                test::<e4m3, e4m3, f32>(64, 32, 2, false);
                test::<e5m2, e4m3, f16>(64, 32, 2, false);
                test::<e4m3, e5m2, f32>(64, 32, 1, true);
            }

            #[$crate::runtime_tests::test_log::test]
            fn int8() {
                test::<i8, i8, i32>(64, 32, 2, false);
                test::<u8, i8, i32>(32, 32, 1, false);
                test::<i8, u8, i32>(64, 32, 1, true);
            }
        }
    };
}
