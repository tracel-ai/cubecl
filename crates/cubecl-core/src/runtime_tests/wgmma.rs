use crate::{
    self as cubecl,
    prelude::barrier::Barrier,
    wgmma::{
        Accumulator, Fragment, Major, MatrixDescriptor, Swizzle, WARPGROUP_M, WgmmaTileLayout,
    },
};
use cubecl::prelude::*;
use cubecl_ir::{
    features::{Tma, WgmmaConfig, WgmmaElems},
    types::MatrixShape,
};
use cubecl_runtime::runtime::Runtime;
use num_traits::NumCast;

use alloc::{vec, vec::Vec};
use std::println;

/// How a tile is laid out in shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileFormat {
    pub major: Major,
    pub swizzle: Swizzle,
}

impl TileFormat {
    pub const K_MAJOR_UNSWIZZLED: Self = Self {
        major: Major::K,
        swizzle: Swizzle::None,
    };
}

/// Where a test reads `A` from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AOperand {
    Shared(TileFormat),
    Registers,
}

/// One warpgroup MMA test: `out = lhs * rhs^T`, `64 x n`, over `k_steps` steps of K.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgmmaCase {
    pub n: usize,
    pub k_steps: usize,
    pub a: AOperand,
    pub b: TileFormat,
    /// Commits each step and waits on the previous one in a loop, rather than committing once.
    /// Only with `A` in shared memory.
    pub pipelined: bool,
}

impl WgmmaCase {
    fn a_layout(&self, size_k: usize) -> WgmmaTileLayout {
        let format = match self.a {
            AOperand::Shared(format) => format,
            AOperand::Registers => TileFormat::K_MAJOR_UNSWIZZLED,
        };
        WgmmaTileLayout {
            major: format.major,
            swizzle: format.swizzle,
            rows: WARPGROUP_M,
            k: size_k,
        }
    }

    fn b_layout(&self, size_k: usize) -> WgmmaTileLayout {
        WgmmaTileLayout {
            major: self.b.major,
            swizzle: self.b.swizzle,
            rows: self.n,
            k: size_k,
        }
    }
}

/// Writes the row-major `rows x k` `source` into `tile`, laid out as `layout`.
#[cube]
fn stage<E: Scalar>(source: &[E], tile: &mut [E], #[comptime] layout: WgmmaTileLayout) {
    let size_k = comptime![layout.k];
    let total = comptime![layout.rows * layout.k];
    let elem_size = E::size();
    #[unroll]
    for step in 0..comptime![total.div_ceil(128)] {
        let i = step * 128 + UNIT_POS as usize;
        if i < total {
            tile[layout.offset(i / size_k, i % size_k, elem_size)] = source[i];
        }
    }
}

/// `out = lhs * rhs^T` for a `64 x size_k` `lhs` and an `n x size_k` `rhs`, both row-major. One
/// warpgroup stages them in shared memory as the case lays them out, and issues the MMAs.
#[cube(launch)]
pub fn kernel_wgmma<A: Scalar, B: Scalar, CD: Numeric>(
    lhs: &[A],
    rhs: &[B],
    out: &mut [CD],
    #[comptime] case: WgmmaCase,
) {
    let a_size = A::size();
    let k = comptime![WgmmaTileLayout::k_step(a_size)];
    let size_k = comptime![k * case.k_steps];
    let a_layout = comptime![case.a_layout(size_k)];
    let b_layout = comptime![case.b_layout(size_k)];
    let n = comptime![case.n];

    let mut smem_a = Shared::new_aligned_slice(64 * size_k, a_layout.alignment());
    let mut smem_b = Shared::new_aligned_slice(n * size_k, b_layout.alignment());
    stage::<A>(lhs, &mut smem_a, a_layout);
    stage::<B>(rhs, &mut smem_b, b_layout);
    // The units wrote shared memory through the generic proxy, and the MMA reads it through the
    // async proxy.
    sync_async_proxy_shared();
    sync_cube();

    let a = MatrixDescriptor::new(&smem_a, a_layout);
    let b = MatrixDescriptor::new(&smem_b, b_layout);
    let mut acc = Accumulator::<CD>::new(n).start();

    if comptime![case.pipelined] {
        let mut released = acc.commit();
        for step in 0..case.k_steps {
            acc.execute(&a.at(0usize, step * k), &b.at(0usize, step * k));
            let read = acc.commit();
            released.wait();
            released = read;
        }
    } else {
        #[unroll]
        for step in 0..case.k_steps {
            let b = b.at(0usize, step * k);
            match comptime![case.a] {
                AOperand::Registers => {
                    let mut fragment = Fragment::<A>::new();
                    #[unroll]
                    for nth in 0..fragment.len() {
                        let (row, col) = fragment.position_of_nth(UNIT_POS, nth as u32);
                        fragment.set(nth, lhs[row as usize * size_k + step * k + col as usize]);
                    }
                    acc.execute_registers(&fragment, &b);
                    // The next step writes the fragment this MMA reads.
                    acc.commit().wait();
                }
                AOperand::Shared(_) => {
                    acc.execute(&a.at(0usize, step * k), &b);
                }
            }
        }
    }
    let acc = acc.wait();

    #[unroll]
    for nth in 0..acc.len() {
        let (row, col) = acc.position_of_nth(UNIT_POS, nth as u32);
        out[row as usize * n + col as usize] = acc.get(nth);
    }
}

/// [`kernel_wgmma`] with tiles of `f16` K-major and swizzled 128 bytes wide, loaded by TMA: K is
/// one 128-byte panel, so each tile is one box.
#[cube(launch)]
pub fn kernel_wgmma_tma(
    lhs: &TensorMap<half::f16, Tiled>,
    rhs: &TensorMap<half::f16, Tiled>,
    out: &mut [f32],
    #[comptime] n: usize,
) {
    let a_layout = comptime![tma_tile(WARPGROUP_M)];
    let b_layout = comptime![tma_tile(n)];
    let size_k = comptime![a_layout.k];
    let k = comptime![WgmmaTileLayout::k_step(F16_BYTES)];

    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut smem_a: Shared<[half::f16]> =
        Shared::new_aligned_slice(WARPGROUP_M * size_k, a_layout.alignment());
    let mut smem_b: Shared<[half::f16]> =
        Shared::new_aligned_slice(n * size_k, b_layout.alignment());
    let bytes = comptime![((WARPGROUP_M + n) * size_k * F16_BYTES) as u32];
    let expected = select(UNIT_POS == 0, bytes, 0);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(lhs, &mut smem_a, 0, 0);
        barrier.tma_load_2d(rhs, &mut smem_b, 0, 0);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let a = MatrixDescriptor::new(&smem_a, a_layout);
    let b = MatrixDescriptor::new(&smem_b, b_layout);
    let mut acc = Accumulator::<f32>::new(n).start();
    #[unroll]
    for step in 0..comptime![size_k / k] {
        acc.execute(&a.at(0usize, step * k), &b.at(0usize, step * k));
    }
    let acc = acc.wait();

    #[unroll]
    for nth in 0..acc.len() {
        let (row, col) = acc.position_of_nth(UNIT_POS, nth as u32);
        out[row as usize * n + col as usize] = acc.get(nth);
    }
}

const F16_BYTES: usize = 2;

/// A `rows x 64` tile of `f16`, one TMA box of K-major lines 128 bytes wide.
fn tma_tile(rows: usize) -> WgmmaTileLayout {
    WgmmaTileLayout {
        major: Major::K,
        swizzle: Swizzle::B128,
        rows,
        k: Swizzle::B128.width() / F16_BYTES,
    }
}

fn supported<A: CubeElement, B: CubeElement, CD: CubeElement>(
    client: &Client,
    n: usize,
    k: usize,
) -> bool {
    let elems = WgmmaElems {
        a: A::cube_type(),
        b: B::cube_type(),
        cd: CD::cube_type(),
    };
    let shape = MatrixShape {
        m: WARPGROUP_M,
        n,
        k,
    };
    let features = client.features();
    let supported = features
        .matmul
        .wgmma
        .iter()
        .any(|config: &WgmmaConfig| config.matches(elems, shape));
    if !supported {
        println!("Skipping wgmma test for {elems:?}, {shape:?}");
    }
    supported
}

// Small integers, exact in every element type and in an `f16` accumulator.
fn lhs_at(i: usize, l: usize) -> i64 {
    ((i * 2 + l) % 5) as i64
}

fn rhs_at(l: usize, j: usize) -> i64 {
    ((l * 3 + j) % 7) as i64
}

fn check<CD: CubeElement + Numeric>(actual: &[CD], n: usize, size_k: usize, what: &str) {
    assert_eq!(
        actual.len(),
        WARPGROUP_M * n,
        "the kernel did not run: {what}"
    );
    for i in 0..WARPGROUP_M {
        for j in 0..n {
            let expected: i64 = (0..size_k).map(|l| lhs_at(i, l) * rhs_at(l, j)).sum();
            let actual = actual[i * n + j].to_f64().unwrap();
            assert_eq!(actual, expected as f64, "{what}: wrong value at ({i}, {j})");
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
    case: WgmmaCase,
) {
    assert!(
        !case.pipelined || case.a != AOperand::Registers,
        "a pipelined case reads A from shared memory"
    );
    let k = WgmmaTileLayout::k_step(size_of::<A>());
    if !supported::<A, B, CD>(&client, case.n, k) {
        return;
    }
    let (m, n) = (WARPGROUP_M, case.n);
    let size_k = k * case.k_steps;

    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..size_k).map(move |l| A::from(lhs_at(i, l)).unwrap()))
        .collect();
    // `n x size_k`, transposed.
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
            case,
        )
    };

    let actual = client.read_one_unchecked(out);
    let what = std::format!(
        "{:?} x {:?} -> {:?}, {case:?}",
        A::cube_type(),
        B::cube_type(),
        CD::cube_type()
    );
    check(CD::from_bytes(&actual), n, size_k, &what);
}

pub fn test_wgmma_tma<R: Runtime>(client: Client, n: usize) {
    let size_k = tma_tile(n).k;
    let k = WgmmaTileLayout::k_step(F16_BYTES);
    if !client.features().tma.contains(Tma::Base)
        || !supported::<half::f16, half::f16, f32>(&client, n, k)
    {
        println!("Skipping the TMA-fed wgmma test for n = {n}: no TMA or no such wgmma");
        return;
    }
    let lhs: Vec<half::f16> = (0..WARPGROUP_M)
        .flat_map(|i| (0..size_k).map(move |l| half::f16::from_f32(lhs_at(i, l) as f32)))
        .collect();
    let rhs: Vec<half::f16> = (0..n)
        .flat_map(|j| (0..size_k).map(move |l| half::f16::from_f32(rhs_at(l, j) as f32)))
        .collect();

    let map = |values: &[half::f16], rows: usize| {
        let shape: cubecl_zspace::Shape = [rows, size_k].into();
        let layout = client.create_tensor_from_slice(half::f16::as_bytes(values), shape.clone(), 2);
        let tensor = unsafe { TensorArg::from_raw_parts(layout.memory, layout.strides, shape) };
        TensorMapArg::new(
            TiledArgs {
                tile_size: cubecl_zspace::shape![rows, size_k],
            },
            tensor,
            half::f16::elem_type_native(),
        )
        .with_swizzle(tma_tile(rows).tensor_map_swizzle())
    };
    let out = client.create_from_slice(f32::as_bytes(&vec![0.0; WARPGROUP_M * n]));

    kernel_wgmma_tma::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(128),
        map(&lhs, WARPGROUP_M),
        map(&rhs, n),
        unsafe { BufferArg::from_raw_parts(out.clone(), WARPGROUP_M * n) },
        n,
    );

    let actual = client.read_one_unchecked(out);
    check(f32::from_bytes(&actual), n, size_k, "TMA-loaded f16 tiles");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_wgmma {
    () => {
        mod wgmma {
            use super::*;
            use cubecl_common::*;
            use cubecl_core::num_traits::cast::NumCast;
            use cubecl_core::runtime_tests::wgmma::{AOperand, TileFormat, WgmmaCase};
            use cubecl_core::wgmma::{Major, Swizzle};
            use half::{bf16, f16};

            const UNSWIZZLED: TileFormat = TileFormat::K_MAJOR_UNSWIZZLED;
            const SWIZZLED: TileFormat = TileFormat {
                major: Major::K,
                swizzle: Swizzle::B128,
            };

            fn test<
                A: CubeElement + Scalar + NumCast,
                B: CubeElement + Scalar + NumCast,
                CD: CubeElement + Numeric,
            >(
                case: WgmmaCase,
            ) {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::wgmma::test_wgmma::<TestRuntime, A, B, CD>(client, case)
            }

            /// `A` and `B` in shared memory, committed once.
            fn shared(n: usize, k_steps: usize, a: TileFormat, b: TileFormat) -> WgmmaCase {
                WgmmaCase {
                    n,
                    k_steps,
                    a: AOperand::Shared(a),
                    b,
                    pipelined: false,
                }
            }

            /// `A` in registers, one step committed and waited on at a time.
            fn registers(n: usize, k_steps: usize, b: TileFormat) -> WgmmaCase {
                WgmmaCase {
                    n,
                    k_steps,
                    a: AOperand::Registers,
                    b,
                    pipelined: false,
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_shapes() {
                for n in [8, 64, 256] {
                    test::<f16, f16, f32>(shared(n, 1, UNSWIZZLED, UNSWIZZLED));
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_every_layout() {
                let swizzles = [Swizzle::None, Swizzle::B32, Swizzle::B64, Swizzle::B128];
                for major in [Major::K, Major::MN] {
                    for swizzle in swizzles {
                        let format = TileFormat { major, swizzle };
                        // Four steps of K are 128 bytes, a whole panel at the widest swizzle.
                        test::<f16, f16, f32>(shared(64, 4, format, UNSWIZZLED));
                        test::<f16, f16, f32>(shared(64, 4, UNSWIZZLED, format));
                    }
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_accumulates_over_k() {
                let b64 = TileFormat {
                    major: Major::K,
                    swizzle: Swizzle::B64,
                };
                let b32 = TileFormat {
                    major: Major::K,
                    swizzle: Swizzle::B32,
                };
                test::<f16, f16, f32>(shared(128, 4, b64, UNSWIZZLED));
                test::<f16, f16, f16>(shared(64, 4, b32, UNSWIZZLED));
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_pipelined() {
                let case = WgmmaCase {
                    pipelined: true,
                    ..shared(64, 8, SWIZZLED, SWIZZLED)
                };
                test::<f16, f16, f32>(case);
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_registers_a() {
                let mn = TileFormat {
                    major: Major::MN,
                    swizzle: Swizzle::B128,
                };
                test::<f16, f16, f32>(registers(64, 1, UNSWIZZLED));
                test::<f16, f16, f32>(registers(128, 2, mn));
            }

            #[$crate::runtime_tests::test_log::test]
            fn bf16() {
                let mn = TileFormat {
                    major: Major::MN,
                    swizzle: Swizzle::B64,
                };
                test::<bf16, bf16, f32>(shared(64, 4, mn, UNSWIZZLED));
                test::<bf16, bf16, f32>(registers(64, 2, UNSWIZZLED));
            }

            #[$crate::runtime_tests::test_log::test]
            fn tf32() {
                test::<tf32, tf32, f32>(shared(8, 1, UNSWIZZLED, UNSWIZZLED));
                test::<tf32, tf32, f32>(shared(64, 4, SWIZZLED, SWIZZLED));
                test::<tf32, tf32, f32>(registers(64, 2, UNSWIZZLED));
            }

            #[$crate::runtime_tests::test_log::test]
            fn fp8() {
                let b64 = TileFormat {
                    major: Major::K,
                    swizzle: Swizzle::B64,
                };
                test::<e4m3, e4m3, f32>(shared(64, 4, SWIZZLED, SWIZZLED));
                test::<e5m2, e4m3, f16>(shared(64, 2, b64, UNSWIZZLED));
                test::<e4m3, e5m2, f32>(registers(64, 1, UNSWIZZLED));
            }

            #[$crate::runtime_tests::test_log::test]
            fn int8() {
                test::<i8, i8, i32>(shared(64, 4, SWIZZLED, SWIZZLED));
                test::<u8, i8, i32>(shared(32, 1, UNSWIZZLED, UNSWIZZLED));
                test::<i8, u8, i32>(registers(64, 1, UNSWIZZLED));
            }

            #[$crate::runtime_tests::test_log::test]
            fn tma_loaded_tiles() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::wgmma::test_wgmma_tma::<TestRuntime>(client, 128);
            }
        }
    };
}
