use crate::{
    self as cubecl,
    prelude::barrier::Barrier,
    wgmma::{Accumulator, Fragment, Major, MatrixDescriptor, Swizzle, WgmmaTileLayout},
};
use cubecl::prelude::*;
use cubecl_ir::features::{Tma, WgmmaConfig};
use cubecl_runtime::runtime::Runtime;
use num_traits::NumCast;

use alloc::{vec, vec::Vec};
use std::println;

/// Where a test reads `A` from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AOperand {
    Shared(Major, Swizzle),
    Registers,
}

/// One warpgroup MMA test: `out = lhs * rhs^T`, `64 x n`, over `k_steps` steps of K.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgmmaCase {
    pub n: usize,
    pub k_steps: usize,
    pub a: AOperand,
    pub b: (Major, Swizzle),
    /// Commits each step and waits on the previous one in a loop, rather than committing once.
    pub pipelined: bool,
}

impl WgmmaCase {
    pub fn new(n: usize, k_steps: usize, a: AOperand, b: (Major, Swizzle)) -> Self {
        Self {
            n,
            k_steps,
            a,
            b,
            pipelined: false,
        }
    }

    pub fn pipelined(self) -> Self {
        Self {
            pipelined: true,
            ..self
        }
    }

    fn a_layout(&self, size_k: usize) -> WgmmaTileLayout {
        let (major, swizzle) = match self.a {
            AOperand::Shared(major, swizzle) => (major, swizzle),
            AOperand::Registers => (Major::K, Swizzle::None),
        };
        WgmmaTileLayout {
            major,
            swizzle,
            rows: 64,
            k: size_k,
        }
    }

    fn b_layout(&self, size_k: usize) -> WgmmaTileLayout {
        WgmmaTileLayout {
            major: self.b.0,
            swizzle: self.b.1,
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
                AOperand::Shared(_, _) => {
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

/// [`kernel_wgmma`] with K-major tiles swizzled 128 bytes wide, loaded by TMA.
#[cube(launch)]
pub fn kernel_wgmma_tma(
    lhs: &TensorMap<half::f16, Tiled>,
    rhs: &TensorMap<half::f16, Tiled>,
    out: &mut [f32],
    #[comptime] n: usize,
) {
    let a_layout = comptime![WgmmaTileLayout {
        major: Major::K,
        swizzle: Swizzle::B128,
        rows: 64,
        k: 64,
    }];
    let b_layout = comptime![WgmmaTileLayout {
        major: Major::K,
        swizzle: Swizzle::B128,
        rows: n,
        k: 64,
    }];
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut smem_a: Shared<[half::f16]> = Shared::new_aligned_slice(64 * 64usize, 1024usize);
    let mut smem_b: Shared<[half::f16]> = Shared::new_aligned_slice(n * 64, 1024usize);
    let bytes = comptime![((64 + n) * 64 * 2) as u32];
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
    for step in 0..4usize {
        acc.execute(&a.at(0usize, step * 16), &b.at(0usize, step * 16));
    }
    let acc = acc.wait();

    #[unroll]
    for nth in 0..acc.len() {
        let (row, col) = acc.position_of_nth(UNIT_POS, nth as u32);
        out[row as usize * n + col as usize] = acc.get(nth);
    }
}

fn supported<A: CubeElement, B: CubeElement, CD: CubeElement>(
    client: &Client,
    n: usize,
    k: usize,
) -> bool {
    let supported = client
        .features()
        .matmul
        .wgmma
        .iter()
        .any(|config: &WgmmaConfig| {
            config.a_type == A::cube_type()
                && config.b_type == B::cube_type()
                && config.cd_type == CD::cube_type()
                && config.k as usize == k
                && config.supports_n(n as u32)
        });
    if !supported {
        println!(
            "Skipping wgmma test for a: {:?} b: {:?}, cd: {:?}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            CD::cube_type()
        );
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
    assert_eq!(actual.len(), 64 * n, "the kernel did not run: {what}");
    for i in 0..64 {
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
    let k = WgmmaTileLayout::k_step(size_of::<A>());
    if !supported::<A, B, CD>(&client, case.n, k) {
        return;
    }
    let (m, n) = (64, case.n);
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
    if !client.features().tma.contains(Tma::Base)
        || !supported::<half::f16, half::f16, f32>(&client, n, 16)
    {
        return;
    }
    let size_k = 64;
    let lhs: Vec<half::f16> = (0..64)
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
        .with_swizzle(TensorMapSwizzle::B128)
    };
    let out = client.create_from_slice(f32::as_bytes(&vec![0.0; 64 * n]));

    kernel_wgmma_tma::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(128),
        map(&lhs, 64),
        map(&rhs, n),
        unsafe { BufferArg::from_raw_parts(out.clone(), 64 * n) },
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
            use cubecl_core::runtime_tests::wgmma::{AOperand, WgmmaCase};
            use cubecl_core::wgmma::{Major, Swizzle};
            use half::{bf16, f16};

            const K_MAJOR: (Major, Swizzle) = (Major::K, Swizzle::None);

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

            fn shared(major: Major, swizzle: Swizzle) -> AOperand {
                AOperand::Shared(major, swizzle)
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_shapes() {
                for n in [8, 64, 256] {
                    let case = WgmmaCase::new(n, 1, shared(Major::K, Swizzle::None), K_MAJOR);
                    test::<f16, f16, f32>(case);
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_every_layout() {
                let swizzles = [Swizzle::None, Swizzle::B32, Swizzle::B64, Swizzle::B128];
                for major in [Major::K, Major::MN] {
                    for swizzle in swizzles {
                        // Four steps of K are 128 bytes, a whole panel at the widest swizzle.
                        let a = WgmmaCase::new(64, 4, shared(major, swizzle), K_MAJOR);
                        test::<f16, f16, f32>(a);
                        let k_major_a = shared(Major::K, Swizzle::None);
                        test::<f16, f16, f32>(WgmmaCase::new(64, 4, k_major_a, (major, swizzle)));
                    }
                }
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_accumulates_over_k() {
                let b64 = shared(Major::K, Swizzle::B64);
                test::<f16, f16, f32>(WgmmaCase::new(128, 4, b64, K_MAJOR));
                let b32 = shared(Major::K, Swizzle::B32);
                test::<f16, f16, f16>(WgmmaCase::new(64, 4, b32, K_MAJOR));
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_pipelined() {
                let b128 = (Major::K, Swizzle::B128);
                let case = WgmmaCase::new(64, 8, shared(b128.0, b128.1), b128);
                test::<f16, f16, f32>(case.pipelined());
            }

            #[$crate::runtime_tests::test_log::test]
            fn f16_registers_a() {
                test::<f16, f16, f32>(WgmmaCase::new(64, 1, AOperand::Registers, K_MAJOR));
                let mn = (Major::MN, Swizzle::B128);
                test::<f16, f16, f32>(WgmmaCase::new(128, 2, AOperand::Registers, mn));
            }

            #[$crate::runtime_tests::test_log::test]
            fn bf16() {
                let mn = shared(Major::MN, Swizzle::B64);
                test::<bf16, bf16, f32>(WgmmaCase::new(64, 4, mn, K_MAJOR));
                test::<bf16, bf16, f32>(WgmmaCase::new(64, 2, AOperand::Registers, K_MAJOR));
            }

            #[$crate::runtime_tests::test_log::test]
            fn tf32() {
                let b128 = (Major::K, Swizzle::B128);
                test::<tf32, tf32, f32>(WgmmaCase::new(
                    8,
                    1,
                    shared(Major::K, Swizzle::None),
                    K_MAJOR,
                ));
                test::<tf32, tf32, f32>(WgmmaCase::new(64, 4, shared(b128.0, b128.1), b128));
                test::<tf32, tf32, f32>(WgmmaCase::new(64, 2, AOperand::Registers, K_MAJOR));
            }

            #[$crate::runtime_tests::test_log::test]
            fn fp8() {
                let b128 = (Major::K, Swizzle::B128);
                test::<e4m3, e4m3, f32>(WgmmaCase::new(64, 4, shared(b128.0, b128.1), b128));
                let b64 = shared(Major::K, Swizzle::B64);
                test::<e5m2, e4m3, f16>(WgmmaCase::new(64, 2, b64, K_MAJOR));
                test::<e4m3, e5m2, f32>(WgmmaCase::new(64, 1, AOperand::Registers, K_MAJOR));
            }

            #[$crate::runtime_tests::test_log::test]
            fn int8() {
                let b128 = (Major::K, Swizzle::B128);
                test::<i8, i8, i32>(WgmmaCase::new(64, 4, shared(b128.0, b128.1), b128));
                test::<u8, i8, i32>(WgmmaCase::new(
                    32,
                    1,
                    shared(Major::K, Swizzle::None),
                    K_MAJOR,
                ));
                test::<i8, u8, i32>(WgmmaCase::new(64, 1, AOperand::Registers, K_MAJOR));
            }

            #[$crate::runtime_tests::test_log::test]
            fn tma_loaded_tiles() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::wgmma::test_wgmma_tma::<TestRuntime>(client, 128);
            }
        }
    };
}
