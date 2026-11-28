//! A storage-tiled operand masks and clamps where its coordinates do: `a` is stored
//! `[m / 16][k][m % 16]`, `M` as two pieces and `K` whole, and contracted through the software
//! matmul, which reads it through its N-D view. The window decides what a coordinate reads; the
//! split into pieces only says where it lies.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_tile::kind::{Boundary, Storage};
use cubecl_tile::{
    Dimension, Level, Levels, Partitioning, Projection, Semiring, SoftwareMatmul, Space, TileSpec,
    TileTensor, TileTensorLaunch,
};

use super::{Form, implied};

const M: Dimension = Dimension(0);
const N: Dimension = Dimension(1);
const K: Dimension = Dimension(2);

/// The rows of a stored tile of `a`.
const TILE_ROWS: usize = 16;
/// The product's rows, three tiles of them, and its columns.
const ROWS: usize = 48;
const COLS: usize = 4;
/// The depth `a` stores.
const DEPTH: usize = 8;

/// `c = a · b`, a leaf of one tile of rows at a time, `a` read up to row `until`.
#[cube(launch)]
fn tiled_bounds_matmul(
    a: &TileTensor<'_>,
    b: &TileTensor<'_>,
    c: &mut TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    until: usize,
) {
    let a = a.tile(&space).within(M, 0usize, until);
    let b = b.tile(&space);
    let mut c = c.tile_mut(&space);
    c.zero();
    for region in space.over(&level).unrolled() {
        let mut c_r = c
            .at(&region)
            .accumulating(comptime!(SoftwareMatmul::new(64)), Semiring::SumProd);
        c_r.mma(&a.at(&region), &b.at(&region));
    }
}

fn value(i: usize) -> f32 {
    ((i * 7 + i / 3) % 9) as f32 - 4.0
}

/// `c = a · b` over `depth` of `k`, `a` stored `DEPTH` deep under `boundaries` and read up to
/// row `until`; the product and `a`'s values as the host holds them.
fn product(depth: usize, until: usize, boundaries: &[Option<Boundary>]) -> (HostData, Vec<f32>) {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, ROWS), (N, COLS), (K, depth)]),
            Levels::leaf(&[(M, TILE_ROWS), (N, COLS), (K, depth)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a: Vec<f32> = (0..ROWS * DEPTH).map(value).collect();
    let mut stored = vec![0f32; ROWS * DEPTH];
    for (at, &v) in a.iter().enumerate() {
        let (m, k) = (at / DEPTH, at % DEPTH);
        stored[m / TILE_ROWS * DEPTH * TILE_ROWS + k * TILE_ROWS + m % TILE_ROWS] = v;
    }
    let b: Vec<f32> = (0..depth * COLS).map(|i| value(i + 5)).collect();
    let dtype = f32::elem_type_native();
    let a_handle = TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&stored)),
        vec![ROWS / TILE_ROWS, DEPTH, TILE_ROWS],
        vec![DEPTH * TILE_ROWS, TILE_ROWS, 1],
        dtype,
    );
    let b_handle = TensorHandle::new(
        client.create_from_slice(f32::as_bytes(&b)),
        vec![depth, COLS],
        vec![COLS, 1],
        dtype,
    );
    let c = TestInput::builder(client.clone(), Shape::from(vec![ROWS, COLS]))
        .zeros()
        .generate_without_host_data();
    let mut spec = TileSpec::new(Projection::tiled(&[M, K], &[(M, 0), (K, 0), (M, 1)]));
    spec.storage = Storage::Tiled(None);
    spec.boundaries = boundaries.iter().copied().collect();
    tiled_bounds_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileTensorLaunch::new(a_handle.binding().into_tensor_arg(), dtype, 1, spec),
        TileTensorLaunch::new(
            b_handle.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        TileTensorLaunch::new(
            c.clone().binding().into_tensor_arg(),
            c.dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        until,
    );
    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    let mut want = vec![0f32; ROWS * COLS];
    for (i, row) in want.chunks_mut(COLS).enumerate().take(until) {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..depth)
                .map(|l| a[i * DEPTH + l.min(DEPTH - 1)] * b[l * COLS + j])
                .sum();
        }
    }
    (got, want)
}

fn assert_product((got, want): (HostData, Vec<f32>), case: &str) {
    for i in 0..ROWS {
        for j in 0..COLS {
            assert_eq!(
                got.get_f32(&[i, j]),
                want[i * COLS + j],
                "({i}, {j}), {case}"
            );
        }
    }
}

/// Rows past `within`'s end read zero, inside a stored tile as past one.
#[test]
fn a_tiled_operand_placed_within_rows_reads_zero_past_them() {
    assert_product(
        product(DEPTH, 40, &[Some(Boundary::Zero), None]),
        "rows up to 40",
    );
}

/// `k` past the buffer's depth reads its last column, `k` stored whole beside `m`'s pieces.
#[test]
fn a_tiled_operand_clamped_along_a_whole_coordinate_reads_its_edge() {
    assert_product(
        product(DEPTH + 2, ROWS, &[None, Some(Boundary::Clamp)]),
        "k clamped onto its last column",
    );
}
