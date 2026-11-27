//! `acc.mma(a, b)` over 8-bit floats, `a` stored as RDNA4's matrix instruction takes its first
//! operand, `[m / 16][k / 8][m % 16][k % 8]`, its fragments loaded straight from it several
//! instruction steps deep: each step reads the fragment's cells where they lie for the first, one
//! step along `k` past them.
//!
//! An 8-bit float product holds its sum in this crate's own transport, so both operands load cell
//! by cell into the instruction's registers. The tiles are stated where the buffer is bound, or
//! read off the buffer when the kernel runs.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    client::Client,
    ir::FloatKind,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Tiling, metadata::Metadata},
};
use cubecl_common::e4m3;
use cubecl_tile::kind::Storage;
use cubecl_tile::{
    Dimension, Level, Levels, Partitioning, Projection, Semiring, Space, TileSpec, TileTensor,
    TileTensorLaunch,
};

use super::{Form, implied};

const M: Dimension = Dimension(0);
const N: Dimension = Dimension(1);
const K: Dimension = Dimension(2);

/// A stored tile of `a`: sixteen rows of eight values of `k`.
const TILE_ROWS: usize = 16;
const TILE_K: usize = 8;
/// The product's rows: three tiles of them.
const ROWS: usize = 48;
/// The contraction's depth, and the leaf's along it: a fragment several instructions deep.
const DEPTH: usize = 128;
const WINDOW: usize = 64;

/// `c = a · b`, a leaf of `WINDOW` along `k` at a time, the sum held in the instruction an 8-bit
/// float product picks.
#[cube(launch)]
fn tiled_fp8_matmul(
    a: &TileTensor<'_>,
    b: &TileTensor<'_>,
    c: &mut TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile_mut(&space);
    let acc = c.accumulator::<f32>(&a, &b, Semiring::SumProd);
    // The leaf cuts the rows into their tiles, each a partition of its own: walked unrolled.
    for region in space.over(&level).unrolled() {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
}

/// The `n × k` of the device's instruction over two `e4m3` operands summing in `f32` whose rows
/// are a stored tile's, `None` where it offers none.
fn fp8_instruction(client: &Client) -> Option<(usize, usize)> {
    let fp8 = ElemType::Float(FloatKind::E4M3);
    let f32_ty = f32::elem_type_native();
    client
        .properties()
        .features
        .matmul
        .mma
        .iter()
        .find(|config| {
            config.a_type == fp8
                && config.b_type == fp8
                && config.cd_type == f32_ty
                && config.m as usize == TILE_ROWS
                && WINDOW.is_multiple_of(config.k as usize)
                && WINDOW > config.k as usize
        })
        .map(|config| (config.n as usize, config.k as usize))
}

/// A value exact in `e4m3`, and every sum of `DEPTH` products of them exact in `f32`.
fn value(i: usize) -> f32 {
    ((i * 7 + i / 5) % 13) as f32 / 4.0 - 1.5
}

/// How `a`'s tiles reach the kernel.
#[derive(Clone, Copy, Debug)]
enum Tiles {
    /// Read off the binding's tiling, so the kernel knows each tile's extent.
    Stated,
    /// A spec naming the buffer's pieces alone, so the kernel reads every extent off the buffer.
    Unstated,
}

/// Launch the product over `a` tiled as `tiles` says and compare every element with the
/// reference; a skip where the device has no 8-bit float instruction.
fn check(tiles: Tiles) {
    let client = cubecl::test_device().client();
    let Some((cols, _)) = fp8_instruction(&client) else {
        ValidationResult::Skipped(
            "the device offers no e4m3 instruction summing in f32 over 16 rows".into(),
        )
        .enforce();
        return;
    };
    let rows = TILE_ROWS;
    let fp8 = ElemType::Float(FloatKind::E4M3);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, ROWS), (N, cols), (K, DEPTH)]),
            Levels::leaf(&[(M, rows), (N, cols), (K, WINDOW)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a: Vec<f32> = (0..ROWS * DEPTH).map(value).collect();
    let b: Vec<f32> = (0..cols * DEPTH).map(|i| value(i * 3 + 1)).collect();
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|&v| e4m3::from_f32(v).to_bits())
            .collect::<Vec<u8>>()
    };
    let mut stored = vec![0f32; ROWS * DEPTH];
    for (at, &v) in a.iter().enumerate() {
        let (m, k) = (at / DEPTH, at % DEPTH);
        let tile = m / TILE_ROWS * (DEPTH / TILE_K) + k / TILE_K;
        stored[tile * TILE_ROWS * TILE_K + m % TILE_ROWS * TILE_K + k % TILE_K] = v;
    }
    let shape = [ROWS / TILE_ROWS, DEPTH / TILE_K, TILE_ROWS, TILE_K];
    let strides = [DEPTH * TILE_ROWS, TILE_ROWS * TILE_K, TILE_K, 1];
    let a_handle = client.create_from_slice(&bits(&stored));
    let b_handle = client.create_from_slice(&bits(&b));
    let c = TestInput::builder(client.clone(), Shape::from(vec![ROWS, cols]))
        .zeros()
        .generate_without_host_data();

    let a_arg = match tiles {
        Tiles::Stated => {
            let metadata = Metadata::new(shape, strides)
                .with_tiling(Tiling::new(&[2, 2]).expect("two pieces a dimension"))
                .expect("the tiling lists the buffer's axes");
            launcher
                .arg(
                    TensorHandle::from_metadata(a_handle, metadata, fp8).binding(),
                    fp8,
                )
                .dimensions(&[M, K])
                .vectorize(TILE_K)
                .build()
                .expect("a load of a tile's row of `k` lies in one piece")
                .arg()
        }
        Tiles::Unstated => {
            let mut spec = TileSpec::new(Projection::tiled(
                &[M, K],
                &[(M, 0), (K, 0), (M, 1), (K, 1)],
            ));
            spec.storage = Storage::Tiled(None);
            TileTensorLaunch::new(
                TensorHandle::new(a_handle, shape.to_vec(), strides.to_vec(), fp8)
                    .binding()
                    .into_tensor_arg(),
                fp8,
                TILE_K,
                spec,
            )
        }
    };
    tiled_fp8_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_arg,
        TileTensorLaunch::new(
            TensorHandle::new(b_handle, vec![cols, DEPTH], vec![DEPTH, 1], fp8)
                .binding()
                .into_tensor_arg(),
            fp8,
            TILE_K,
            TileSpec::direct(&[N, K]),
        ),
        TileTensorLaunch::new(
            c.clone().binding().into_tensor_arg(),
            c.dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for i in 0..ROWS {
        for j in 0..cols {
            let want: f32 = (0..DEPTH)
                .map(|l| a[i * DEPTH + l] * b[j * DEPTH + l])
                .sum();
            assert_eq!(got.get_f32(&[i, j]), want, "({i}, {j}), tiles {tiles:?}");
        }
    }
}

#[test]
fn an_fp8_lhs_stored_in_tiles_contracts_through_manual_loads() {
    check(Tiles::Stated);
}

#[test]
fn a_tiled_operand_with_unstated_tile_extents_contracts_through_manual_loads() {
    check(Tiles::Unstated);
}
