//! Stages filled by the copy engine ([`Delivery::AsyncPerUnit`], `cp.async`) against the same
//! kernels filled by load and store ([`Delivery::SyncPerUnit`]): the delivery is a property of the
//! operand, so each kernel here is launched twice, unchanged, and the two answers must be the
//! reference's and each other's.
//!
//! A device without the copy engine skips: the async half has nothing to run on.

use crate::support::{
    HostData, HostDataType, TestInput, TileInput, ValidationResult, assert_equals_approx,
};
use cubecl::{
    bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_tile::kind::Boundary;
use cubecl_tile::launch::{BoundaryPolicy, Delivery};
use cubecl_tile::*;

use super::references;
use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The software instruction the ring's leaf runs.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

const DELIVERIES: [Delivery; 2] = [Delivery::SyncPerUnit, Delivery::AsyncPerUnit];

/// Whether this device has the copy engine; a test without it is skipped, not passed.
fn has_copy_engine(client: &Client) -> bool {
    if client.properties().features.copy_async {
        return true;
    }
    ValidationResult::Skipped("the device has no async copy".to_string()).enforce();
    false
}

/// `c = a · b` over a K walk whose blocks are staged `depth` deep and filled one region ahead of
/// the one contracted: the pipelined schedule, its slots published through the barrier.
#[cube(launch)]
fn ring_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] depth: usize,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();

    let walk = space.walk();
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth, None);
    stages.pipelined(walk, |slot, region| {
        let c_block = c.at(region);
        slot.consume(|a_s, b_s| {
            for cell in region {
                let mut c_cell = c_block
                    .at(&cell)
                    .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
                c_cell.mma(&a_s.at(&cell), &b_s.at(&cell));
            }
        });
    });
}

/// Each region of `src` staged by a blocking `copy_from`, then copied out of the stage: the fill
/// and the wait a kernel without a pipeline runs.
#[cube(launch)]
fn staged_copy(
    src: &TileArg<'_>,
    dst: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let src = src.tile(&space);
    let dst = dst.tile(&space);
    let mut stage = src.stage(comptime!(level.clone()), StageStorage::Strided);
    for region in space.over(&level) {
        stage.copy_from(&src.at(&region));
        sync_cube();
        let mut out = dst.at(&region);
        out.copy_from(&stage);
        // The stage is refilled next region, once every unit has read it.
        sync_cube();
    }
}

/// [`staged_copy`] over packed words: the stage keeps them packed, as they lie, and the copy out
/// of it unpacks.
#[cube(launch)]
fn staged_packed_copy(
    src: &TileArg<'_>,
    dst: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let src = src.tile(&space);
    let dst = dst.tile(&space);
    let mut stage = src.stage(comptime!(level.clone()), StageStorage::Strided);
    for region in space.over(&level) {
        stage.copy_from(&src.at(&region));
        sync_cube();
        let mut out = dst.at(&region);
        out.copy_from(&stage);
        sync_cube();
    }
}

fn ring_matmul_with(
    m: usize,
    n: usize,
    k: usize,
    block_k: usize,
    depth: usize,
    delivery: Delivery,
) -> HostData {
    let client = cubecl::test_device().client();
    let tile = 4usize;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, block_k / tile)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    ring_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.bind(&launcher).delivery(delivery).build().unwrap().arg(),
        b.bind(&launcher).delivery(delivery).build().unwrap().arg(),
        c.arg(&launcher, 1),
        launcher.partitioning_arg(),
        depth,
    );
    HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32)
}

fn check_ring_matmul(m: usize, n: usize, k: usize, block_k: usize, depth: usize) {
    let client = cubecl::test_device().client();
    if !has_copy_engine(&client) {
        return;
    }
    let (_, expected) = TestInput::builder(client, shape![m / 4, n / 4, 4, 4])
        .custom(references::tiled_matmul(m, n, k, 4))
        .generate_with_f32_host_data();
    let [sync, copied] =
        DELIVERIES.map(|delivery| ring_matmul_with(m, n, k, block_k, depth, delivery));
    assert_equals_approx(&sync, &expected, 1e-3).enforce();
    assert_equals_approx(&copied, &expected, 1e-3).enforce();
    assert_equals_approx(&copied, &sync, 0.0).enforce();
}

/// The pipelined walk, two slots deep: each region is copied in while the one before it is
/// contracted, and the slot's barrier publishes it only once its copies have landed.
#[test]
fn an_async_ring_walk_double_buffered() {
    check_ring_matmul(8, 8, 16, 4, 2);
}

/// Three slots over a walk the depth does not divide, so the drain publishes a slot the prologue
/// filled on a different lap.
#[test]
fn an_async_ring_walk_triple_buffered_over_an_odd_walk() {
    check_ring_matmul(8, 8, 20, 4, 3);
}

/// `rows` of a `rows_space`-row space are real; the rest overhang the tensor and read as zero.
fn staged_copy_with(
    rows: usize,
    rows_space: usize,
    cols: usize,
    tile_rows: usize,
    delivery: Delivery,
) -> Vec<f32> {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let (src, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .arange()
        .generate_with_f32_host_data();
    // Poisoned, so a zero in the overhang is one the fill wrote.
    let dst = TestInput::builder(client.clone(), shape![rows_space, cols])
        .dtype(dtype)
        .uniform(3, 10.0, 100.0)
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows_space), (N, cols)]),
            Levels::leaf(&[(M, tile_rows), (N, cols)])
                .walk_every(&[M, N])
                .build(),
        ),
        Form::Static,
    );
    let masked = rows != rows_space;
    let src_spec = TileSpec::direct(&[M, N]).delivery(delivery);
    let src_spec = match masked {
        true => src_spec.boundary(BoundaryPolicy::Every(Boundary::Zero)),
        false => src_spec,
    };
    staged_copy::launch(
        &client,
        CubeCount::new_single(),
        launcher.cube_dim(),
        TileArgLaunch::new(src.binding().into_tensor_arg(), dtype, 1, src_spec),
        TileArgLaunch::new(
            dst.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );
    let got = HostData::from_tensor_handle(&client, dst, HostDataType::F32);
    (0..rows_space * cols)
        .map(|i| got.get_f32(&[i / cols, i % cols]))
        .collect()
}

fn check_staged_copy(rows: usize, rows_space: usize, cols: usize, tile_rows: usize) {
    let client = cubecl::test_device().client();
    if !has_copy_engine(&client) {
        return;
    }
    let want: Vec<f32> = (0..rows_space * cols)
        .map(|i| if i < rows * cols { i as f32 } else { 0.0 })
        .collect();
    for delivery in DELIVERIES {
        let got = staged_copy_with(rows, rows_space, cols, tile_rows, delivery);
        assert_eq!(got, want, "{delivery:?}");
    }
}

/// A blocking `copy_from` into a stage: the copy waits on a barrier of its own before the stage
/// is read.
#[test]
fn an_async_copy_from_waits_for_its_stage() {
    check_staged_copy(16, 16, 16, 4);
}

/// The last region overhangs the tensor by two rows: the engine is handed an empty run for them
/// and zero-fills, as a masked load would have written zeros.
#[test]
fn an_async_copy_zero_fills_past_the_edge() {
    check_staged_copy(14, 16, 16, 4);
}

fn staged_packed_copy_with(
    values: &[i32],
    rows: usize,
    cols: usize,
    tile_rows: usize,
    delivery: Delivery,
) -> Vec<f32> {
    let field = QuantValue::Q8S;
    let bits = field.size_bits();
    let factor = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();

    let client = cubecl::test_device().client();
    let input = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols)]),
            Levels::leaf(&[(M, tile_rows), (N, cols)])
                .walk_every(&[M, N])
                .build(),
        ),
        Form::Static,
    );
    staged_packed_copy::launch(
        &client,
        CubeCount::new_single(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            input.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::direct(&[M, N]).packed(field).delivery(delivery),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    (0..rows * cols)
        .map(|i| got.get_f32(&[i / cols, i % cols]))
        .collect()
}

/// A packed stage is its words as they lie, so the engine copies them unchanged; the copy out of
/// the stage unpacks four 8-bit values from each.
#[test]
fn an_async_copy_stages_packed_words() {
    let (rows, cols, tile_rows) = (8, 16, 4);
    let client = cubecl::test_device().client();
    if !has_copy_engine(&client) {
        return;
    }
    let values: Vec<i32> = (0..rows * cols)
        .map(|i| -128 + (i as i32 * 7) % 256)
        .collect();
    let want: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    for delivery in DELIVERIES {
        let got = staged_packed_copy_with(&values, rows, cols, tile_rows, delivery);
        assert_eq!(got, want, "{delivery:?}");
    }
}
