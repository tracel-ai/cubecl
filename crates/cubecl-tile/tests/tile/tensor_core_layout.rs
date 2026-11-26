//! A matrix stored for a tensor core: `[m, k]` in tiles of sixteen by sixteen, each listed as
//! `[m%8, k/2%4, k/8%2, m/8%2, k%2]`, so neither dimension's pieces are listed coarsest first.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::zspace::{Tiling, metadata::Metadata};
use cubecl::{bytes::Bytes, client::Client, prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_tile::launch::Decline;
use cubecl_tile::layout::{LineMisfit, TileMisfit};
use cubecl_tile::{Dimension, Partitioning, Space, TileTensor};

use super::uncut;

const M: Dimension = Dimension(0);
const K: Dimension = Dimension(1);

const ROWS: usize = 48;
const COLUMNS: usize = 80;

/// Every value of `input` into `output` through their views, a vector of `K` at a time.
#[cube(launch)]
fn copy_lines<E: Numeric, W: Size>(
    input: &TileTensor<'_>,
    output: &mut TileTensor<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let mut output = output.tile_mut(&space);
    let r = input.view::<E, W>();
    let mut w = output.view_mut::<E, W>();
    let shape = r.shape();
    let rows = shape[0];
    let lines = shape[1];
    for i in 0..rows {
        for j in 0..lines {
            let mut pos = CoordsDyn::new();
            pos.push(i);
            pos.push(j);
            w.write(pos.clone(), r.read(pos));
        }
    }
}

/// Where the value at `m`, `k` lies: its tile, then its place in it.
fn offset(m: usize, k: usize) -> usize {
    let tile = m / 16 * (COLUMNS / 16) + k / 16;
    tile * 256 + m % 8 * 32 + k / 2 % 4 * 8 + k / 8 % 2 * 4 + m / 8 % 2 * 2 + k % 2
}

/// A `[m, k]` matrix of `values` stored for a tensor core, back to back.
fn stored(client: &Client, values: Vec<f32>) -> TensorHandle {
    let metadata = Metadata::new(
        [ROWS / 16, COLUMNS / 16, 8, 4, 2, 2, 2],
        [COLUMNS / 16 * 256, 256, 32, 8, 4, 2, 1],
    )
    .with_tiling(
        Tiling::from_physical_dims(&[&[0, 5, 2][..], &[1, 4, 3, 6]])
            .expect("each dim should be one piece"),
    )
    .expect("the tiling should list the buffer's dims");
    let handle = client.create(Bytes::from_elems(values));
    TensorHandle::from_metadata(handle, metadata, f32::elem_type_native())
}

/// `K` read `width` values a vector, each value lands where its coordinates say.
fn check_read(width: usize) {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let mut values = vec![0f32; ROWS * COLUMNS];
    for m in 0..ROWS {
        for k in 0..COLUMNS {
            values[offset(m, k)] = (m * COLUMNS + k) as f32;
        }
    }
    let input = stored(&client, values);
    let output = TestInput::builder(client.clone(), shape![ROWS, COLUMNS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Space::new(&[(M, ROWS), (K, COLUMNS)]);
    let launch = uncut(&client, &space, &space);
    let bind = |binding| {
        launch
            .arg(binding, dtype)
            .dimensions(&[M, K])
            .vectorize(width)
            .build()
            .expect("a pair of K should lie back to back")
            .arg()
    };
    copy_lines::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        width,
        bind(input.binding()),
        bind(output.clone().binding()),
        launch.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for m in 0..ROWS {
        for k in 0..COLUMNS {
            assert_eq!(got.get_f32(&[m, k]), (m * COLUMNS + k) as f32, "({m}, {k})");
        }
    }
}

#[test]
fn a_tensor_core_layout_is_read_a_value_at_a_time() {
    check_read(1);
}

#[test]
fn a_tensor_core_layout_is_read_in_its_finest_piece() {
    check_read(2);
}

#[test]
fn a_tensor_core_layout_is_written() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let (input, _) = TestInput::builder(client.clone(), shape![ROWS, COLUMNS])
        .dtype(dtype)
        .arange()
        .generate_with_f32_host_data();
    let output = stored(&client, vec![0f32; ROWS * COLUMNS]);
    let written = output.handle.clone();

    let space = Space::new(&[(M, ROWS), (K, COLUMNS)]);
    let launch = uncut(&client, &space, &space);
    let bind = |binding| {
        launch
            .arg(binding, dtype)
            .dimensions(&[M, K])
            .build()
            .expect("a value should always be read")
            .arg()
    };
    copy_lines::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        1,
        bind(input.binding()),
        bind(output.binding()),
        launch.partitioning_arg(),
        dtype,
    );

    let bytes = client
        .read_one(written)
        .expect("the buffer should be read back");
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    for m in 0..ROWS {
        for k in 0..COLUMNS {
            assert_eq!(got[offset(m, k)], (m * COLUMNS + k) as f32, "({m}, {k})");
        }
    }
}

/// Four values of `K` are two pieces of two, and what lies beside the first is the other half of
/// the rows: no load of four holds them.
#[test]
fn a_vector_across_the_pieces_is_declined() {
    let client = cubecl::test_device().client();
    let space = Space::new(&[(M, ROWS), (K, COLUMNS)]);
    let launch = uncut(&client, &space, &space);
    let declined = launch
        .arg(
            stored(&client, vec![0f32; ROWS * COLUMNS]).binding(),
            f32::elem_type_native(),
        )
        .dimensions(&[M, K])
        .vectorize(4)
        .build()
        .err();
    assert_eq!(
        declined,
        Some(Decline::WidthNotServed {
            width: 4,
            why: LineMisfit::Tile(TileMisfit::RunsOut { wanted: K }),
        })
    );
}
