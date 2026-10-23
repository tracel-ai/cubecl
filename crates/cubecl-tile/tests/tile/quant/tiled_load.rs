//! NVFP4 values stored in tiles and loaded a rectangle at a time: each load is four words, two
//! along `K` and two columns, 16 values of one scale block by 2 columns, and the decoding copy
//! lands every value where it belongs under its own column's scale.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::zspace::shape;
use cubecl::{bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle};
use cubecl_common::e2m1;
use cubecl_tile::{
    Axis, Launcher, Level, Levels, Partitioning, Projection, Space, StageStorage, TileArg,
    TileArgLaunch, TileSpec,
    launch::Grid,
    layout::{StorageLevels, StoragePartitioning, VectorTile},
};

use crate::tile::uncut;

const N: Axis = Axis(0);
/// `K` as blocks of one scale make it: which block, and where inside it.
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

/// `output = unpack(values) ⊗ scales`, the values read as many words a load as they are bound
/// wide.
#[cube(launch)]
fn tiled_scaled_copy(
    values: &TileArg<'_>,
    scales: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
) {
    let values = values.tile(&space);
    let scales = scales.tile(&space);
    let mut output = output.tile(&space);
    output.dequantize_from(&values.mul(&scales));
}

/// `stage = values` as the words they are, then `output = unpack(stage) ⊗ scales`: the weight
/// staged packed over `level`'s region, decoded out of the stage.
#[cube(launch)]
fn tiled_packed_stage(
    values: &TileArg<'_>,
    scales: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let values = values.tile(&space);
    let scales = scales.tile(&space);
    let mut output = output.tile(&space);
    let mut stage = values.stage(level, StageStorage::Strided);
    stage.copy_from(&values);
    sync_cube();
    output.dequantize_from(&stage.mul(&scales));
}

/// Where the weight is decoded from.
#[derive(Clone, Copy, Debug)]
enum Read {
    /// Straight from its stored tiles.
    Stored,
    /// From a stage its words were copied into as they lie.
    PackedStage,
}

/// A `[n, kb, ki]` NVFP4 weight stored a word of 8 along `KI`, then 2 words along `KI` by 2
/// columns, the next such tile along `KB`: every load of four words is 16 values of one block by
/// 2 columns, each column under its own scale. Copied out four values a line along `KI`, every
/// value is its code times its column's block scale.
#[test]
fn a_rectangle_load_decodes_each_value_under_its_own_scale() {
    let storage = StorageLevels::new(&[(KI, 8)])
        .tile(&[(KI, 2), (N, 2)])
        .grid(&[KB, N, KI]);
    assert_eq!(
        VectorTile::new(storage.tiles(), KI, 32).unwrap().extents(),
        &[(KI, 16), (N, 2)]
    );
    check_tiled(storage, 4, 4, Read::Stored);
}

/// NVFP4 in tile order as a gemm reads it, a column's block a load in tiles of sixteen columns,
/// staged as its words and decoded out of the stage: the stage holds every word where the
/// stored tiles put it.
#[test]
fn a_tile_order_weight_staged_as_its_words_decodes_out_of_the_stage() {
    let storage = StorageLevels::new(&[(KI, 8)])
        .tile(&[(KI, 2), (N, 16)])
        .grid(&[KI, KB, N]);
    check_tiled(storage, 32, 2, Read::PackedStage);
}

/// `cols` columns of two blocks of sixteen stored as `storage` states, read `words_per_load` words a load
/// and decoded as `read` says, against the host.
fn check_tiled(storage: StoragePartitioning, cols: usize, words_per_load: usize, read: Read) {
    let (blocks, block) = (2, 16);
    let client = cubecl::test_device().client();
    let axes = [N, KB, KI];
    let geometry = storage
        .physical(&[(N, cols), (KB, blocks), (KI, block)])
        .unwrap();
    let labels = geometry.labels(&axes);
    // Every code, cycled, laid down where the storage puts its value.
    let code = |n: usize, kb: usize, ki: usize| ((n * blocks + kb) * block + ki) % 16;
    let offset = |coords: [usize; 3]| {
        let mut rest = coords;
        let mut offset = 0;
        for d in (0..labels.len()).rev() {
            let a = axes.iter().position(|&a| a == labels[d]).unwrap();
            let extent = geometry.shape()[d];
            offset += (rest[a] % extent) * geometry.strides()[d];
            rest[a] /= extent;
        }
        offset
    };
    let values = cols * blocks * block;
    let mut words = vec![0u32; values / 8];
    for n in 0..cols {
        for kb in 0..blocks {
            for ki in 0..block {
                let at = offset([n, kb, ki]);
                words[at / 8] |= (code(n, kb, ki) as u32) << ((at % 8) * 4);
            }
        }
    }
    let w_tensor = TensorHandle::new(
        client.create(Bytes::from_elems(words)),
        geometry.shape().to_vec(),
        geometry.strides().to_vec(),
        u32::elem_type_native(),
    );
    // Halves, so the reference is exact.
    let s: Vec<f32> = (0..cols * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let dtype = f32::elem_type_native();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![cols, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![cols, blocks, block])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let mut w_spec = TileSpec::new(Projection::tiled(&axes, &labels)).packed(QuantValue::E2M1);
    w_spec.stored_tiles = storage.tiles().to_vec();
    let space = Space::new(&[(N, cols), (KB, blocks), (KI, block)]);
    let (count, dim) = (CubeCount::new_single(), CubeDim::new_single());
    // Each arm launches its own kernel, so each binds its own arguments.
    match read {
        Read::Stored => tiled_scaled_copy::launch(
            &client,
            count,
            dim,
            TileArgLaunch::new(
                w_tensor.binding().into_tensor_arg(),
                u32::elem_type_native(),
                words_per_load,
                w_spec,
            ),
            TileArgLaunch::new(
                s_tensor.binding().into_tensor_arg(),
                dtype,
                1,
                TileSpec::direct(&[N, KB]),
            ),
            TileArgLaunch::new(
                out.clone().binding().into_tensor_arg(),
                dtype,
                4,
                TileSpec::direct(&axes),
            ),
            uncut(&client, &space, &space).partitioning_arg(),
        ),
        Read::PackedStage => {
            // One cube over the whole box: the region the stage holds.
            let partitioning = Partitioning::new(
                space.clone(),
                Levels::leaf(&[(N, cols), (KB, blocks), (KI, block)])
                    .cubes(&[N])
                    .build(),
            );
            let level = partitioning.level(0);
            let single = Grid::Stated {
                cube_count: CubeCount::new_single(),
                cube_dim: CubeDim::new_single(),
            };
            let launcher = Launcher::new(&client, partitioning, &space, single).unwrap();
            tiled_packed_stage::launch(
                &client,
                count,
                dim,
                TileArgLaunch::new(
                    w_tensor.binding().into_tensor_arg(),
                    u32::elem_type_native(),
                    words_per_load,
                    w_spec,
                ),
                TileArgLaunch::new(
                    s_tensor.binding().into_tensor_arg(),
                    dtype,
                    1,
                    TileSpec::direct(&[N, KB]),
                ),
                TileArgLaunch::new(
                    out.clone().binding().into_tensor_arg(),
                    dtype,
                    4,
                    TileSpec::direct(&axes),
                ),
                launcher.partitioning_arg(),
                level,
            )
        }
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for n in 0..cols {
        for kb in 0..blocks {
            for ki in 0..block {
                let want = e2m1::from_bits(code(n, kb, ki) as u8).to_f32() * s[n * blocks + kb];
                let have = got.get_f32(&[n, kb, ki]);
                assert!(
                    (have - want).abs() < 1e-6,
                    "{read:?} at (n {n}, kb {kb}, ki {ki}): got {have}, want {want}"
                );
            }
        }
    }
}
