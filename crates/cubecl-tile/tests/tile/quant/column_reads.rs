//! A packed rhs stored a word along `K` and then several columns, contracted into a register
//! block: each load brings one word of every column it holds, and the leaf splits it into their
//! runs along `K`. The answer is the untiled one, whatever the number of columns a load holds.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::zspace::shape;
use cubecl::{bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle};
use cubecl_common::e2m1;
use cubecl_tile::layout::StorageLevels;
use cubecl_tile::*;

use crate::tile::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// `K` as blocks of one scale, and the position inside one.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// NVFP4's block: sixteen values under one scale, two words of eight.
const BLOCK: usize = 16;
const PER_WORD: usize = 8;

/// `c = x · (W ⊗ s)`, the weight and the scales each read as many as they are bound wide a load,
/// into a register block that lives across the walk over `K` and drains once.
#[cube(launch)]
fn column_one_row_product<E: Numeric>(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(&space);
    let values = w.tile(&space);
    let w = values.mul(&scale.tile(&space));
    let c = c.tile(&space);
    for cube in space {
        let x = x.at(&cube);
        let values = values.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        let acc = c.accumulator::<E>(
            &x,
            &values,
            comptime!(Instruction::Registers {
                config: RegisterBlock::new(64)
            }),
            Semiring::SUM_PROD,
        );
        for step in cube {
            let mut acc_s = acc.at(&step);
            acc_s.mma(&x.at(&step), &w.at(&step));
        }
        acc.drained_into(&c);
    }
}

/// [`column_gemv`] with the accumulator in memory: the output zeroed once, and every step's
/// block seeded from it and committed back ([`Tile::mma`]).
#[cube(launch)]
fn column_one_row_product_in_memory(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
) {
    let x = x.tile(&space);
    let w = w.tile(&space).mul(&scale.tile(&space));
    let c = c.tile(&space);
    for cube in &c {
        let mut c_w = c.at(&cube);
        c_w.zero();
    }
    for cube in space {
        let x = x.at(&cube);
        let w = w.at(&cube);
        let c = c.at(&cube);
        for step in cube {
            let mut c_s = c
                .at(&step)
                .accumulating(comptime!(RegisterBlock::new(64)), Semiring::SUM_PROD);
            c_s.mma(&x.at(&step), &w.at(&step));
        }
    }
}

/// Where the value at `coords` (along `axes`) sits in a buffer of `geometry`, its dims named
/// `labels`, outermost first: each coordinate spread over the dims its axis names, finest last.
pub(crate) fn offset(
    geometry: &Geometry,
    labels: &[Axis],
    axes: &[Axis],
    coords: &[usize],
) -> usize {
    let mut rest = coords.to_vec();
    let mut offset = 0;
    for d in (0..labels.len()).rev() {
        let a = axes.iter().position(|&a| a == labels[d]).unwrap();
        let extent = geometry.shape()[d];
        offset += (rest[a] % extent) * geometry.strides()[d];
        rest[a] /= extent;
    }
    offset
}

/// The scales `[N, KB]`, plain, or stored 2 blocks along `KB` then `columns` columns and read a
/// stored tile a load: the handle, how it is bound, and the load's width.
fn scales(
    client: &Client,
    s: &[f32],
    (cols, blocks): (usize, usize),
    columns: Option<usize>,
) -> (TensorHandle, TileSpec, usize) {
    let dtype = f32::elem_type_native();
    let Some(columns) = columns else {
        let handle = TensorHandle::new_contiguous(
            vec![cols, blocks],
            client.create(Bytes::from_elems(s.to_vec())),
            dtype,
        );
        return (handle, TileSpec::direct(&[N, KB]), 1);
    };
    let axes = [N, KB];
    let storage = StorageLevels::new(&[(KB, 2)])
        .tile(&[(N, columns)])
        .grid(&[N, KB]);
    let geometry = storage.physical(&[(N, cols), (KB, blocks)]).unwrap();
    let labels = geometry.labels(&axes);
    let mut stored = vec![0f32; cols * blocks];
    for n in 0..cols {
        for kb in 0..blocks {
            stored[offset(&geometry, &labels, &axes, &[n, kb])] = s[n * blocks + kb];
        }
    }
    let handle = TensorHandle::new(
        client.create(Bytes::from_elems(stored)),
        geometry.shape().to_vec(),
        geometry.strides().to_vec(),
        dtype,
    );
    let mut spec = TileSpec::new(Projection::tiled(&axes, &labels));
    spec.stored_tiles = storage.tiles().to_vec();
    (handle, spec, 2 * columns)
}

/// `columns` columns a load, `leaf_columns` a leaf: the weight `[N, KB, KI]` stored a word of 8
/// along `KI`, then `columns` columns, the rest `N`-major so a group of columns keeps its whole
/// `K` together. The scales plain, or stored `scale_columns` columns a load.
fn one_row_product_against_the_reference(
    columns: usize,
    leaf_columns: usize,
    in_memory: bool,
    scale_columns: Option<usize>,
) {
    let (cols, blocks) = (16, 8);
    let client = cubecl::test_device().client();

    let axes = [N, KB, KI];
    let storage = StorageLevels::new(&[(KI, PER_WORD)])
        .tile(&[(N, columns)])
        .grid(&[N, KB, KI]);
    let geometry = storage
        .physical(&[(N, cols), (KB, blocks), (KI, BLOCK)])
        .unwrap();
    let labels = geometry.labels(&axes);

    let code = |n: usize, kb: usize, ki: usize| ((n * 7 + kb * 3 + ki) % 16) as u32;
    let mut words = vec![0u32; cols * blocks * BLOCK / PER_WORD];
    for n in 0..cols {
        for kb in 0..blocks {
            for ki in 0..BLOCK {
                let at = offset(&geometry, &labels, &axes, &[n, kb, ki]);
                words[at / PER_WORD] |= code(n, kb, ki) << ((at % PER_WORD) * 4);
            }
        }
    }
    let w_tensor = TensorHandle::new(
        client.create(Bytes::from_elems(words)),
        geometry.shape().to_vec(),
        geometry.strides().to_vec(),
        u32::elem_type_native(),
    );

    let depth = blocks * BLOCK;
    let x: Vec<f32> = (0..depth).map(|i| (i % 5) as f32 - 2.0).collect();
    // Halves, so the reference is exact.
    let s: Vec<f32> = (0..cols * blocks)
        .map(|i| ((i % 6) as f32 + 1.0) / 2.0)
        .collect();
    let dtype = f32::elem_type_native();
    let (x_tensor, _) = TestInput::builder(client.clone(), shape![1, blocks, BLOCK])
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, s_spec, scale_width) = scales(&client, &s, (cols, blocks), scale_columns);
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks), (KI, BLOCK)]),
            Levels::leaf(&[(N, leaf_columns), (KB, 2)])
                .walk_every(&[KB])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    );

    let mut w_spec = TileSpec::new(Projection::tiled(&axes, &labels)).packed(QuantValue::E2M1);
    w_spec.stored_tiles = storage.tiles().to_vec();
    // One statement of the arguments, for either kernel.
    macro_rules! launch {
        ($kernel:ident $(, $dtype:expr)?) => {
            $kernel::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                TileArgLaunch::new(
                    x_tensor.binding().into_tensor_arg(),
                    dtype,
                    PER_WORD,
                    TileSpec::direct(&[M, KB, KI]),
                ),
                TileArgLaunch::new(
                    w_tensor.binding().into_tensor_arg(),
                    u32::elem_type_native(),
                    columns,
                    w_spec,
                ),
                TileArgLaunch::new(
                    s_tensor.binding().into_tensor_arg(),
                    dtype,
                    scale_width,
                    s_spec,
                ),
                TileArgLaunch::new(
                    c.clone().binding().into_tensor_arg(),
                    dtype,
                    1,
                    TileSpec::direct(&[M, N]),
                ),
                launcher.partitioning_arg(),
                $($dtype,)?
            )
        };
    }
    match in_memory {
        true => launch!(column_one_row_product_in_memory),
        false => launch!(column_one_row_product, dtype),
    }

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..blocks)
            .flat_map(|kb| (0..BLOCK).map(move |ki| (kb, ki)))
            .map(|(kb, ki)| {
                let w = e2m1::from_bits(code(n, kb, ki) as u8).to_f32();
                x[kb * BLOCK + ki] * w * s[n * blocks + kb]
            })
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "{columns} columns a load, scales {scale_columns:?}, column {n}: got {have}, want {want}"
        );
    }
}

/// One column a load: the word alone, the path every untiled weight already takes.
#[test]
fn one_column_a_load_is_a_word_along_k() {
    one_row_product_against_the_reference(1, 4, false, None);
}

/// Two words a load, one of each column, split into their runs along `K`.
#[test]
fn two_columns_a_load_split_into_their_runs() {
    one_row_product_against_the_reference(2, 4, false, None);
}

/// Four words a load, the leaf's every column in one read.
#[test]
fn four_columns_a_load_cover_the_leaf() {
    one_row_product_against_the_reference(4, 4, false, None);
}

/// A leaf of eight columns reads two loads of four per step of `K`.
#[test]
fn a_leaf_wider_than_a_load_reads_several() {
    one_row_product_against_the_reference(4, 8, false, None);
}

/// The accumulator in memory takes the same loads the same way.
#[test]
fn an_accumulator_in_memory_splits_the_same_loads() {
    one_row_product_against_the_reference(4, 8, true, None);
}

/// The scales stored 2 blocks by 2 columns: one load a pair of columns' scales.
#[test]
fn scales_stored_across_columns_read_a_load_for_several() {
    one_row_product_against_the_reference(4, 4, false, Some(2));
}

/// Four columns' scales a load, beside four columns' values a load, both accumulators.
#[test]
fn scales_and_values_both_read_four_columns_a_load() {
    one_row_product_against_the_reference(4, 8, false, Some(4));
    one_row_product_against_the_reference(4, 8, true, Some(4));
}
