//! An NVFP4 rhs in tile order, `[n / 16][k / 16][16 columns][16 values]`, contracted into a
//! register block a block of `K` a step: a load is two words along `K` by two columns, the leaf's
//! step spans one block (an axis of one beside the columns), and its scales are a packed factor
//! one word wide, read between the values' loads. The answer is the host's.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::quant::scheme::ScaleDtype;
use cubecl::zspace::shape;
use cubecl::{bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle};
use cubecl_common::e2m1;
use cubecl_tile::kind::Field;
use cubecl_tile::layout::StorageLevels;
use cubecl_tile::*;

use super::column_reads::offset;
use crate::tile::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// `K` as blocks of one scale, and the position inside one.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// NVFP4's block: sixteen values under one scale, two words of eight.
const BLOCK: usize = 16;
const PER_WORD: usize = 8;
/// Columns a tile holds side by side.
const TILE: usize = 16;

/// `c = x · (W ⊗ s)`, the weight read as many words a load as it is bound wide and its scales a
/// packed `f32` one word a load, into a register block that lives across the walk over `K` and
/// drains once.
#[cube(launch)]
fn tile_order_gemv<E: Numeric>(
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
                config: RegisterBlock::new(256)
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

/// Two words along `K` by two columns a load, a leaf of one tile's sixteen columns by one block.
#[test]
fn tile_order_reads_two_words_by_two_columns_a_load() {
    let (cols, blocks) = (2 * TILE, 8);
    let client = cubecl::test_device().client();

    let axes = [N, KB, KI];
    let storage = StorageLevels::new(&[(KI, PER_WORD)])
        .tile(&[(KI, BLOCK / PER_WORD), (N, 2)])
        .tile(&[(N, TILE / 2)])
        .grid(&[KI, KB, N]);
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
    let s_tensor = TensorHandle::new_contiguous(
        vec![cols, blocks],
        client.create(Bytes::from_elems(s.clone())),
        u32::elem_type_native(),
    );
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks), (KI, BLOCK)]),
            Levels::leaf(&[(N, TILE), (KB, 1)])
                .walk_every(&[KB])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    );

    let mut w_spec = TileSpec::new(Projection::tiled(&axes, &labels)).packed(QuantValue::E2M1);
    w_spec.stored_tiles = storage.tiles().to_vec();
    let s_spec = TileSpec::direct(&[N, KB]).packed(Field::of_scale(ScaleDtype::F32));
    // A load: two words along `K` by two columns, four words.
    let load_words = 2 * BLOCK / PER_WORD;
    tile_order_gemv::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_tensor.binding().into_tensor_arg(),
            dtype,
            BLOCK,
            TileSpec::direct(&[M, KB, KI]),
        ),
        TileArgLaunch::new(
            w_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            load_words,
            w_spec,
        ),
        TileArgLaunch::new(
            s_tensor.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            s_spec,
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

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
            "column {n}: got {have}, want {want}"
        );
    }
}
