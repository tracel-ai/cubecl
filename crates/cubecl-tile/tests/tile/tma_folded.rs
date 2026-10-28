//! A box the TMA engine loads through a descriptor whose dims are not the tile's own axes
//! ([`TileArgLaunch::tensor_map_folded`]): a `rows × k` matrix read over the block-split axes
//! `(M, KB, KI)` a packed product states its float input over, the descriptor over the matrix as
//! it lies, each position landing at the column `KB · block + KI`. Each cube lands its box in
//! shared memory and copies it out over the same axes.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::features::Tma;
use cubecl::prelude::*;
use cubecl::zspace::{shape, strides};
use cubecl_tile::launch::{TmaCoordinates, TmaTerm};
use cubecl_tile::*;

const M: Axis = Axis(0);
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

/// Rows and values of `k` of the matrix, and the block `k` is split in.
const ROWS: usize = 32;
const K: usize = 128;
const BLOCK: usize = 16;
/// The box one cube loads: rows, then blocks of `k`.
const BOX: (usize, usize) = (16, 2);

#[cube(launch)]
fn load_by_tma(input: &TileArg<'_>, out: &TileArg<'_>, space: Partitioning) {
    let src = input.tile(&space);
    let dst = out.tile(&space);
    for cube in &space {
        let mut dst_box = dst.at(&cube);
        let mut staged = dst_box.smem_box();
        staged.copy_from(&src.at(&cube));
        dst_box.copy_from(&staged);
    }
}

/// Every cell of the matrix holds its own index, loaded box by box through the folded descriptor
/// and copied out: a block landed at the wrong column, or a box a row off, shows up as another
/// cell's index.
#[test]
fn a_block_split_box_lands_through_a_folded_descriptor() {
    let client = cubecl::test_device().client();
    if !client.features().tma.contains(Tma::Base) {
        eprintln!("this device offers no bulk tensor copy, so no box is loaded by TMA");
        return;
    }
    let dtype = f32::elem_type_native();
    let input = TestInput::builder(client.clone(), shape![ROWS, K])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![ROWS, K])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let blocks = K / BLOCK;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, ROWS), (KB, blocks), (KI, BLOCK)]),
            Levels::leaf(&[(M, BOX.0), (KB, BOX.1), (KI, BLOCK)])
                .cubes(&[M, KB])
                .build(),
        ),
        Form::Static,
    );
    // The output over the same axes, the matrix's bytes restated: `KB` steps a block.
    let mut restated = output.clone().binding();
    restated.shape = shape![ROWS, blocks, BLOCK];
    restated.strides = strides![K, BLOCK, 1];
    let dst = launcher
        .arg(restated, dtype)
        .axes(&[M, KB, KI])
        .vectorize(1)
        .build()
        .unwrap();
    // The descriptor reads the matrix as one of a batch, `[1, rows, k]`, a box of the cube's rows
    // by its blocks' values.
    let matrix = unsafe {
        TensorArg::from_raw_parts(
            input.handle.clone(),
            strides![ROWS * K, K, 1],
            shape![1, ROWS, K],
        )
    };
    let map = TensorMapArg::new(
        TiledArgs {
            tile_size: shape![1, BOX.0, BOX.1 * BLOCK],
        },
        matrix,
        dtype,
    );
    let coordinates = TmaCoordinates {
        dims: vec![
            vec![],
            vec![TmaTerm::axis(0)],
            vec![TmaTerm::axis(1).times(BLOCK as u32), TmaTerm::axis(2)],
        ],
    };
    let src = TileArgLaunch::tensor_map_folded(
        map,
        &[M, KB, KI],
        &[ROWS as u32, blocks as u32, BLOCK as u32],
        coordinates,
    );
    load_by_tma::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        src,
        dst.arg(),
        launcher.partitioning_arg(),
    );
    let loaded = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..K {
            assert_eq!(
                loaded.get_f32(&[row, col]),
                (row * K + col) as f32,
                "the folded box landed the wrong value at [{row}, {col}]"
            );
        }
    }
}
