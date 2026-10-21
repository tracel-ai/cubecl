//! A box the TMA engine stores ([`TmaStored`]): each cube copies its box of the input into a box
//! in shared memory ([`Tile::smem_box`]), and the box leaves in one bulk store through a tensor
//! map ([`Tile::copy_from`]). The tensor's edge cuts boxes short on both axes, which the
//! descriptor clips, so a store past the edge would land on the cells the test reads as zero.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::features::Tma;
use cubecl::prelude::*;
use cubecl::zspace::shape;
use cubecl_tile::launch::{Destination, TmaBox, TmaOperand, TmaStored};
use cubecl_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
/// Rows and columns of the tensor: not whole boxes along either axis.
const ROWS: usize = 40;
const COLS: usize = 24;
/// The box one cube stores, and the descriptor moves.
const BOX: (usize, usize) = (16, 16);

#[cube(launch)]
fn store_by_tma<E: Float, O: Destination>(
    input: &TileArg<'_>,
    out: &O::Arg<E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(&space);
    let dst = O::tile::<E, Const<1>>(out, &space);
    for cube in &space {
        let mut dst_box = dst.at(&cube);
        let mut staged = dst_box.smem_box();
        staged.copy_from(&src.at(&cube));
        dst_box.copy_from(&staged);
    }
}

/// Every cell of the tensor holds its own index, stored box by box by the TMA engine, and only
/// those: a box clipped at the edge writes nothing past it, a box landed a row or a column off
/// shows up as another cell's index.
#[test]
fn a_box_stored_by_tma_lands_whole_and_clipped() {
    let client = cubecl::test_device().client();
    if !client.features().tma.contains(Tma::Base) {
        eprintln!("this device offers no bulk tensor copy, so no box is stored by TMA");
        return;
    }
    let dtype = f32::elem_type_native();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, BOX.0), (COL, BOX.1)])
                .cubes(&[ROW, COL])
                .build(),
        ),
        Form::Static,
    );
    // Bound through the launcher, which masks the reads of a box the edge cuts short.
    let src = launcher
        .arg(input.binding(), dtype)
        .axes(&[ROW, COL])
        .vectorize(1)
        .build()
        .unwrap();
    // The descriptor reads a matrix as one of a batch: `[1, rows, cols]`.
    let matrix = unsafe {
        TensorArg::from_raw_parts(
            output.handle.clone(),
            [ROWS * COLS, COLS, 1].into(),
            [1, ROWS, COLS].into(),
        )
    };
    let map = TensorMapArg::new(
        TiledArgs {
            tile_size: shape![1, BOX.0, BOX.1],
        },
        matrix,
        dtype,
    );
    let out = TmaOperand {
        map,
        axes: vec![ROW, COL],
        shape: TmaBox {
            rows: ROWS as u32,
            cols: COLS as u32,
            batch: None,
            transposed: false,
        },
    };
    store_by_tma::launch::<TmaStored>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        src.arg(),
        out.arg(),
        launcher.partitioning_arg(),
        dtype,
    );
    let stored = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                stored.get_f32(&[row, col]),
                (row * COLS + col) as f32,
                "the box stored the wrong value at [{row}, {col}]"
            );
        }
    }
}
