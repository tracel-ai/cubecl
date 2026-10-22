//! Dequantization on its own: `out = weights ⊗ scales`, no contraction anywhere.
//!
//! A quantized tensor is values stored small beside a coarser tensor of scales. Decoding it is not
//! a matmul verb and not a quantization feature: it is the elementwise product of two tiles, one
//! spanning fewer axes than the other and so spreading each value across the axes it omits.
//!
//! Proven here with the axes doing all the work. `COL` is spelled `(CB, CI)`: which block of
//! columns, and where inside it. The weights address both and the scales address `CB` alone, so
//! one scale per block is a fact about which axes the operand distinguishes; nothing divides.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{
    bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

const ROW: Axis = Axis(0);
/// Which block of columns, and where inside it. Together they are the column axis.
const CB: Axis = Axis(1);
const CI: Axis = Axis(2);

/// `out = weights ⊗ scales`. The weights arrive packed and unpack at the read; the scales
/// broadcast over the axis they omit. Neither fact is stated here; both are the operands'.
#[cube(launch)]
fn dequantize(weights: &TileArg<'_>, scales: &TileArg<'_>, out: &TileArg<'_>, space: Partitioning) {
    let weights = weights.tile(&space);
    let scales = scales.tile(&space);
    let mut out = out.tile(&space);
    out.product(&weights, &scales);
}

#[test]
fn a_packed_tensor_decodes_against_its_scales() {
    let (rows, blocks, inside, scale_units) = (4, 4, 8, 4);
    let cols = blocks * inside;
    let field = QuantValue::Q4S;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    // Every value a signed 4-bit field represents, cycling.
    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + (i as i32 % span))
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = w
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    // Distinct per `(row, block)`, halves so the reference is exact.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    // The scales are an operand like the others, and the axis they omit is the whole statement
    // that one of their values covers a block of columns.
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, rows), (CB, blocks), (CI, inside)]),
            Levels::leaf(&[(ROW, rows), (CB, blocks), (CI, inside)])
                .walk_every(&[ROW, CB, CI])
                .build(),
        ),
        Form::Static,
    );

    // Shape and strides count values; the packing says how many share a stored word.
    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // `COL` is one physical dim that `(CB, CI)` partition, so each operand that spans both says
    // so; the scales span only `CB` and address it as it stands.
    let split = || {
        Projection::new(
            &[ROW, CB, CI],
            &[
                PhysicalAxisMap::of(ROW),
                PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
            ],
        )
    };
    let launcher = implied(
        &client,
        Partitioning::new(
            launch.space().clone(),
            launch.partitioning().levels().to_vec(),
        ),
        Form::Dynamic,
    );
    let w_op = launcher
        .arg(w_tensor.clone().binding(), w_tensor.dtype)
        .gathered(split())
        .packed(field)
        .vectorize(factor)
        .build()
        .unwrap();
    // Four scales per read: one read covers four blocks of columns, and each of its units is
    // taken by the run of values that block holds.
    let s_op = launcher
        .arg(s_tensor.binding(), dtype)
        .axes(&[ROW, CB])
        .vectorize(scale_units)
        .build()
        .unwrap();
    let out_op = launcher
        .arg(out.clone().binding(), dtype)
        .gathered(split())
        .vectorize(factor)
        .build()
        .unwrap();

    dequantize::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w_op.arg(),
        s_op.arg(),
        out_op.arg(),
        launch.partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for row in 0..rows {
        for col in 0..cols {
            let want = w[row * cols + col] as f32 * s[row * blocks + col / inside];
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-6,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}
