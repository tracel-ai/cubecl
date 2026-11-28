//! Dequantization on its own: `out = weights ⊗ scales`, no contraction anywhere.
//!
//! A quantized tensor is values stored small beside a coarser tensor of scales. Decoding it is not
//! a matmul verb and not a quantization feature: it is the elementwise product of two tiles, one
//! spanning fewer dimensions than the other and so spreading each value across the dimensions it
//! omits.
//!
//! Proven here with the dimensions doing all the work. `COL` is spelled `(CB, CI)`: which block of
//! columns, and where inside it. The weights address both and the scales address `CB` alone, so
//! one scale per block is a fact about which dimensions the operand distinguishes; nothing divides.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{
    bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

const ROW: Dimension = Dimension(0);
/// Which block of columns, and where inside it. Together they are the column dimension.
const CB: Dimension = Dimension(1);
const CI: Dimension = Dimension(2);

/// `out = weights ⊗ scales`. The weights arrive packed and unpack at the read; the scales
/// broadcast over the dimension they omit. Neither fact is stated here; both are the operands'.
#[cube(launch)]
fn dequantize(
    weights: &TileTensor<'_>,
    scales: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    space: Partitioning,
) {
    let weights = weights.tile(&space);
    let scales = scales.tile(&space);
    let mut out = out.tile_mut(&space);
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

    // The scales are an operand like the others, and the dimension they omit is the whole statement
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

    // `COL` is one physical axis that `(CB, CI)` partition, so each operand that spans both says
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
        .dimensions(&[ROW, CB])
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

/// `out = weights ⊗ scales` decoded into memory, as a fill decodes a stage.
#[cube(launch)]
fn dequantize_into(
    weights: &TileTensor<'_>,
    scales: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    space: Partitioning,
) {
    let weights = weights.tile(&space).mul(&scales.tile(&space));
    let mut out = out.tile_mut(&space);
    out.dequantize_from(&weights);
}

/// `e2m1` codes decoded into `e4m3` under `f32` block scales: each value is its code times its
/// block's scale, rounded once. Some scales are no power of two, whose mantissa a decode that moved
/// the codes' exponents would drop, and some so small that the products are subnormal `e4m3`
/// values, or round to one or to zero.
#[test]
fn e2m1_codes_decode_into_e4m3_under_any_scale() {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    const CODES_PER_WORD: usize = 8;
    const SCALES: [f32; 8] = [
        0.75,
        1.25,
        1.0,
        1.0 / (1 << 5) as f32,
        1.0 / (1 << 6) as f32,
        1.0 / (1 << 8) as f32,
        1.0 / (1 << 10) as f32,
        1.0 / (1 << 12) as f32,
    ];
    let (rows, blocks, inside) = (4, 2, 32);
    let cols = blocks * inside;
    let client = cubecl::test_device().client();
    let e4m3_type = ElemType::Float(cubecl::ir::FloatKind::E4M3);
    let f32_type = f32::elem_type_native();
    if !client.properties().supports_type(e4m3_type) {
        crate::support::ValidationResult::Skipped("device stores no e4m3".to_string()).enforce();
        return;
    }

    // Every code, cycling, eight a word, the lowest first.
    let codes: Vec<u32> = (0..rows * cols).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(CODES_PER_WORD)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &code)| acc | (code << (4 * j)))
        })
        .collect();
    let s = SCALES.to_vec();
    assert_eq!(s.len(), rows * blocks);

    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(f32_type)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(vec![0u8; rows * cols])),
        e4m3_type,
    );

    let split = || {
        Projection::new(
            &[ROW, CB, CI],
            &[
                PhysicalAxisMap::of(ROW),
                PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
            ],
        )
    };
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
        .packed(QuantValue::E2M1)
        .vectorize(CODES_PER_WORD)
        .build()
        .unwrap();
    let s_op = launcher
        .arg(s_tensor.binding(), f32_type)
        .dimensions(&[ROW, CB])
        .build()
        .unwrap();
    let out_op = launcher
        .arg(out.clone().binding(), e4m3_type)
        .gathered(split())
        .vectorize(CODES_PER_WORD)
        .build()
        .unwrap();
    dequantize_into::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w_op.arg(),
        s_op.arg(),
        out_op.arg(),
        launch.partitioning_arg(),
    );

    let got = client.read_one_unchecked_tensor(out.into_copy_descriptor());
    for (at, &byte) in got.iter().enumerate() {
        let code = codes[at] as usize;
        let sign = if code & 8 != 0 { -1.0 } else { 1.0 };
        let product = sign * MAGNITUDES[code & 7] * s[(at / cols) * blocks + (at % cols) / inside];
        let want = cubecl_common::e4m3::from_f32(product);
        assert_eq!(byte, want.to_bits(), "value {at}, {product}");
    }
}
