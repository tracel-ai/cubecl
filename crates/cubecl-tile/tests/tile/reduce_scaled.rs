//! A reduction over a scaled operand: every value read under the scales it carries.
//!
//! The scales are the operand's own ([`Tile::mul`]), so `reduce_axis` reads the values as the
//! contraction does, under them: a row of packed weights sums to the row of their dequantized
//! values, not of their codes.

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
/// The output's one column: a register block drains into a matrix.
const C: Axis = Axis(3);

/// The register block the reduction in registers accumulates in.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `out[row] = Σ_col weights[row, col] · scales[row, col / inside]`, reduced into memory.
#[cube(launch)]
fn reduce_scaled_into_memory(
    weights: &TileTensor<'_>,
    scales: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let weights = weights.tile(&space).mul(&scales.tile(&space));
    let mut out = out.tile(&space);
    out.init::<f32>(0.0f32);
    for region in space.over(&level) {
        let mut out_region = out.at(&region);
        out_region.reduce_axis_accumulate(&weights.at(&region), Monoid::Sum);
    }
}

/// [`reduce_scaled_into_memory`] into a register block that lives across the walk.
#[cube(launch)]
fn reduce_scaled_in_registers(
    weights: &TileTensor<'_>,
    scales: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let weights = weights.tile(&space).mul(&scales.tile(&space));
    let out = out.tile(&space);
    let mut sum = out.block_reducer::<f32>(&weights, REGISTER_BLOCK, Monoid::Sum);
    sum.init::<f32>(0.0f32);
    for region in space.over(&level) {
        let mut sum_region = sum.at(&region);
        sum_region.reduce_axis_accumulate(&weights.at(&region), Monoid::Sum);
    }
    sum.drained_into(&out);
}

/// Packed signed 4-bit weights, `rows × blocks · inside`, each block of `inside` columns of a row
/// under its own scale, reduced along the columns into memory or into registers, a leaf
/// `leaf_blocks` blocks long: a line a block.
fn check_reduce_under_scales(registers: bool, blocks: usize, leaf_blocks: usize) {
    let (rows, inside) = (4, 8);
    let cols = blocks * inside;
    let field = QuantValue::Q4S;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..rows * cols)
        .map(|i| -(span / 2) + ((i * 5) as i32 % span))
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
    let want: Vec<f32> = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| w[row * cols + col] as f32 * s[row * blocks + col / inside])
                .sum()
        })
        .collect();

    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, rows), (C, 1), (CB, blocks), (CI, inside)]),
            Levels::leaf(&[(ROW, rows), (C, 1), (CB, leaf_blocks), (CI, inside)])
                .walk_every(&[CB])
                .build(),
        ),
        Form::Static,
    );

    let w_tensor = TensorHandle::new_contiguous(
        vec![rows, cols],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![rows, 1])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let w_op = launch
        .arg(w_tensor.clone().binding(), w_tensor.dtype)
        .gathered(Projection::new(
            &[ROW, CB, CI],
            &[
                PhysicalAxisMap::of(ROW),
                PhysicalAxisMap::disjoint(&[(CB, inside), (CI, 1)]),
            ],
        ))
        .packed(field)
        .vectorize(factor)
        .build()
        .unwrap();
    let s_op = launch
        .arg(s_tensor.binding(), dtype)
        .axes(&[ROW, CB])
        .build()
        .unwrap();
    let out_op = launch
        .arg(out.clone().binding(), dtype)
        .axes(&[ROW, C])
        .build()
        .unwrap();
    let level = launch.partitioning().level(0);
    match registers {
        false => reduce_scaled_into_memory::launch(
            &client,
            launch.cube_count(),
            launch.cube_dim(),
            w_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launch.partitioning_arg(),
            level,
        ),
        true => reduce_scaled_in_registers::launch(
            &client,
            launch.cube_count(),
            launch.cube_dim(),
            w_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launch.partitioning_arg(),
            level,
        ),
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for (row, want) in want.iter().enumerate() {
        let have = got.get_f32(&[row, 0]);
        assert!(
            (have - want).abs() < 1e-3,
            "registers {registers}, row {row}: got {have}, want {want}"
        );
    }
}

#[test]
fn a_reduction_into_memory_reads_its_values_under_their_scales() {
    check_reduce_under_scales(false, 4, 1);
}

#[test]
fn a_reduction_in_registers_reads_its_values_under_their_scales() {
    check_reduce_under_scales(true, 4, 1);
}

/// A leaf of 32 lines, past the walk the leaf unrolls, sums the same rolled.
#[test]
fn a_long_leaf_reads_its_values_under_their_scales_rolled() {
    check_reduce_under_scales(false, 64, 32);
    check_reduce_under_scales(true, 64, 32);
}
