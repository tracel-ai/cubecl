//! The decode gemv's serving geometry, in the split-axis spelling.
//!
//! `y = W · x` with `W` the weight's physical `[d_out, d_in]` buffer as the *lhs*, the
//! orientation a decode step streams, where the contraction runs along the buffer's contiguous
//! direction. `K` is spelled `(KB, KI)`, so one scale per block is the operand omitting `KI`.
//!
//! Two shapes, and the engine picks which by what it will accept. With the accumulator in memory
//! the activation is read one `K`-contiguous line a step, which lets a step fold whole lines;
//! promoted to registers, the block lines its cells along the accumulator, so reads are per cell.
//!
//! Both compute the same answer and neither is a ladder rung of the other.
//! [`cubek-matmul`'s `QuantGemv` routine] is built on the first.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    bytes::Bytes, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle, zspace::shape,
};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

/// The weight's output dimension: rows of the lhs.
const M: Axis = Axis(0);
/// Activation rows projected in one launch (1 at decode).
const N: Axis = Axis(1);
/// The contraction as the two axes a scale block makes of it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// The partials fold through the sink's element between `K` steps. Three levels: this cube's
/// strip of rows, this plane's group of rows, this unit's rows against its share of the
/// contraction, the leaf running under a block of `budget` scalars.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn decode_gemv(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    scale: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] budget: usize,
) {
    let w = w.tile(&space);
    let x = x.tile(&space);
    let scale = scale.tile(&space);
    let out = out.tile(&space);
    // This instance's windows of `out`, each initialized once: the level projected
    // onto `out`'s own axes walks nothing it does not span.
    for cube in &out {
        let mut out_w = out.at(&cube);
        out_w.zero();
    }
    for cube in space {
        let out_cube = out.at(&cube);
        let w_cube = w.at(&cube);
        let x_cube = x.at(&cube);
        let scale_cube = scale.at(&cube);
        for plane in cube {
            let out_plane = out_cube.at(&plane);
            let w_plane = w_cube.at(&plane);
            let x_plane = x_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            for unit in plane {
                let mut out_unit = out_plane
                    .at(&unit)
                    .accumulating(comptime!(RegisterBlock::new(budget)), Semiring::SUM_PROD);
                out_unit.mma(
                    &w_plane.at(&unit).mul(&scale_plane.at(&unit)),
                    &x_plane.at(&unit),
                );
            }
        }
    }
}

/// The same, with the partials living in registers for the whole `K` walk: the kernel opens a
/// register block above the walk and drains it once.
///
/// The activation is read *scalar* here, and has to be: a promoted block lines its cells along
/// the accumulator, so a rhs lined along the contraction folds a whole step into one cell and
/// `RegisterData::mma_scaled` refuses it; the shipping shape thus keeps its accumulator in memory.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn decode_gemv_promoted<E: Numeric>(
    w: &TileArg<'_>,
    x: &TileArg<'_>,
    scale: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] budget: usize,
    #[define(E)] _dtype: ElemType,
) {
    let w = w.tile(&space);
    let x = x.tile(&space);
    let scale = scale.tile(&space);
    let out = out.tile(&space);
    let acc = out.accumulator::<E>(
        &w,
        &x,
        comptime!(Instruction::Registers {
            config: RegisterBlock::new(budget)
        }),
        Semiring::SUM_PROD,
    );
    for cube in space {
        let acc_cube = acc.at(&cube);
        let w_cube = w.at(&cube);
        let x_cube = x.at(&cube);
        let scale_cube = scale.at(&cube);
        for plane in cube {
            let acc_plane = acc_cube.at(&plane);
            let w_plane = w_cube.at(&plane);
            let x_plane = x_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            for unit in plane {
                let mut acc_unit = acc_plane.at(&unit);
                acc_unit.mma(
                    &w_plane.at(&unit).mul(&scale_plane.at(&unit)),
                    &x_plane.at(&unit),
                );
            }
        }
    }
    acc.drained_into(&out);
}

#[test]
fn the_serving_geometry_computes_the_decode_gemv() {
    serving_geometry(false, true);
}

#[test]
fn a_promoted_accumulator_spans_the_whole_decode_walk() {
    serving_geometry(true, true);
}

/// **A unit level that cuts nothing is a loop of one.** Every unit of the plane is handed the
/// plane's whole box (not an empty one) and the leaf runs on all of them alike: planes that own
/// their cells, written with the same three loops as a real unit cut, and giving the same answer.
#[test]
fn a_unit_level_that_cuts_nothing_hands_every_unit_the_plane() {
    serving_geometry(false, false);
}

/// `promoted`: where the accumulator lives. `units_cut`: whether the unit level distributes rows
/// and words to the units, or names no axis and hands every unit the plane's box.
fn serving_geometry(promoted: bool, units_cut: bool) {
    let field = QuantValue::Q8S;
    let bits = field.size_bits();
    let factor = 32 / bits;

    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }
    let plane = client.properties().hardware.plane_size_max as usize;

    // The block, and the geometry cut from it: a unit takes one stored word per step, so a
    // group of `block / factor` units covers one block of `K`, and the rest of the plane
    // carries rows.
    let (block, blocks) = (32, 4);
    let (d_in, n) = (block * blocks, 1);
    let group_units = block / factor;
    if !plane.is_multiple_of(group_units) {
        ValidationResult::Skipped(format!(
            "a {plane}-unit plane does not split into {group_units}-unit groups"
        ))
        .enforce();
        return;
    }
    let groups = plane / group_units;
    let rows_per_unit = 2;
    let rows_per_plane = groups * rows_per_unit;
    let num_planes = 2;
    let rows_per_cube = num_planes * rows_per_plane;
    let d_out = rows_per_cube * 2;

    let span = 1i32 << bits;
    let w: Vec<i32> = (0..d_out * d_in)
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
    let x: Vec<f32> = (0..d_in).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..d_out * blocks)
        .map(|i| (i % 9) as f32 / 4.0 + 0.25)
        .collect();

    // The activation is read one `K`-contiguous line a step where the accumulator sits in
    // memory, and cell by cell where it is promoted (see the kernel above).
    let dtype = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, d_out), (N, n), (KB, blocks), (KI, block)]),
            match units_cut {
                true => Levels::leaf(&[(M, rows_per_unit), (KI, factor), (KB, 1)])
                    .walk_every(&[KB])
                    .units(&[(M, groups), (KI, group_units)])
                    .interleaved(KI),
                false => Levels::leaf(&[(M, rows_per_plane), (KB, 1)])
                    .walk_every(&[KB])
                    .units(&[]),
            }
            .planes(&[(M, num_planes)])
            .cubes(&[M])
            .build(),
        ),
        Form::Static,
    );
    // The leaf's budget: one scalar per row a unit owns, per value of the word it takes a step.
    let budget = rows_per_unit * factor;

    let w_tensor = TensorHandle::new_contiguous(
        vec![d_out, d_in],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let x_shape = if promoted {
        shape![d_in, n]
    } else {
        shape![n, d_in]
    };
    let (x_tensor, _) = TestInput::builder(client.clone(), x_shape)
        .dtype(dtype)
        .custom(x.clone())
        .generate_with_f32_host_data();
    let (s_tensor, _) = TestInput::builder(client.clone(), shape![d_out, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![d_out, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let w_op = launcher
        .arg(w_tensor.binding(), u32::elem_type_native())
        .gathered(Projection::new(
            &[M, KB, KI],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
            ],
        ))
        .packed(field)
        .vectorize(factor)
        .build()
        .unwrap();
    let x_projection = if promoted {
        Projection::new(
            &[KB, KI, N],
            &[
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                PhysicalAxisMap::of(N),
            ],
        )
    } else {
        Projection::new(
            &[N, KB, KI],
            &[
                PhysicalAxisMap::of(N),
                PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
            ],
        )
    };
    let x_op = launcher
        .arg(x_tensor.binding(), dtype)
        .gathered(x_projection)
        .vectorize(if promoted { 1 } else { factor })
        .build()
        .unwrap();
    // One scale per `(row, block of K)`: `KI` is carried and addressed by nothing.
    let s_op = launcher
        .arg(s_tensor.binding(), dtype)
        .axes(&[M, KB])
        .build()
        .unwrap();
    let out_op = launcher
        .arg(out.clone().binding(), dtype)
        .axes(&[M, N])
        .build()
        .unwrap();

    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    if promoted {
        decode_gemv_promoted::launch(
            &client,
            count,
            dim,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launcher.partitioning_arg(),
            budget,
            dtype,
        );
    } else {
        decode_gemv::launch(
            &client,
            count,
            dim,
            w_op.arg(),
            x_op.arg(),
            s_op.arg(),
            out_op.arg(),
            launcher.partitioning_arg(),
            budget,
        );
    }

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for m in 0..d_out {
        let want: f32 = (0..d_in)
            .map(|k| w[m * d_in + k] as f32 * s[m * blocks + k / block] * x[k])
            .sum();
        let have = got.get_f32(&[m, 0]);
        assert!(
            (have - want).abs() < 1e-2,
            "at row {m}: got {have}, want {want}"
        );
    }
}
