//! A convolution's gathered input staged dense over its logical axes (im2col in shared memory)
//! and contracted by tensor-core instructions, which refuse a gathered operand read in place.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::layout::{PhysicalAxisMap, Scale};
use cubecl_tile::stage::RowChunks;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};
use half::f16;

use super::{Form, implied};

const OH: Axis = Axis(0);
const CO: Axis = Axis(1);
const RH: Axis = Axis(2);
const CI: Axis = Axis(3);

const CO_EXTENT: usize = 16;
const RH_EXTENT: usize = 4;

#[cube(launch)]
fn conv_dense_stage_kernel<E: Numeric>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] storage: StageStorage,
    #[comptime] planes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    for cube in space.over(&cubes) {
        let (input, weight, out) = (input.at(&cube), weight.at(&cube), out.at(&cube));
        let acc = out.accumulator::<E>(&input, &weight, Semiring::SUM_PROD);
        let walk = cube.over(&steps);
        let mut stages = Stages::smem(&walk, &input, &weight, storage.clone(), 1usize, None);
        stages.pipelined(walk, |slot, region| {
            let mut acc_r = acc.at(region);
            slot.consume(|i, w| {
                // A stage taller than one block holds a block per plane, each contracting its own.
                if planes > 1 {
                    for plane in region {
                        let mut acc_p = acc_r.at(&plane);
                        acc_p.mma(&i.at(&plane), &w.at(&plane));
                    }
                } else {
                    acc_r.mma(i, w);
                }
            });
        });
        acc.drained_into(&out);
    }
}

/// Which instruction's shapes the leaf is cut to; the accumulator picks the instruction for it.
#[derive(Clone, Copy)]
enum Unit {
    Cmma,
    Mma,
}

/// `input[o * stride + r * tap + offset]` for each output `o` and tap `r`, zero outside `in_len`.
struct Case {
    oh: usize,
    stride: usize,
    tap: isize,
    offset: isize,
    in_len: usize,
    /// Taps a k step spans; the rest of the step is channels.
    taps: usize,
    /// Row blocks the plane level lays side by side, so the stage is that many blocks tall.
    planes: usize,
    /// A plane holds a 2 by 2 grid of fragments, which it loads from the stage before contracting
    /// each cell, over twice the output channels.
    grid: bool,
}

/// The shape of `unit`'s instruction if the device offers it over `f16` summed in `f32`, else a
/// skip.
fn instruction_shape(unit: Unit, client: &cubecl::client::Client) -> Option<(usize, usize, usize)> {
    let (f16_ty, f32_ty) = (f16::elem_type_native(), f32::elem_type_native());
    let features = &client.properties().features.matmul;
    let (configs, wanted): (_, &[(usize, usize, usize)]) = match unit {
        Unit::Cmma => (&features.cmma, &[(16, 16, 16)]),
        Unit::Mma => (&features.mma, &[(16, 8, 8), (16, 8, 16)]),
    };
    let found = wanted.iter().find(|w| {
        configs.iter().any(|c| {
            c.a_type == f16_ty
                && c.b_type == f16_ty
                && c.cd_type == f32_ty
                && (c.m as usize, c.n as usize, c.k as usize) == **w
        })
    });
    let Some(&(m, n, k)) = found else {
        ValidationResult::Skipped("device offers no f16 instruction of the wanted shape".into())
            .enforce();
        return None;
    };
    Some((m, n, k))
}

impl Case {
    fn check(&self, unit: Unit) {
        let client = cubecl::test_device().client();
        let Some((m, n, k)) = instruction_shape(unit, &client) else {
            return;
        };
        let Case {
            oh,
            stride,
            tap,
            offset,
            in_len,
            taps,
            planes,
            grid,
        } = *self;
        let cells = if grid { 2 } else { 1 };
        let (co, rh) = (cells * CO_EXTENT, RH_EXTENT);
        // A k step is `taps` taps by `k / taps` channels; the walk takes two channel steps.
        let ci_step = k / taps;
        let ci = 2 * ci_step;
        assert!(oh % m == 0 && co % n == 0);

        let launcher = implied(
            &client,
            Partitioning::new(Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]), {
                let levels = Levels::leaf(&[(OH, m), (CO, n), (RH, taps), (CI, ci_step)]);
                let levels = match grid {
                    true => levels.walk(&[(OH, 2), (CO, 2)]),
                    false => levels,
                };
                let levels = match planes {
                    1 => levels,
                    _ => levels.planes(&[(OH, planes)]),
                };
                levels.walk_every(&[RH, CI]).cubes(&[OH, CO]).build()
            }),
            Form::Static,
        );
        let storage = StageStorage::Tiled {
            block: launcher.partitioning().leaf().extents(),
            chunks: RowChunks::InOrder,
        };

        // Small integers, exact in `f16` and in the `f32` sums.
        let in_data: Vec<f32> = (0..in_len * ci).map(|i| (i % 7) as f32 - 1.0).collect();
        let w_data: Vec<f32> = (0..rh * ci * co).map(|i| (i % 5) as f32 - 1.0).collect();
        let (f16_ty, f32_ty) = (f16::elem_type_native(), f32::elem_type_native());
        let (in_handle, _) = TestInput::builder(client.clone(), shape![in_len, ci])
            .dtype(f16_ty)
            .custom(in_data.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), shape![rh, ci, co])
            .dtype(f16_ty)
            .custom(w_data.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![oh, co])
            .dtype(f32_ty)
            .uniform(4242, 10., 100.)
            .generate_without_host_data();

        let rows = PhysicalAxisMap::scaled(&[
            (OH, Scale::Static(stride as isize)),
            (RH, Scale::Static(tap)),
        ])
        .shifted(offset);
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[rows, PhysicalAxisMap::of(CI)],
        ))
        .boundary(BoundaryPolicy::Every(Boundary::Zero));

        conv_dense_stage_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileTensorLaunch::new(in_handle.binding().into_tensor_arg(), f16_ty, 1, in_spec),
            TileTensorLaunch::new(
                w_handle.binding().into_tensor_arg(),
                f16_ty,
                1,
                TileSpec::direct(&[RH, CI, CO]),
            ),
            TileTensorLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[OH, CO]),
            ),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            storage,
            planes,
            f32_ty,
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        for o in 0..oh {
            for c in 0..co {
                let mut want = 0.0f32;
                for r in 0..rh {
                    let h = (o * stride) as isize + r as isize * tap + offset;
                    if h < 0 || h as usize >= in_len {
                        continue;
                    }
                    for i in 0..ci {
                        want += in_data[h as usize * ci + i] * w_data[(r * ci + i) * co + c];
                    }
                }
                assert_eq!(got.get_f32(&[o, c]), want, "wrong at ({o}, {c})");
            }
        }
    }
}

/// Stride, dilation and padding at once, over several row tiles so a tile's rows start past the
/// buffer's first.
const FORWARD: Case = Case {
    oh: 48,
    stride: 2,
    tap: 2,
    offset: -1,
    in_len: 99,
    taps: 2,
    planes: 1,
    grid: false,
};

#[test]
fn conv_dense_stage_cmma() {
    FORWARD.check(Unit::Cmma);
}

#[test]
fn conv_dense_stage_mma() {
    FORWARD.check(Unit::Mma);
}

/// The data of a convolution's backward: `grad[o + padding - r * dilation]`, whose rows reach
/// below the gradient at the first outputs and past it at the last.
#[test]
fn conv_dense_stage_cmma_backward() {
    Case {
        oh: 16,
        stride: 1,
        tap: -2,
        offset: 3,
        in_len: 17,
        ..FORWARD
    }
    .check(Unit::Cmma);
}

/// A stage two row blocks tall is tiled, so a fragment's row stride is the block's `taps *
/// channels`, not the channels a step holds.
const TILED: Case = Case {
    oh: 32,
    stride: 1,
    tap: 1,
    offset: 0,
    in_len: 35,
    taps: 2,
    planes: 2,
    grid: false,
};

#[test]
fn conv_dense_stage_cmma_tiled_stage_4_channels() {
    Case { taps: 4, ..TILED }.check(Unit::Cmma);
}

#[test]
fn conv_dense_stage_cmma_tiled_stage_8_channels() {
    TILED.check(Unit::Cmma);
}

#[test]
fn conv_dense_stage_mma_tiled_stage() {
    Case { taps: 4, ..TILED }.check(Unit::Mma);
}

/// One plane over a 2 by 2 grid of fragments, whose lhs rows step by the stride of the output's
/// row axis, `taps * channels`, not the channels of one tap.
const GRID: Case = Case {
    planes: 1,
    grid: true,
    ..TILED
};

#[test]
fn conv_dense_stage_cmma_fragment_grid_taps_2() {
    GRID.check(Unit::Cmma);
}

/// Rows 4 channels apart would put the lanes' 16-byte reads 8 bytes apart.
#[test]
fn conv_dense_stage_cmma_fragment_grid_taps_4() {
    Case { taps: 4, ..GRID }.check(Unit::Cmma);
}
