//! Convolution as a client of the tile DSL: a gather-reduce whose reduce axes are *abstract*.
//!
//! A convolution's window axes are not physical axes of the input. They address the same physical
//! axis its output axes address, at their own coefficient: `Ih = Oh*stride + Rh*dilation`. So the
//! input tile spans more logical axes than its buffer has physical ones, and tiles overlap.
//!
//! That mapping is the operand's [`Projection`]; everything else (the space, the walk, `Tile::at`,
//! the leaf) is the same machinery matmul runs on.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    prelude::*,
    zspace::{Shape, shape},
};

use super::{Form, implied};
use cubecl_tile::layout::Divisor;
use cubecl_tile::layout::Offset;
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::layout::Scale;
use cubecl_tile::ops::matmul::LoadMethod;
use cubecl_tile::ops::matmul::MmaIo;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

// Output positions, output channels, window taps, input channels. `OW`/`RW` are the second
// spatial pair the 2-D case adds.
const OH: Axis = Axis(0);
const CO: Axis = Axis(1);
const RH: Axis = Axis(2);
const CI: Axis = Axis(3);
const OW: Axis = Axis(4);
const RW: Axis = Axis(5);

/// The software instruction the leaves here run under unless a test states another: a 16-cell
/// budget, no edge split, no unit fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// Where a kernel reads its inputs from: where they lie, or a shared-memory stage it fills per
/// region of its first level, `depth` slots deep, the gathered input served in `width`-wide lines
/// where one is stated (a scalar operand padded out to whole lines).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    InPlace,
    Smem { depth: usize, width: Option<usize> },
}

/// The same body matmul runs: the operands' spaces say what is contracted, their projections say
/// how they are addressed, and the leaf does the rest. The input's line width, along its fastest
/// contracted axis, is what the leaf splits into line index and unit; a real width is needed.
#[cube(launch)]
fn conv_kernel(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] config: RegisterBlock,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    for region in space.over(&level) {
        let mut out_region = out.at(&region).accumulating(config, Semiring::SUM_PROD);
        out_region.mm(&input.at(&region), &weight.at(&region));
    }
}

/// [`conv_kernel`] with both inputs staged into shared memory per region. A gathered operand's
/// region is copied into a stage shaped like the physical *window* it reads, compacted onto the
/// lattice its stride and dilation reach, and the leaf gathers out of the stage as out of gmem.
#[cube(launch)]
fn conv_kernel_smem(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] config: RegisterBlock,
    #[comptime] depth: usize,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &input, &weight, StageStorage::Strided, depth, None);
    stages.pipelined(walk, |slot, region| {
        let mut out_region = out.at(region).accumulating(config, Semiring::SUM_PROD);
        slot.consume(|input, weight| {
            out_region.mm(input, weight);
        });
    });
}

/// [`conv_kernel_smem`] with the scalar input alone staged, padded out to `width`-wide lines;
/// the weight is read where it lies.
#[cube(launch)]
fn conv_kernel_smem_padded(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] config: RegisterBlock,
    #[comptime] depth: usize,
    #[comptime] width: usize,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    let walk = space.over(&level);
    let mut stages = Stages::smem_single_at(
        &walk,
        &input,
        StageStorage::Strided,
        comptime!(Some(width)),
        depth,
        None,
    );
    stages.pipelined(walk, |slot, region| {
        let mut out_region = out.at(region).accumulating(config, Semiring::SUM_PROD);
        let weight = weight.at(region);
        slot.consume(|input| {
            out_region.mm(input, &weight);
        });
    });
}

/// [`conv_kernel`] over two levels, both read where the operands lie: a second descent, where a
/// rational window's leftover phase has to accumulate rather than restart.
#[cube(launch)]
fn conv_kernel_two_levels(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] config: RegisterBlock,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    for outer in space.over(&outer) {
        let out_outer = out.at(&outer);
        let input_outer = input.at(&outer);
        let weight_outer = weight.at(&outer);
        for inner in outer.over(&inner) {
            let mut out_inner = out_outer
                .at(&inner)
                .accumulating(config, Semiring::SUM_PROD);
            out_inner.mm(&input_outer.at(&inner), &weight_outer.at(&inner));
        }
    }
}

/// [`conv_kernel_two_levels`] with both inputs staged at the first level, the second walking
/// windows of the stage.
#[cube(launch)]
fn conv_kernel_two_levels_smem(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] config: RegisterBlock,
    #[comptime] depth: usize,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    let walk = space.over(&outer);
    let mut stages = Stages::smem(&walk, &input, &weight, StageStorage::Strided, depth, None);
    stages.pipelined(walk, |slot, region| {
        let out_outer = out.at(region);
        slot.consume(|input, weight| {
            for inner in region.over(&inner) {
                let mut out_inner = out_outer
                    .at(&inner)
                    .accumulating(config, Semiring::SUM_PROD);
                out_inner.mm(&input.at(&inner), &weight.at(&inner));
            }
        });
    });
}

/// Small integers, so the accumulation is exact in `f32` and the reference can be compared for
/// equality rather than closeness.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    in_shape: Shape,
    w_shape: Shape,
    out_shape: Shape,
    in_spec: TileSpec,
    w_axes: &[Axis],
    out_spec: TileSpec,
    launcher: Launcher,
    in_v: usize,
    config: RegisterBlock,
    stage: Stage,
) -> (HostData, Vec<f32>, Vec<f32>) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let in_data = ramp(in_shape.num_elements(), 7);
    let w_data = ramp(w_shape.num_elements(), 5);

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // Every binding stays scalar-unit: its argument carries the width, and `Tile::of` is what
    // converts the shape and strides to lines.
    let in_arg = TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, in_v, in_spec);
    let w_arg = TileArgLaunch::new(
        w_handle.binding().into_tensor_arg(),
        f32_ty,
        1,
        TileSpec::direct(w_axes),
    );
    let out_arg = TileArgLaunch::new(
        out_handle.clone().binding().into_tensor_arg(),
        f32_ty,
        1,
        out_spec,
    );
    let cube_count = launcher.cube_count();
    let cube_dim = launcher.cube_dim();
    // The kernel that walks this space: one loop per level, the stage where `stage` says.
    match (launcher.partitioning().levels().len(), stage) {
        (1, Stage::InPlace) => conv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            in_arg,
            w_arg,
            out_arg,
            config,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
        (1, Stage::Smem { depth, width: None }) => conv_kernel_smem::launch(
            &client,
            cube_count,
            cube_dim,
            in_arg,
            w_arg,
            out_arg,
            config,
            depth,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
        (
            1,
            Stage::Smem {
                depth,
                width: Some(width),
            },
        ) => {
            assert_eq!(
                in_v, 1,
                "conv: a padded stage is filled from a scalar operand"
            );
            conv_kernel_smem_padded::launch(
                &client,
                cube_count,
                cube_dim,
                in_arg,
                w_arg,
                out_arg,
                config,
                depth,
                width,
                launcher.partitioning_arg(),
                launcher.partitioning().level(0),
            )
        }
        (2, Stage::InPlace) => conv_kernel_two_levels::launch(
            &client,
            cube_count,
            cube_dim,
            in_arg,
            w_arg,
            out_arg,
            config,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
        ),
        (2, Stage::Smem { depth, width: None }) => conv_kernel_two_levels_smem::launch(
            &client,
            cube_count,
            cube_dim,
            in_arg,
            w_arg,
            out_arg,
            config,
            depth,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
        ),
        (levels, stage) => panic!("conv: no kernel walks {levels} levels under {stage:?}"),
    }

    (
        HostData::from_tensor_handle(&client, out_handle, HostDataType::F32),
        in_data,
        w_data,
    )
}

// ---- 1-D -------------------------------------------------------------------

/// `out[o, co] = Σ_{r, ci} w[r, ci, co] · in[o·stride + r·dilation, ci]`.
struct Conv1d {
    oh: usize,
    co: usize,
    rh: usize,
    ci: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    /// The input length a valid (unpadded) convolution consumes: the last output position's last
    /// tap, plus one.
    fn in_len(&self) -> usize {
        (self.oh - 1) * self.stride + (self.rh - 1) * self.dilation + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let x = input[(o * self.stride + r * self.dilation) * self.ci + i];
                        let w = weight[(r * self.ci + i) * self.co + c];
                        acc += x * w;
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_co, 1, false, Stage::InPlace)
    }

    /// `check` with the input served in `in_v`-wide lines and, when `checked`, both the input and
    /// the output bounds-masked: the two gather-path axes a plain `check` leaves degenerate.
    /// `stage` picks gathering from gmem or from a compacted stage, and how deeply it is buffered.
    fn check_at(&self, tile_oh: usize, tile_co: usize, in_v: usize, checked: bool, stage: Stage) {
        self.check_at_with_block(
            tile_oh,
            tile_co,
            in_v,
            checked,
            stage,
            RegisterBlock::new(16),
        )
    }

    /// `check_at` with an explicit register block. Kept separate so the fan-out path can be traced
    /// on the CPU test runtime even though production CPU launches select the flat walk.
    fn check_at_with_block(
        &self,
        tile_oh: usize,
        tile_co: usize,
        in_v: usize,
        checked: bool,
        stage: Stage,
        config: RegisterBlock,
    ) {
        let launcher = implied(
            &cubecl::test_device().client(),
            Partitioning::new(
                Space::new(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]),
                Levels::leaf(&[(OH, tile_oh), (CO, tile_co), (RH, self.rh), (CI, self.ci)])
                    .walk_every(&[OH, CO, RH, CI])
                    .build(),
            ),
            Form::Static,
        );

        // The input's one gathered physical axis: the output position at `stride`, the tap at
        // `dilation`.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.stride), (RH, self.dilation)]),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .boundary(if checked {
            BoundaryPolicy::Every(Boundary::Zero)
        } else {
            BoundaryPolicy::Unchecked
        });

        let (got, input, weight) = run(
            shape![self.in_len(), self.ci],
            shape![self.rh, self.ci, self.co],
            shape![self.oh, self.co],
            in_spec,
            &[RH, CI, CO],
            TileSpec::direct(&[OH, CO]).boundary(if checked {
                BoundaryPolicy::Every(Boundary::Zero)
            } else {
                BoundaryPolicy::Unchecked
            }),
            launcher.clone(),
            in_v,
            config,
            stage,
        );

        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "conv1d {stage:?} stride {} dilation {} tile {tile_oh}x{tile_co}: \
                     wrong at ({o}, {c})",
                    self.stride,
                    self.dilation
                );
            }
        }
    }
}

/// Stride 1, dilation 1: consecutive windows overlap by `rh - 1`, the plainest gather there is.
#[test]
fn conv1d_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// Stride 2: the output steps twice as fast as the input, so a tile's span is no longer its edge.
#[test]
fn conv1d_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check(4, 4);
}

/// Dilation 2: the taps themselves are spread, so the two coefficients differ.
#[test]
fn conv1d_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 2,
    }
    .check(3, 4);
}

/// Both at once, and a tap count that does not divide anything.
#[test]
fn conv1d_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check(2, 2);
}

/// A 1x1 window is a matmul wearing a projection: the gather degenerates and must still agree.
#[test]
fn conv1d_single_tap_is_a_matmul() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// One output tile covering everything: the whole receptive field is one window.
#[test]
fn conv1d_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(6, 4);
}

/// Padded 1-D convolution: taps that underflow physical bounds mask to 0.
#[test]
fn conv1d_padded_underflow_masks_to_zero() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, 3), (CO, 4), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]).shifted(-(padding as isize)),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        launcher.clone(),
        1,
        RegisterBlock::new(16),
        Stage::InPlace,
    );

    // Reference with zero padding:
    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "padded conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// Padded 1-D convolution under `Boundary::Clamp`: taps that go past physical bounds read the
/// edge cell instead of masking to zero.
#[test]
fn conv1d_padded_underflow_clamps_to_edge() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, 3), (CO, 4), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]).shifted(-(padding as isize)),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Clamp));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        launcher.clone(),
        1,
        RegisterBlock::new(16),
        Stage::InPlace,
    );

    // Reference clamping the tap to the input's edge row instead of masking to zero:
    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let hc = h.clamp(0, in_len as isize - 1) as usize;
                    let x = input[hc * ci + i];
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "clamped conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// The same padded convolution under a staged schedule.
#[test]
fn conv1d_padded_staged_underflow_masks_to_zero() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, 3), (CO, 4), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]).shifted(-(padding as isize)),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        launcher.clone(),
        1,
        RegisterBlock::new(16),
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );

    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "padded staged conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// The gathered operand in two-wide lines. Its innermost axis is `CI`, the fastest contracted one,
/// so the leaf splits each reduce step into the line index it reads and the unit it broadcasts:
/// the one arithmetic on the gather path that a width of `1` leaves dead.
#[test]
fn conv1d_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, false, Stage::InPlace);
}

/// The GPU specialization uses fixed unit extracts. Exercise that path on the CPU test runtime as
/// well, even though production CPU launches select the compact flat walk.
#[test]
fn conv1d_vectorized_input_fanout() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at_with_block(
        4,
        4,
        2,
        false,
        Stage::InPlace,
        RegisterBlock::new(16).component_fanout(),
    );
}

/// Vectorized with stride and dilation both off `1`, so the line split and the affine advance are
/// exercised at once rather than one at a time.
#[test]
fn conv1d_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(3, 2, 2, false, Stage::InPlace);
}

/// An output extent the tile edge does not divide: the last tile's receptive field runs past the
/// input's real length. The masking happens under the projection, so the overhanging tap must
/// read `0` and the nonexistent output cell must not be written; every valid position matches.
#[test]
fn conv1d_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, true, Stage::InPlace);
}

/// The same overhang with a stride, where the window runs further past the end: the mask is on the
/// physical axis, so how far the gather reached past it must not matter.
#[test]
fn conv1d_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, true, Stage::InPlace);
}

// ---- through the Launcher --------------------------------------------------

impl Conv1d {
    /// [`check_at`](Conv1d::check_at) driven end to end by [`Launcher`]: the input reaches the
    /// launch through [`StridedTileSource::gathered`] instead of a hand-built [`TileSpec`], so the
    /// kernel projects from the kernel-form nest and one compiled kernel serves every shape.
    ///
    /// `padding` shifts the window's origin, which the builder's derived check has to arm on by
    /// itself. `dynamic` is the axis set the kernel takes at runtime, `None` for all of them.
    fn check_launched_over(
        &self,
        tile_oh: usize,
        tile_co: usize,
        padding: usize,
        dynamic: Option<&[Axis]>,
        stage: Stage,
    ) {
        let client = cubecl::test_device().client();
        let f32_ty = f32::elem_type_native();

        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]),
                Levels::leaf(&[(OH, tile_oh), (CO, tile_co), (RH, self.rh), (CI, self.ci)])
                    .walk_every(&[OH, CO, RH, CI])
                    .build(),
            ),
            Form::Static,
        );

        // Padding shortens the input by exactly what it shifts the window back by, so the last
        // output position's last tap still lands on the final row.
        let in_len = self.in_len() - padding;
        let in_shape = shape![in_len, self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        // Every axis here can go runtime, including the two the input gathers over: `OH` is stated
        // by the output, which maps it identically, and `RH` by the weight. Only an axis no
        // operand witnesses has to stay static, which is what `dynamic` narrows to.
        let launch = match dynamic {
            Some(axes) => implied(
                &client,
                Partitioning::new(
                    launcher.space().clone(),
                    launcher.partitioning().levels().to_vec(),
                ),
                Form::DynamicAlong(axes),
            ),
            None => implied(
                &client,
                Partitioning::new(
                    launcher.space().clone(),
                    launcher.partitioning().levels().to_vec(),
                ),
                Form::Dynamic,
            ),
        };
        let in_arg = launch
            .arg(in_handle.binding(), f32_ty)
            .gathered(Projection::new(
                &[OH, RH, CI],
                &[
                    PhysicalAxisMap::affine(&[(OH, self.stride), (RH, self.dilation)])
                        .shifted(-(padding as isize)),
                    PhysicalAxisMap::of(CI),
                ],
            ))
            .build()
            .unwrap();
        let w_arg = launch
            .arg(w_handle.binding(), f32_ty)
            .axes(&[RH, CI, CO])
            .build()
            .unwrap();
        let out_arg = launch
            .arg(out_handle.clone().binding(), f32_ty)
            .axes(&[OH, CO])
            .build()
            .unwrap();

        match stage {
            Stage::InPlace => conv_kernel::launch(
                &client,
                launch.cube_count(),
                launch.cube_dim(),
                in_arg.arg(),
                w_arg.arg(),
                out_arg.arg(),
                RegisterBlock::new(16),
                launch.partitioning_arg(),
                launch.partitioning().level(0),
            ),
            Stage::Smem { depth, width: None } => conv_kernel_smem::launch(
                &client,
                launch.cube_count(),
                launch.cube_dim(),
                in_arg.arg(),
                w_arg.arg(),
                out_arg.arg(),
                RegisterBlock::new(16),
                depth,
                launch.partitioning_arg(),
                launch.partitioning().level(0),
            ),
            Stage::Smem { width: Some(_), .. } => {
                panic!("launched conv1d: no padded stage through the launcher")
            }
        }

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference_padded(&input, &weight, in_len, padding);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "launched conv1d {stage:?} padding {padding}: wrong at ({o}, {c})"
                );
            }
        }
    }

    /// [`check_launched_over`](Conv1d::check_launched_over) with every axis runtime.
    fn check_launched(&self, tile_oh: usize, tile_co: usize, padding: usize, stage: Stage) {
        self.check_launched_over(tile_oh, tile_co, padding, None, stage);
    }
}

/// The two axes the input gathers over, `OH` and `RH`, kept static while the rest goes runtime:
/// the launch a gathered operand was restricted to before any operand could state a gathered
/// axis' size. It must keep working: an axis no operand witnesses still reaches the kernel static.
#[test]
fn conv1d_launched_static_window() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_launched_over(3, 4, 0, Some(&[CO, CI]), Stage::InPlace);
}

/// The same convolution with `OH` and `RH` runtime as well, at a tiling neither divides: the walk
/// counts are `div_ceil`s of sizes read off the output (`OH`) and the weight (`RH`), and the
/// trailing partial tile is masked exactly as the static form's is.
#[test]
fn conv1d_launched_dynamic_window_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 2,
    }
    .check_launched(4, 4, 1, Stage::InPlace);
}

/// The plainest gather off the builder: nothing overhangs and the origin cannot go negative, so
/// the derived check is off and the kernel's own extents are all runtime.
#[test]
fn conv1d_launched_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(4, 4, 0, Stage::InPlace);
}

/// Stride and dilation both off `1`, staged: the affine advance and the compaction have to survive
/// the dynamic space the launcher hands the kernel.
#[test]
fn conv1d_launched_staged_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_launched(
        2,
        2,
        0,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// A padded window with no overhang anywhere: nothing about the tiling says the operand needs a
/// check, so the mask is armed by the negative origin alone and the padded taps must read zero.
#[test]
fn conv1d_launched_padding_arms_the_check() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(3, 4, 1, Stage::InPlace);
}

/// An output extent the tile edge does not divide, so the overhang arms the check through the
/// affine map: the tap that runs past the input reads `0` and the row that does not exist is not
/// written.
#[test]
fn conv1d_launched_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(4, 4, 0, Stage::InPlace);
}

// ---- runtime coefficients --------------------------------------------------

impl Conv1d {
    /// [`check`](Conv1d::check) with both coefficients `Dynamic`, so the same convolution runs off
    /// a kernel that was compiled without knowing either. Their bounds are the exact values here,
    /// the tightest a stage can be sized at; `check_dynamic_padded` covers the staged schedule.
    fn check_dynamic(&self, tile_oh: usize, tile_co: usize) {
        let client = cubecl::test_device().client();
        let f32_ty = f32::elem_type_native();

        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]),
                Levels::leaf(&[(OH, tile_oh), (CO, tile_co), (RH, self.rh), (CI, self.ci)])
                    .walk_every(&[OH, CO, RH, CI])
                    .build(),
            ),
            Form::Static,
        );

        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[
                    (OH, Scale::Dynamic { max: self.stride }),
                    (RH, Scale::Dynamic { max: self.dilation }),
                ]),
                PhysicalAxisMap::of(CI),
            ],
        ));

        let in_shape = shape![self.in_len(), self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        conv_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec)
                .runtime_map(vec![self.stride as u32, self.dilation as u32], Vec::new()),
            TileArgLaunch::new(
                w_handle.binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[RH, CI, CO]),
            ),
            TileArgLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                TileSpec::direct(&[OH, CO]),
            ),
            REGISTER_BLOCK,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "dynamic conv1d stride {} dilation {}: wrong at ({o}, {c})",
                    self.stride,
                    self.dilation
                );
            }
        }
    }
}

/// The degenerate coefficients, which a `Static` projection evaluates away entirely: the runtime path
/// has to produce the same answer without that evaluation.
#[test]
fn conv1d_dynamic_unit_coefficients() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic(4, 4);
}

/// A stride and a dilation that both exceed one, over several tiles: the window advance and the
/// receptive field are then runtime values, which is what the descent has to size its window from.
#[test]
fn conv1d_dynamic_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_dynamic(2, 2);
}

impl Conv1d {
    /// The padded reference: a tap outside `[0, in_len)` reads zero.
    fn reference_padded(
        &self,
        input: &[f32],
        weight: &[f32],
        in_len: usize,
        padding: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let h = (o * self.stride + r * self.dilation) as isize - padding as isize;
                        let x = if h >= 0 && (h as usize) < in_len {
                            input[(h as usize) * self.ci + i]
                        } else {
                            0.0
                        };
                        acc += x * weight[(r * self.ci + i) * self.co + c];
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    /// A padded convolution whose padding is an `Offset::Dynamic`, so the window origin is placed
    /// at runtime and the underflow guard is armed without knowing the sign. `dynamic_scales` also
    /// hands stride and dilation over at runtime; `staged` reads both inputs out of a smem stage.
    fn check_dynamic_padded(
        &self,
        tile_oh: usize,
        tile_co: usize,
        padding: usize,
        in_len: usize,
        dynamic_scales: bool,
        staged: bool,
    ) {
        let client = cubecl::test_device().client();
        let f32_ty = f32::elem_type_native();

        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]),
                Levels::leaf(&[(OH, tile_oh), (CO, tile_co), (RH, self.rh), (CI, self.ci)])
                    .walk_every(&[OH, CO, RH, CI])
                    .build(),
            ),
            Form::Static,
        );

        let gathered = if dynamic_scales {
            PhysicalAxisMap::scaled(&[
                (OH, Scale::Dynamic { max: self.stride }),
                (RH, Scale::Dynamic { max: self.dilation }),
            ])
            .shifted(Offset::Dynamic)
        } else {
            PhysicalAxisMap::affine(&[(OH, self.stride), (RH, self.dilation)])
                .shifted(Offset::Dynamic)
        };
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[gathered, PhysicalAxisMap::of(CI)],
        ))
        .boundary(BoundaryPolicy::Every(Boundary::Zero));
        let w_spec = TileSpec::direct(&[RH, CI, CO]);

        let in_shape = shape![in_len, self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        let in_binding = in_handle.binding();
        let w_binding = w_handle.binding();
        let out_binding = out_handle.clone().binding();
        let offset = -(padding as i32);
        let out_spec = TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero));
        let cube_count = launcher.cube_count();
        let cube_dim = launcher.cube_dim();
        let coefficients = match dynamic_scales {
            true => vec![self.stride as u32, self.dilation as u32],
            false => Vec::new(),
        };
        let in_arg = TileArgLaunch::new(in_binding.into_tensor_arg(), f32_ty, 1, in_spec)
            .runtime_map(coefficients, vec![offset]);
        let w_arg = TileArgLaunch::new(w_binding.into_tensor_arg(), f32_ty, 1, w_spec);
        let out_arg = TileArgLaunch::new(out_binding.into_tensor_arg(), f32_ty, 1, out_spec);
        match staged {
            false => conv_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                in_arg,
                w_arg,
                out_arg,
                REGISTER_BLOCK,
                launcher.partitioning_arg(),
                launcher.partitioning().level(0),
            ),
            true => conv_kernel_smem::launch(
                &client,
                cube_count,
                cube_dim,
                in_arg,
                w_arg,
                out_arg,
                REGISTER_BLOCK,
                1,
                launcher.partitioning_arg(),
                launcher.partitioning().level(0),
            ),
        }

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference_padded(&input, &weight, in_len, padding);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "dynamic padded conv1d staged {staged} padding {padding}: wrong at ({o}, {c})"
                );
            }
        }
    }
}

/// A runtime padding under a direct schedule: the taps that underflow must still mask to zero,
/// though nothing proved at comptime that the origin was negative.
#[test]
fn conv1d_dynamic_padding_direct() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 6, false, false);
}

/// The same padding staged: a dynamic offset costs no comptime window geometry at all, so it
/// survives the compaction into smem without even a bound to declare.
#[test]
fn conv1d_dynamic_padding_staged() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 6, false, true);
}

/// Stride, dilation and padding all runtime at once: the whole affine map arrives with the launch.
#[test]
fn conv1d_all_dynamic() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 8, true, false);
}

/// The same thing staged: the runtime coefficients size their smem box off their declared bounds,
/// so the stage holds the window and the launch's actual stride and dilation read out of it.
#[test]
fn conv1d_all_dynamic_staged() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 8, true, true);
}

// ---- 2-D -------------------------------------------------------------------

/// `out[oh, ow, co] = Σ_{rh, rw, ci} w[rh, rw, ci, co] · in[oh·sh + rh·dh, ow·sw + rw·dw, ci]`.
/// Two abstract dimensions, each gathering its own input physical axis.
struct Conv2d {
    oh: usize,
    ow: usize,
    co: usize,
    rh: usize,
    rw: usize,
    ci: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
}

impl Conv2d {
    fn in_h(&self) -> usize {
        (self.oh - 1) * self.sh + (self.rh - 1) * self.dh + 1
    }
    fn in_w(&self) -> usize {
        (self.ow - 1) * self.sw + (self.rw - 1) * self.dw + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let (iw, ci, co) = (self.in_w(), self.ci, self.co);
        let mut out = vec![0.0f32; self.oh * self.ow * co];
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..co {
                    let mut acc = 0.0f32;
                    for rh in 0..self.rh {
                        for rw in 0..self.rw {
                            for i in 0..ci {
                                let h = oh * self.sh + rh * self.dh;
                                let w_ = ow * self.sw + rw * self.dw;
                                let x = input[(h * iw + w_) * ci + i];
                                let k = weight[((rh * self.rw + rw) * ci + i) * co + c];
                                acc += x * k;
                            }
                        }
                    }
                    out[(oh * self.ow + ow) * co + c] = acc;
                }
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_ow: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_ow, tile_co, Stage::InPlace)
    }

    /// `check` under `stage`: `InPlace` gathers straight out of gmem, `Smem` compacts the two
    /// gathered physical axes into a dense stage first.
    fn check_at(&self, tile_oh: usize, tile_ow: usize, tile_co: usize, stage: Stage) {
        let launcher = implied(
            &cubecl::test_device().client(),
            Partitioning::new(
                Space::new(&[
                    (OH, self.oh),
                    (OW, self.ow),
                    (CO, self.co),
                    (RH, self.rh),
                    (RW, self.rw),
                    (CI, self.ci),
                ]),
                Levels::leaf(&[
                    (OH, tile_oh),
                    (OW, tile_ow),
                    (CO, tile_co),
                    (RH, self.rh),
                    (RW, self.rw),
                    (CI, self.ci),
                ])
                .walk_every(&[OH, OW, CO, RH, RW, CI])
                .build(),
            ),
            Form::Static,
        );

        // Two gathered physical axes, one per spatial axis pair; the channel axis rides identity.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, OW, RH, RW, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.sh), (RH, self.dh)]),
                PhysicalAxisMap::affine(&[(OW, self.sw), (RW, self.dw)]),
                PhysicalAxisMap::of(CI),
            ],
        ));

        let (got, input, weight) = run(
            shape![self.in_h(), self.in_w(), self.ci],
            shape![self.rh, self.rw, self.ci, self.co],
            shape![self.oh, self.ow, self.co],
            in_spec,
            &[RH, RW, CI, CO],
            TileSpec::direct(&[OH, OW, CO]),
            launcher.clone(),
            1,
            RegisterBlock::new(16),
            stage,
        );

        let want = self.reference(&input, &weight);
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..self.co {
                    assert_eq!(
                        got.get_f32(&[oh, ow, c]),
                        want[(oh * self.ow + ow) * self.co + c],
                        "conv2d {stage:?}: wrong at ({oh}, {ow}, {c})"
                    );
                }
            }
        }
    }
}

#[test]
fn conv2d_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(2, 2, 4);
}

/// Different stride and dilation per spatial axis, so the two projected physical axes cannot be
/// confused.
#[test]
fn conv2d_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check(2, 3, 2);
}

/// One tile covering the whole output: every axis's window is the full receptive field.
#[test]
fn conv2d_single_tile() {
    Conv2d {
        oh: 3,
        ow: 3,
        co: 2,
        rh: 2,
        rw: 2,
        ci: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(3, 3, 2);
}

// ---- staged ----------------------------------------------------------------

// The same convolutions with the input staged in shared memory. A gathered operand's region is
// copied into a shared-memory tile shaped like the physical *window* it reads, `span(oh, rh) × ci`,
// compacted onto the lattice its stride and dilation reach, so each input element is stored once.
//
// The stage keeps the operand's own projection, so the leaf gathers out of smem exactly as it
// gathered out of gmem, and every case here must agree with its direct (in-place) twin.

/// Stride 1: consecutive windows overlap by `rh - 1`, so the window the stage holds
/// (`oh + rh - 1`) is smaller than the `oh × rh` logical cells reading it.
#[test]
fn conv1d_staged_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        1,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Stride and dilation both off `1`: the fill's per-axis advance and the receptive-field span are
/// what place each staged element, so neither can collapse to the edge.
#[test]
fn conv1d_staged_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_at(
        2,
        2,
        1,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// A single tap at unit stride: the window is exactly the output's own extent, so the stage is the
/// smallest it ever gets while the projection stays affine (`Rh` is still a declared axis at
/// coefficient `1`, so the leaf stays on the N-D nest). The answer must not move.
#[test]
fn conv1d_staged_single_tap() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        1,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// One region covering everything: the walk runs once, so the fill is the whole kernel.
#[test]
fn conv1d_staged_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        6,
        4,
        1,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Staged in two-wide lines: the fill moves whole lines, and the innermost axis is the one the
/// projection is required to carry at coefficient `1` for that to be sound.
#[test]
fn conv1d_staged_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        2,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Vectorized with stride and dilation both off `1`.
#[test]
fn conv1d_staged_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(
        3,
        2,
        2,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// The overhanging last tile: the mask sits under the projection, so a tap past the input's real
/// length must stage a `0` rather than whatever the buffer holds there.
#[test]
fn conv1d_staged_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        1,
        true,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// The same overhang with a stride, where the window runs further past the end.
#[test]
fn conv1d_staged_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(
        2,
        4,
        1,
        true,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// A single tap at stride 2 reaches only every second input position, so the stage keeps half of
/// the window it spans and the fill steps by two. One of the three non-unit-step cases:
/// `conv2d_staged_mixed_steps`, `conv1d_staged_vectorized_strided_and_dilated` (`gcd(2, 2) = 2`).
#[test]
fn conv1d_staged_single_tap_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        1,
        false,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// The overhang masked *and* the input lined: the fill steps whole lines through a window whose
/// last taps read past the input, so the line split and the mask have to hold at once.
#[test]
fn conv1d_staged_vectorized_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(
        4,
        4,
        2,
        true,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Double buffering drives the same fill from two slots on alternating regions, so a gathered
/// source has to re-window per slot rather than once per walk.
#[test]
fn conv1d_staged_double_buffered() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(
        2,
        4,
        1,
        false,
        Stage::Smem {
            depth: 2,
            width: None,
        },
    );
}

/// Two gathered physical axes staged at once: the fill decodes one destination coordinate per
/// logical axis, so both affine maps are resolved from the same index.
#[test]
fn conv2d_staged_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check_at(
        2,
        2,
        4,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Different stride and dilation per spatial axis, so the two staged physical axes cannot be
/// confused for one another.
#[test]
fn conv2d_staged_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check_at(
        2,
        3,
        2,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// One compacted physical axis per step: `h` has `gcd(2, 2) = 2`, so its stage keeps every second
/// row and the fill steps through it, while `w` is dense and steps by one. The only case where
/// `CompactionStep` carries more than one distinct step, so a transposed or broadcast step shows up
/// here.
#[test]
fn conv2d_staged_mixed_steps() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 2,
        dw: 1,
    }
    .check_at(
        2,
        2,
        4,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

// ---- the manual-mma leaf ---------------------------------------------------

/// The resident promote, zero, mma, drain kernel of the matmul tests, with a *gathered* lhs. The
/// accumulator is sized by the whole contraction (taps times channels), both inputs stage into
/// smem (the input into its compacted window); the fragment load merges tap and channel into `k`.
#[cube(launch)]
fn conv_mma_kernel<E: Numeric>(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    #[comptime] io: MmaIo,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    let acc = out.accumulator::<E>(
        &input,
        &weight,
        comptime!(Instruction::Mma {
            io,
            operands: MmaOperands::Plain,
        }),
        Semiring::SUM_PROD,
    );
    // The walk selects fragments by coordinate, so it is unrolled.
    let walk = space.over(&level).unrolled();
    let mut stages = Stages::smem(&walk, &input, &weight, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut acc_region = acc.at(region);
        slot.consume(|input, weight| {
            acc_region.mma(input, weight);
        });
    });
    acc.drained_into(&out);
}

#[test]
fn conv1d_mma_leaf() {
    conv1d_mma_leaf_with(MmaIo::manual());
}

#[test]
fn conv1d_mma_leaf_gathered_lhs_ignores_ldmatrix() {
    conv1d_mma_leaf_with(MmaIo {
        lhs_load_method: LoadMethod::LoadMatrix,
        ..MmaIo::manual()
    });
}

fn conv1d_mma_leaf_with(io: MmaIo) {
    let client = cubecl::test_device().client();
    // The *shape*, not just the feature: a backend can advertise manual mma and offer only
    // `16x16x16` (gfx1151 does), and running `8x8x8` there is an instruction the hardware does
    // not have: it reads back zeros, which looks like a leaf bug and is a missing guard.
    let f32_native = f32::elem_type_native();
    let offers_8x8x8 = client.properties().features.matmul.mma.iter().any(|c| {
        c.a_type == f32_native
            && c.b_type == f32_native
            && c.cd_type == f32_native
            && (c.m, c.n, c.k) == (8, 8, 8)
    });
    if !offers_8x8x8 {
        ValidationResult::Skipped("backend offers no 8x8x8 f32 manual mma".to_string()).enforce();
        return;
    }

    // `k = rh * ci = 8`, an 8x8x8 instruction, as the matmul mma test uses.
    let (oh, co, rh, ci) = (8usize, 8usize, 2usize, 4usize);
    let (stride, dilation) = (1usize, 1usize);
    let in_len = (oh - 1) * stride + (rh - 1) * dilation + 1;

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]),
            PhysicalAxisMap::of(CI),
        ],
    ));

    let f32_ty = f32::elem_type_native();
    let in_data = ramp(in_len * ci, 7);
    let w_data = ramp(rh * ci * co, 5);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![in_len, ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![rh, ci, co])
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![oh, co])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    conv_mma_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            w_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[RH, CI, CO]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[OH, CO]),
        ),
        io,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for o in 0..oh {
        for c in 0..co {
            let want: f32 = (0..rh)
                .flat_map(|r| (0..ci).map(move |i| (r, i)))
                .map(|(r, i)| {
                    let h = o * stride + r * dilation;
                    in_data[h * ci + i] * w_data[(r * ci + i) * co + c]
                })
                .sum();
            assert_eq!(
                got.get_f32(&[o, c]),
                want,
                "conv1d mma leaf: wrong at ({o}, {c})"
            );
        }
    }
}

// ---- rational (fractional scale) -------------------------------------------

/// `out[o, co] = Σ_{r, ci} w[r, ci, co] · in[⌊(o·scale + r·tap + offset) / divisor⌋, ci]`: the
/// continuous-resize mapping, where an output position lands between input cells so the window
/// steps a whole cell on some outputs and stays put on others.
struct Resize1d {
    oh: usize,
    co: usize,
    rh: usize,
    ci: usize,
    in_len: usize,
    scale: usize,
    tap: usize,
    offset: isize,
    divisor: usize,
}

impl Resize1d {
    /// The input cell tap `r` of output `o` reads, `None` outside the buffer, where
    /// [`Boundary::Zero`] contributes nothing.
    fn source(&self, o: usize, r: usize) -> Option<usize> {
        let numerator = (o * self.scale + r * self.tap) as isize + self.offset;
        let h = numerator.div_euclid(self.divisor as isize);
        (h >= 0 && (h as usize) < self.in_len).then_some(h as usize)
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let x = self.source(o, r).map_or(0.0, |h| input[h * self.ci + i]);
                        acc += x * weight[(r * self.ci + i) * self.co + c];
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    /// `OH` cut at `oh` under the leaf and the other axes whole, then one walk per count above
    /// it: a count nests a second descent, which is where a rational window's leftover phase has
    /// to accumulate rather than restart.
    fn space(&self, oh: usize, oh_counts: &[usize]) -> Launcher {
        let leaf = Levels::leaf(&[(OH, oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]);
        let tiling = oh_counts.iter().fold(leaf, |tiling, &count| {
            tiling.walk(&[(OH, count), (CO, 1), (RH, 1), (CI, 1)])
        });
        implied(
            &cubecl::test_device().client(),
            Partitioning::new(
                Space::new(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]),
                tiling.walk_every(&[OH, CO, RH, CI]).build(),
            ),
            Form::Static,
        )
    }

    fn check(&self, oh: usize, oh_counts: &[usize]) {
        self.check_with(oh, oh_counts, 1, Stage::InPlace);
    }

    /// The scalar operand staged into `width`-wide padded shared-memory lines.
    fn check_padded(&self, oh: usize, oh_counts: &[usize], width: usize) {
        self.check_with(
            oh,
            oh_counts,
            1,
            Stage::Smem {
                depth: 1,
                width: Some(width),
            },
        );
    }

    fn check_with(&self, oh: usize, oh_counts: &[usize], vector_size: usize, stage: Stage) {
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.scale), (RH, self.tap)])
                    .shifted(self.offset)
                    .over(self.divisor),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .boundary(BoundaryPolicy::Every(Boundary::Zero));

        let (got, input, weight) = run(
            shape![self.in_len, self.ci],
            shape![self.rh, self.ci, self.co],
            shape![self.oh, self.co],
            in_spec,
            &[RH, CI, CO],
            TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            self.space(oh, oh_counts),
            vector_size,
            RegisterBlock::new(16),
            stage,
        );

        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "resize1d {}/{} offset {} oh {oh} x {oh_counts:?} v {vector_size} {stage:?}: \
                     wrong at ({o}, {c})",
                    self.scale,
                    self.divisor,
                    self.offset
                );
            }
        }
    }
}

/// A gathered resample whose `CI = 3` is padded into 4-wide shared memory lines.
#[test]
fn resize1d_staged_padded_stage_width() {
    Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_padded(2, &[], 4);
}

/// The tap axis carries the divisor as its coefficient, so it survives the division whole: the
/// resample shape, `⌊(o·4 - 2) / 6⌋ + r`. The last outputs read past the input, which
/// `Boundary::Zero` masks.
#[test]
fn resize1d_rational_static() {
    Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check(2, &[]);
}

/// The same resample staged: the input tile stages uncompacted into shared
/// memory.
#[test]
fn resize1d_staged_static() {
    Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(
        2,
        &[],
        1,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Rational gather driven under double buffering on alternating slots.
#[test]
fn resize1d_staged_double_buffered() {
    Resize1d {
        oh: 8,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 6,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(
        2,
        &[],
        1,
        Stage::Smem {
            depth: 2,
            width: None,
        },
    );
}

/// A tap coefficient the divisor does not cancel: the window itself is fractionally dilated, so
/// two taps can land on one input cell. Walked over two levels, where the second descent starts
/// from the phase the first left over instead of from the projection's own offset.
#[test]
fn resize1d_rational_fractional_taps() {
    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 3,
        ci: 2,
        in_len: 8,
        scale: 5,
        tap: 2,
        offset: -2,
        divisor: 3,
    };
    resize.check(3, &[]);
    resize.check(1, &[3]);
}

/// Staged rational gather with fractional taps.
#[test]
fn resize1d_staged_fractional_taps() {
    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 3,
        ci: 2,
        in_len: 8,
        scale: 5,
        tap: 2,
        offset: -2,
        divisor: 3,
    };
    resize.check_with(
        3,
        &[],
        1,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
    resize.check_with(
        1,
        &[3],
        1,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// Staged rational gather served in two-wide lines on the ungathered `CI` axis.
#[test]
fn resize1d_staged_vectorized() {
    Resize1d {
        oh: 6,
        co: 4,
        rh: 2,
        ci: 4,
        in_len: 6,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(
        2,
        &[],
        2,
        Stage::Smem {
            depth: 1,
            width: None,
        },
    );
}

/// [`conv_kernel`] with the gathered input copied into a stage per region: a stage of an input
/// with a runtime map is born with its dynamic coefficients before any fill addresses it.
#[cube(launch)]
fn conv_kernel_input_staged(
    input: &TileArg<'_>,
    weight: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    let mut stage = input.stage(comptime!(level.clone()), StageStorage::Strided);
    for region in space.over(&level) {
        stage.copy_from(&input.at(&region));
        sync_cube();
        let mut out_region = out
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out_region.mm(&stage, &weight.at(&region));
        sync_cube();
    }
}

#[test]
fn resize1d_rational_dynamic() {
    check_resize1d_rational_dynamic(false);
}

#[test]
fn resize1d_rational_dynamic_staged() {
    check_resize1d_rational_dynamic(true);
}

fn check_resize1d_rational_dynamic(staged: bool) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    };
    let launcher = resize.space(2, &[]);

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::scaled(&[(OH, Scale::Static(4)), (RH, Scale::Static(6))])
                .shifted(Offset::Dynamic)
                .over(Divisor::Dynamic {
                    min: resize.divisor,
                }),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    let in_data = ramp(resize.in_len * resize.ci, 7);
    let w_data = ramp(resize.rh * resize.ci * resize.co, 5);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![resize.in_len, resize.ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![resize.rh, resize.ci, resize.co])
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![resize.oh, resize.co])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // The one carried coefficient is the divisor: both scales are Static, so it is the whole
    // carrier (`Projection::dynamic_divisor_index` puts it at 0).
    let input = TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec)
        .runtime_map(vec![resize.divisor as u32], vec![resize.offset as i32]);
    let weight = TileArgLaunch::new(
        w_handle.binding().into_tensor_arg(),
        f32_ty,
        1,
        TileSpec::direct(&[RH, CI, CO]),
    );
    let out = TileArgLaunch::new(
        out_handle.clone().binding().into_tensor_arg(),
        f32_ty,
        1,
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
    );
    match staged {
        true => conv_kernel_input_staged::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            input,
            weight,
            out,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
        false => conv_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            input,
            weight,
            out,
            REGISTER_BLOCK,
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
        ),
    }

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let want = resize.reference(&in_data, &w_data);
    for o in 0..resize.oh {
        for c in 0..resize.co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * resize.co + c],
                "resize1d dynamic (staged {staged}): wrong at ({o}, {c})"
            );
        }
    }
}

/// A gathered convolution with multiple reduce axes (`[RH, CI]`, where `RH = 3, CI = 3`),
/// with the gathered input staged at line width 4. The flat walk must extract units using the
/// fastest contracted axis (`CI`) coordinate, not flat step `p`.
#[test]
fn conv1d_staged_padded_multi_axis_reduce_component_indexing() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 3;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, 3), (CO, 4), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]).shifted(-(padding as isize)),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        launcher.clone(),
        1,
        RegisterBlock::new(16),
        Stage::Smem {
            depth: 1,
            width: Some(4),
        },
    );

    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "conv1d_staged_padded_multi_axis_reduce: wrong at ({o}, {c})"
            );
        }
    }
}

/// A staged padded conv with `component_fanout = true` requested on the register block.
/// The innermost reduction extent `CI = 3` is not divisible by stage line width `4`,
/// so it correctly falls back to the coordinate-decoded flat walk rather than corrupting units.
#[test]
fn conv1d_staged_padded_multi_axis_reduce_component_fanout() {
    let (in_len, ci, co, rh) = (16, 3, 4, 3);
    let (stride, dilation, padding) = (1, 1, 1);
    let oh = (in_len + 2 * padding - (rh - 1) * dilation - 1) / stride + 1;

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
            Levels::leaf(&[(OH, 3), (CO, 4), (RH, rh), (CI, ci)])
                .walk_every(&[OH, CO, RH, CI])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]).shifted(-(padding as isize)),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        launcher.clone(),
        1,
        RegisterBlock::new(16).component_fanout(),
        Stage::Smem {
            depth: 1,
            width: Some(4),
        },
    );

    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "conv1d_staged_padded_multi_axis_reduce_component_fanout: wrong at ({o}, {c})"
            );
        }
    }
}
