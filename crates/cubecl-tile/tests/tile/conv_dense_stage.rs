//! A convolution's gathered input staged dense over its dimensions (im2col in shared memory)
//! and contracted by tensor-core instructions, which refuse a gathered operand read in place.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::layout::{PhysicalAxisMap, Scale};
use cubecl_tile::stage::RowChunks;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};
use half::f16;

use super::{Form, implied};

const OH: Dimension = Dimension(0);
const CO: Dimension = Dimension(1);
const RH: Dimension = Dimension(2);
const CI: Dimension = Dimension(3);

// A folded 2-D convolution's output positions, its output columns (`OH` its rows), and its
// second tap.
const P: Dimension = Dimension(4);
const OW: Dimension = Dimension(5);
const RW: Dimension = Dimension(6);

const CO_EXTENT: usize = 16;
const RH_EXTENT: usize = 4;

#[cube(launch)]
fn conv_dense_stage_kernel<E: Numeric>(
    input: &TileTensor<'_>,
    weight: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] storage: StageStorage,
    #[comptime] planes: usize,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile_mut(&space);
    for cube in space.over(&cubes) {
        let (input, weight, out) = (input.at(&cube), weight.at(&cube), out.at(&cube));
        let acc = out.accumulator::<E>(&input, &weight, Semiring::SumProd);
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
            order: StageOrder::AsSource,
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
/// row dimension, `taps * channels`, not the channels of one tap.
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

/// A 2-D convolution whose output positions are one dimension, `P = oh * ow` (`Projection::folded`
/// on the input and the output), the input's rows and columns each read off one digit of it,
/// zero outside the input:
/// `in[(p / ow) * sh + rh * dh - ph, (p % ow) * sw + rw * dw - pw]`.
struct Folded {
    out: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    /// Row blocks the plane level lays side by side, which cuts `P` a second time.
    planes: usize,
}

impl Folded {
    const TAPS: usize = 3;

    fn input(&self) -> [usize; 2] {
        [0, 1].map(|a| {
            (self.out[a] - 1) * self.stride[a] + (Self::TAPS - 1) * self.dilation[a] + 1
                - 2 * self.padding[a]
        })
    }

    /// The input cell output position `p` reads at taps `(rh, rw)`, `None` in the padding.
    fn source(&self, p: usize, taps: [usize; 2]) -> Option<[usize; 2]> {
        let at = [p / self.out[1], p % self.out[1]];
        let input = self.input();
        let mut cell = [0; 2];
        for a in 0..2 {
            let x = (at[a] * self.stride[a] + taps[a] * self.dilation[a]) as isize
                - self.padding[a] as isize;
            if x < 0 || x as usize >= input[a] {
                return None;
            }
            cell[a] = x as usize;
        }
        Some(cell)
    }

    fn check(&self, unit: Unit) {
        let client = cubecl::test_device().client();
        let Some((m, n, k)) = instruction_shape(unit, &client) else {
            return;
        };
        let positions = self.out[0] * self.out[1];
        let (co, rh, ci) = (CO_EXTENT, Self::TAPS, 2 * k);
        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&[(P, positions), (CO, co), (RH, rh), (RW, rh), (CI, ci)]),
                {
                    let levels = Levels::leaf(&[(P, m), (CO, n), (RH, 1), (RW, 1), (CI, k)]);
                    let levels = match self.planes {
                        1 => levels,
                        planes => levels.planes(&[(P, planes)]),
                    };
                    levels.walk_every(&[RH, RW, CI]).cubes(&[P, CO]).build()
                },
            ),
            Form::Static,
        );
        let storage = StageStorage::Tiled {
            block: launcher.partitioning().leaf().extents(),
            chunks: RowChunks::InOrder,
            order: StageOrder::AsSource,
        };

        let [ih, iw] = self.input();
        let in_data: Vec<f32> = (0..ih * iw * ci).map(|i| (i % 7) as f32 - 1.0).collect();
        let w_data: Vec<f32> = (0..rh * rh * ci * co)
            .map(|i| (i % 5) as f32 - 1.0)
            .collect();
        let (f16_ty, f32_ty) = (f16::elem_type_native(), f32::elem_type_native());
        let (in_handle, _) = TestInput::builder(client.clone(), shape![ih, iw, ci])
            .dtype(f16_ty)
            .custom(in_data.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), shape![rh, rh, ci, co])
            .dtype(f16_ty)
            .custom(w_data.clone())
            .generate_with_f32_host_data();
        let [oh, ow] = self.out;
        let out_handle = TestInput::builder(client.clone(), shape![oh, ow, co])
            .dtype(f32_ty)
            .uniform(4242, 10., 100.)
            .generate_without_host_data();

        let fold = [(OH, oh), (OW, ow)];
        let rows = PhysicalAxisMap::affine(&[(OH, self.stride[0]), (RH, self.dilation[0])])
            .shifted(-(self.padding[0] as isize));
        let columns = PhysicalAxisMap::affine(&[(OW, self.stride[1]), (RW, self.dilation[1])])
            .shifted(-(self.padding[1] as isize));
        let in_spec = TileSpec::new(
            Projection::new(
                &[OH, OW, RH, RW, CI],
                &[rows, columns, PhysicalAxisMap::of(CI)],
            )
            .folded(P, &fold),
        )
        .boundary(BoundaryPolicy::Every(Boundary::Zero));
        let out_spec = TileSpec::new(Projection::direct(&[OH, OW, CO]).folded(P, &fold))
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
                TileSpec::direct(&[RH, RW, CI, CO]),
            ),
            TileTensorLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                f32_ty,
                1,
                out_spec,
            ),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            storage,
            self.planes,
            f32_ty,
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        for p in 0..positions {
            for c in 0..co {
                let mut want = 0.0f32;
                for r in 0..rh {
                    for s in 0..rh {
                        let Some([h, w]) = self.source(p, [r, s]) else {
                            continue;
                        };
                        for i in 0..ci {
                            want += in_data[(h * iw + w) * ci + i]
                                * w_data[((r * rh + s) * ci + i) * co + c];
                        }
                    }
                }
                assert_eq!(
                    got.get_f32(&[p / ow, p % ow, c]),
                    want,
                    "wrong at ({p}, {c})"
                );
            }
        }
    }
}

/// An 8 by 8 map padded to its own size: whole row tiles, each spanning several image rows.
const FOLDED_SAME: Folded = Folded {
    out: [8, 8],
    stride: [1, 1],
    dilation: [1, 1],
    padding: [1, 1],
    planes: 1,
};

#[test]
fn conv_dense_stage_cmma_folded() {
    FOLDED_SAME.check(Unit::Cmma);
}

/// Every pair unequal, so the rows and columns cannot be confused; a 5 by 9 map overhangs too.
#[test]
fn conv_dense_stage_cmma_folded_asymmetric() {
    Folded {
        out: [5, 9],
        stride: [1, 2],
        dilation: [2, 1],
        padding: [2, 0],
        planes: 1,
    }
    .check(Unit::Cmma);
}

/// The plane level cuts `P` again under the cube's cut, so the window's start along it adds up.
#[test]
fn conv_dense_stage_cmma_folded_planes() {
    Folded {
        planes: 2,
        ..FOLDED_SAME
    }
    .check(Unit::Cmma);
}
