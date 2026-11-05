//! A convolution's gathered input staged dense over its logical axes (im2col in shared memory)
//! and contracted by tensor-core instructions, which refuse a gathered operand read in place.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::ops::matmul::MmaIo;
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
    #[comptime] instruction: Instruction,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight = weight.tile(&space);
    let out = out.tile(&space);
    for cube in space.over(&cubes) {
        let (input, weight, out) = (input.at(&cube), weight.at(&cube), out.at(&cube));
        let acc = out.accumulator::<E>(&input, &weight, instruction, Semiring::SUM_PROD);
        let walk = cube.over(&steps);
        let mut stages = Stages::smem(
            &walk,
            &input,
            &weight,
            comptime!(storage.clone()),
            1usize,
            None,
        );
        stages.pipelined(walk, |slot, region| {
            let mut acc_r = acc.at(region);
            slot.consume(|i, w| {
                acc_r.mma(i, w);
            });
        });
        acc.drained_into(&out);
    }
}

#[derive(Clone, Copy)]
enum Unit {
    Cmma,
    Mma,
}

/// `input[o * stride + r * tap + offset]` for each output `o` and tap `r`, zero outside `in_len`.
struct Case {
    oh: usize,
    stride: usize,
    tap: usize,
    offset: isize,
    in_len: usize,
}

/// The instruction's shape if the device offers it over `f16` summed in `f32`, else a skip.
fn instruction_shape(
    unit: Unit,
    client: &cubecl::client::Client,
) -> Option<(Instruction, usize, usize, usize)> {
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
    let instruction = match unit {
        Unit::Cmma => Instruction::Cmma,
        Unit::Mma => Instruction::Mma {
            io: MmaIo::manual(),
            operands: MmaOperands::Plain,
        },
    };
    Some((instruction, m, n, k))
}

impl Case {
    fn check(&self, unit: Unit) {
        let client = cubecl::test_device().client();
        let Some((instruction, m, n, k)) = instruction_shape(unit, &client) else {
            return;
        };
        let Case {
            oh,
            stride,
            tap,
            offset,
            in_len,
        } = *self;
        let (co, rh) = (CO_EXTENT, RH_EXTENT);
        // A k step is two taps by `k / 2` channels; the walk takes two steps along each.
        let ci_step = k / 2;
        let ci = 2 * ci_step;
        assert!(oh % m == 0 && co % n == 0);

        let launcher = implied(
            &client,
            Partitioning::new(
                Space::new(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)]),
                Levels::leaf(&[(OH, m), (CO, n), (RH, 2), (CI, ci_step)])
                    .walk_every(&[RH, CI])
                    .cubes(&[OH, CO])
                    .build(),
            ),
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

        let rows = PhysicalAxisMap::affine(&[(OH, stride), (RH, tap)]).shifted(offset);
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
            instruction,
            f32_ty,
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        for o in 0..oh {
            for c in 0..co {
                let mut want = 0.0f32;
                for r in 0..rh {
                    let h = (o * stride + r * tap) as isize + offset;
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
};

#[test]
fn conv_dense_stage_cmma() {
    FORWARD.check(Unit::Cmma);
}

#[test]
fn conv_dense_stage_mma() {
    FORWARD.check(Unit::Mma);
}
