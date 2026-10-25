//! Procedural tiles accept user-defined recipes.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_common::ComptimeFloat;
use cubecl_tile::procedural::AffineCoordinate;
use cubecl_tile::procedural::Constant;
use cubecl_tile::procedural::Phase;
use cubecl_tile::procedural::Procedural;
use cubecl_tile::procedural::Product;
use cubecl_tile::procedural::Recipe;
use cubecl_tile::procedural::RecipeCoords;
use cubecl_tile::procedural::RecipeExpand;
use cubecl_tile::procedural::Sum;
use cubecl_tile::procedural::affine_along;
use cubecl_tile::procedural::product_of;
use cubecl_tile::procedural::sum_of;
use cubecl_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;

/// The rational `Phase` under test, `(2 * col - 3) / 4`, whose numerator goes negative so the
/// floor is not the truncating division.
const SCALE: u32 = 2;
const OFFSET: i32 = -3;
const DIVISOR: u32 = 4;

#[derive(CubeType, Clone)]
struct AxisValue {
    #[cube(comptime)]
    axis: Axis,
    scale: f32,
}

#[cube]
impl<T: Float> Recipe<T> for AxisValue {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        T::cast_from(coordinates.along(self.axis)) * T::cast_from(self.scale)
    }
}

/// Walk the whole space and copy each region of `source`, evaluated where it is read, into
/// `output`.
#[cube]
fn materialize(
    source: &Tile,
    output: &TileArg<'_>,
    space: &Partitioning,
    #[comptime] level: Level,
) {
    let output = output.tile(space);
    for region in source.over(&level) {
        let mut output_region = output.at(&region);
        output_region.copy_from(&source.at(&region));
    }
}

/// Forces a scalar value into a runtime GPU variable by adding a runtime zero,
/// preventing the expander from evaluating it away at compile time.
#[cube]
fn runtime_scalar<E: Numeric>(value: E) -> E {
    value + E::cast_from(0u32.runtime())
}

/// `x = offset + coordinate[COL]`, built so the recipe carries genuinely runtime scalars across
/// the staging walk rather than evaluating at comptime.
#[cube]
fn along_col<E: Float>(#[comptime] offset: ComptimeFloat<f32>) -> AffineCoordinate<E> {
    affine_along(
        COL,
        runtime_scalar::<E>(E::new(comptime!(f64::from(offset.get())))),
        runtime_scalar::<E>(E::new(1.0)),
    )
}

/// The product recipe evaluated where it is read: every region of the walk copies its window of
/// the source straight into the output.
#[cube(launch)]
fn product_kernel_in_place<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Product<AffineCoordinate<E>, AffineCoordinate<E>>>(
        comptime!(space.space().clone()),
        product_of(
            affine_along(ROW, E::from_int(0), E::from_int(1)),
            affine_along(COL, E::from_int(0), E::from_int(1)),
        ),
    )
    .tile::<E>();
    materialize(&source, output, &space, level.clone());
}

/// The same recipe materialized into shared memory first: one slot fills each region's
/// window cooperatively, and the output copies the stage. The grid must be the same either way.
#[cube(launch)]
fn product_kernel_staged<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Product<AffineCoordinate<E>, AffineCoordinate<E>>>(
        comptime!(space.space().clone()),
        product_of(
            affine_along(ROW, E::from_int(0), E::from_int(1)),
            affine_along(COL, E::from_int(0), E::from_int(1)),
        ),
    )
    .tile::<E>();
    let output = output.tile(&space);
    let walk = source.over(&level);
    let mut stages = Stages::smem_single(&walk, &source, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut output_region = output.at(region);
        slot.consume(|staged| {
            output_region.copy_from(staged);
        });
    });
}

/// The recipe under test: `row - frac((col * scale + offset) / divisor)`, a two-axis expression
/// built by composition, which is the shape a resampling weight takes.
#[cube]
fn affine_plus_phase<E: Float>(
    #[comptime] space: Space,
    scale: u32,
    offset: i32,
    divisor: u32,
) -> Tile {
    Procedural::new::<E, Sum<AffineCoordinate<E>, Phase<E>>>(
        space,
        sum_of(
            affine_along(ROW, E::from_int(0), runtime_scalar::<E>(E::new(1.0))),
            Phase::<E> {
                coefficient: runtime_scalar::<E>(E::new(-1.0)),
                axis: COL,
                numerator_scale: scale,
                numerator_offset: offset,
                divisor,
            },
        ),
    )
    .tile::<E>()
}

/// `launch_ratio` decides whether the fraction is spelled in constants, which evaluate at expand time,
/// or in values the kernel only receives at launch. Both go through the same `Phase`, and the grid
/// must come out the same.
#[cube(launch)]
fn phase_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] launch_ratio: bool,
    #[define(E)] _dtype: ElemType,
) {
    let source = if comptime!(launch_ratio) {
        affine_plus_phase::<E>(
            comptime!(space.space().clone()),
            SCALE.runtime(),
            OFFSET.runtime(),
            DIVISOR.runtime(),
        )
    } else {
        affine_plus_phase::<E>(comptime!(space.space().clone()), SCALE, OFFSET, DIVISOR)
    };
    materialize(&source, output, &space, level.clone());
}

#[cube(launch)]
fn rebase_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, AxisValue>(
        comptime!(space.space().clone()),
        AxisValue {
            axis: ROW,
            scale: 2.0,
        },
    )
    .tile::<E>();
    materialize(&source, output, &space, level);
}

#[cube(launch)]
fn constant_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Constant<E>>(
        comptime!(space.space().clone()),
        Constant::<E> {
            value: runtime_scalar::<E>(E::new(-1.25)),
        },
    )
    .tile::<E>();
    materialize(&source, output, &space, level.clone());
}

#[cube(launch)]
fn affine_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] offset: ComptimeFloat<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, AffineCoordinate<E>>(
        comptime!(space.space().clone()),
        along_col::<E>(offset),
    )
    .tile::<E>();
    materialize(&source, output, &space, level.clone());
}

/// A procedural tile over an integer element type: [`Recipe`] is defined over `Numeric`, so
/// nothing about a procedural source requires floats.
#[cube(launch)]
fn integer_kernel<E: Int>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Constant<E>>(
        comptime!(space.space().clone()),
        Constant::<E> {
            value: runtime_scalar::<E>(E::new(7)),
        },
    )
    .tile::<E>();
    materialize(&source, output, &space, level.clone());
}

/// A direct procedural read must use the masked view path on trailing partial tiles.
#[cube(launch)]
fn direct_copy_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Constant<E>>(
        comptime!(space.space().clone()),
        Constant::<E> {
            value: runtime_scalar::<E>(E::new(1.0)),
        },
    )
    .tile::<E>();
    let mut output = output.tile(&space);
    output.copy_from(&source);
}

/// Descend once before a direct read: the procedural bound remains in the parent coordinates.
#[cube(launch)]
fn divided_direct_copy_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::new::<E, Constant<E>>(
        comptime!(space.space().clone()),
        Constant::<E> {
            value: runtime_scalar::<E>(E::new(1.0)),
        },
    )
    .tile::<E>();
    let region = Region::trailing(
        comptime!(0usize),
        comptime!(space.space().clone()),
        comptime!(level.clone()),
        0usize,
        0usize,
    );
    let source = source.at(&region);
    let output = output.tile(&space);
    let mut output = output.at(&region);
    output.copy_from(&source);
}

struct Harness {
    client: Client,
    dtype: ElemType,
    launcher: Launcher,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: cubecl::test_device().client(),
            dtype: f32::elem_type_native(),
            launcher: implied(
                &cubecl::test_device().client(),
                Partitioning::new(
                    Space::new(&[(ROW, ROWS), (COL, COLS)]),
                    Levels::leaf(&[(ROW, 2), (COL, 3)])
                        .walk_every(&[ROW, COL])
                        .build(),
                ),
                Form::Static,
            ),
        }
    }

    fn output(&self) -> TensorHandle {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle) -> HostData {
        HostData::from_tensor_handle(&self.client, output, HostDataType::F32)
    }
}

/// The output launch argument, served at the output's element.
macro_rules! output_arg {
    ($output:expr) => {
        TileArgLaunch::new(
            $output.clone().binding().into_tensor_arg(),
            $output.dtype,
            1,
            TileSpec::direct(&[ROW, COL]),
        )
    };
}

/// Assert every cell equals `expected(row, col)`, tolerating float rounding.
fn assert_grid(got: &HostData, expected: impl Fn(usize, usize) -> f32) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let want = expected(row, col);
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-5,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}

/// A finite comptime offset for [`along_col`].
fn offset(value: f32) -> ComptimeFloat<f32> {
    ComptimeFloat::new(value).unwrap()
}

#[test]
fn user_recipe_evaluates_in_place() {
    let h = Harness::new();
    let output = h.output();
    product_kernel_in_place::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        h.dtype,
    );
    assert_grid(&h.read(output), |row, col| (row * col) as f32);
}

#[test]
fn user_recipe_materializes_through_a_staged_walk() {
    let h = Harness::new();
    let output = h.output();
    product_kernel_staged::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        h.dtype,
    );
    assert_grid(&h.read(output), |row, col| (row * col) as f32);
}

#[test]
fn selecting_a_region_rebases_the_recipe_origin() {
    let h = Harness::new();
    let output = h.output();
    rebase_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        h.dtype,
    );
    assert_grid(&h.read(output), |row, _| 2.0 * row as f32);
}

/// Walk the affine-plus-phase recipe with the fraction constant or passed at launch; the grid is the
/// same either way.
fn check_phase(launch_ratio: bool) {
    let h = Harness::new();
    let output = h.output();
    phase_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        launch_ratio,
        h.dtype,
    );
    assert_grid(&h.read(output), |row, col| {
        let numerator = (SCALE * col as u32) as i32 + OFFSET;
        row as f32 - numerator.rem_euclid(DIVISOR as i32) as f32 / DIVISOR as f32
    });
}

#[test]
fn a_sum_of_an_affine_term_and_a_phase_reads_both_axes() {
    check_phase(false);
}

#[test]
fn a_phase_takes_a_ratio_fixed_only_at_launch() {
    check_phase(true);
}

#[test]
fn constant_evaluates_its_value_everywhere() {
    let h = Harness::new();
    let output = h.output();
    constant_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        h.dtype,
    );
    assert_grid(&h.read(output), |_, _| -1.25);
}

#[test]
fn affine_coordinates_evaluate_absolute_positions() {
    let h = Harness::new();
    let output = h.output();
    affine_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
        offset(-2.5),
        h.dtype,
    );
    // Constant down each column: the recipe reads COL only, and the value must hold for the lower
    // regions of the walk, whose origin the rebasing has moved.
    assert_grid(&h.read(output), |_, col| -2.5 + col as f32);
}

#[test]
fn a_procedural_tile_works_over_an_integer_element_type() {
    let client = cubecl::test_device().client();
    let dtype = i32::elem_type_native();
    let launcher = Harness::new().launcher;
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    integer_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        output_arg!(output),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::I32);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(got.get_i32(&[row, col]), 7);
        }
    }
}

#[test]
fn direct_copy_masks_the_trailing_partial_tile() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, 4)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        Form::Static,
    );
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    direct_copy_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        output_arg!(output),
        launcher.partitioning_arg(),
        dtype,
    );

    assert_grid(
        &HostData::from_tensor_handle(&client, output, HostDataType::F32),
        |_, _| 1.0,
    );
}

#[test]
fn divided_direct_copy_preserves_the_parent_bound() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, 4)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        Form::DynamicAlong(&[ROW]),
    );
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    divided_direct_copy_kernel::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        output_arg!(output),
        launch.partitioning_arg(),
        launch.partitioning().level(0),
        dtype,
    );

    assert_grid(
        &HostData::from_tensor_handle(&client, output, HostDataType::F32),
        |row, col| if row < 2 && col < 4 { 1.0 } else { 0.0 },
    );
}
