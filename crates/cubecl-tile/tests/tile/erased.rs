//! A tile backed by an [`ErasedTensor`] reads and stores where a buffer-backed
//! one reads and stores.
//!
//! An erased backing never forms the address: the tile walks its layout exactly as over memory,
//! and the walk ends in a call rather than a load or store. The property to pin is that the walk
//! is *unchanged*: the same kernel over the same space and spec touches the same place either way.
//!
//! That is what makes a fused operand a drop-in for the kernel it replaces, on either side:
//! [`WriteOnly`] for fuse-on-write, [`ReadOnly`] for fuse-on-read.
//!
//! The values are `row * COLS + col` rather than anything smooth, so a touch
//! that lands one element over shows up as another cell's value instead of as a
//! near miss.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, TileInput};
use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
    zspace::shape,
};
use cubecl_tile::kind::GlobalOperand;
use cubecl_tile::kind::Write;
use cubecl_tile::layout::RuntimeGeometry;
use cubecl_tile::procedural::Procedural;
use cubecl_tile::procedural::Recipe;
use cubecl_tile::procedural::RecipeCoords;
use cubecl_tile::procedural::RecipeExpand;
use cubecl_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;

/// `row * COLS + col`.
#[derive(CubeType, Clone)]
struct Position;

#[cube]
impl<T: Float> Recipe<T> for Position {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        T::cast_from(coordinates.along(ROW)) * T::cast_from(COLS as u32)
            + T::cast_from(coordinates.along(COL))
    }
}

/// The store as it has always been: the operand's own buffer.
#[cube(launch)]
fn buffer_kernel<E: Float>(out: &TileArg<'_>, space: Partitioning, #[define(E)] _dtype: ElemType) {
    let mut dst = out.tile(&space);
    let src =
        Procedural::new::<E, Position>(comptime!(space.space().clone()), Position {}).tile::<E>();
    dst.copy_from(&src);
}

/// The same store, reached through a sink.
///
/// The sink here is backed by the very tensor the buffer kernel writes, which is
/// what makes the two comparable: nothing about the destination differs except
/// that this one is addressed through [`Tile::of_sink`] and lands in a call.
#[cube(launch)]
fn sink_kernel<E: Float>(out: &TileArg<'_>, space: Partitioning, #[define(E)] _dtype: ElemType) {
    // The geometry a sink cannot be asked for, taken off the tensor behind it.
    let geometry =
        RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(out.lines::<E, Const<1>>(), 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out.lines::<E, Const<1>>());
    let mut dst = GlobalOperand::sink::<E>(
        sink,
        geometry,
        1usize,
        comptime!(space.space().clone()),
        comptime!(out.spec.clone()),
        Write::Replace,
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    let src =
        Procedural::new::<E, Position>(comptime!(space.space().clone()), Position {}).tile::<E>();
    dst.copy_from(&src);
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

/// The nest both kernels walk, cut so the store is not one contiguous run,
/// a sink that only happened to work on a dense window would pass a flatter one.
fn space(form: Form) -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, 3)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        form,
    )
}

fn run(sink: bool) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = space(Form::Static);
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    match sink {
        true => sink_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            output_arg!(output),
            launcher.partitioning_arg(),
            dtype,
        ),
        false => buffer_kernel::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            output_arg!(output),
            launcher.partitioning_arg(),
            dtype,
        ),
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// The whole contract: the two destinations are the same walk.
#[test]
fn a_sink_stores_where_a_buffer_stores() {
    let through_sink = run(true);
    let through_buffer = run(false);
    for row in 0..ROWS {
        for col in 0..COLS {
            let expected = (row * COLS + col) as f32;
            assert_eq!(
                through_buffer.get_f32(&[row, col]),
                expected,
                "the buffer store is wrong at [{row}, {col}], so the comparison means nothing"
            );
            assert_eq!(
                through_sink.get_f32(&[row, col]),
                expected,
                "the sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

// ===========================================================================
// The host half: `Launcher::geometry`
// ===========================================================================

/// The same sink store, with the destination's [`TileSpec`] and geometry derived on the host by
/// [`Launcher::geometry`] instead of stated at the call site.
///
/// The pair a fused store actually reaches for. The kernels above take their spec off a `TileArg`,
/// but a destination written through a call has none, which is why `geometry` exists: it runs the
/// derivation a bound operand runs and hands both halves, the spec *and* the settled geometry.
#[cube(launch)]
fn derived_sink_kernel<E: Float>(
    out: &Tensor<Vector<E, Const<1>>>,
    rows: u32,
    cols: u32,
    row_stride: u32,
    col_stride: u32,
    space: Partitioning,
    #[comptime] spec: TileSpec,
    #[define(E)] _dtype: ElemType,
) {
    // The settled geometry, arriving as scalars because the host is where it was settled.
    let mut geometry = RuntimeGeometry::new();
    geometry.push(rows, row_stride);
    geometry.push(cols, col_stride);

    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(out);
    let mut dst = GlobalOperand::sink::<E>(
        sink,
        geometry,
        1usize,
        comptime!(space.space().clone()),
        spec,
        Write::Replace,
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    let src =
        Procedural::new::<E, Position>(comptime!(space.space().clone()), Position {}).tile::<E>();
    dst.copy_from(&src);
}

/// A sink whose spec and geometry both come from [`Launcher::geometry`] stores what the buffer
/// stores: the host-derived pair addresses the destination the same way the binding-derived one
/// does, which is the only thing that makes a fused store a drop-in for the kernel it replaces.
#[test]
fn a_launcher_derived_spec_addresses_the_sink() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = space(Form::Static);
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // What the destination would have been, had it been a tensor to bind.
    let derived = launcher
        .unbound(&Geometry::new(&[(ROWS, COLS), (COLS, 1)]))
        .axes(&[ROW, COL])
        .vectorize(1)
        .build_spec()
        .unwrap();
    assert_eq!(derived.geometry.shape(), [ROWS, COLS]);
    assert_eq!(derived.geometry.strides(), [COLS, 1]);

    derived_sink_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        output.clone().binding().into_tensor_arg(),
        derived.geometry.shape()[0] as u32,
        derived.geometry.shape()[1] as u32,
        derived.geometry.strides()[0] as u32,
        derived.geometry.strides()[1] as u32,
        launcher.partitioning_arg(),
        derived.spec,
        dtype,
    );

    let stored = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                stored.get_f32(&[row, col]),
                (row * COLS + col) as f32,
                "the launcher-derived sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

// ===========================================================================
// A contraction into a sink
// ===========================================================================

const M: Axis = Axis(2);
const N: Axis = Axis(3);
const K: Axis = Axis(4);
/// The register block the promoted accumulator contracts through.
const BLOCK: RegisterBlock = RegisterBlock::new(16);

/// The promoted matmul, storing to a buffer: `launch_promoted_matmul`'s twin, kept here so the
/// comparison is between two kernels in one file rather than across suites.
#[cube(launch)]
fn buffer_matmul<EA: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<EA>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    // The K steps select the one fragment by comptime coordinate, so the walk unrolls.
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
}

/// The same contraction, draining into a sink.
///
/// The accumulator is register-resident, which is the whole point: the contraction folds `K` in
/// the block and the scope's drain is the *only* touch of the destination. A drain writes each
/// cell once and never reads it back, so it asks a sink for nothing a sink does not have.
#[cube(launch)]
fn sink_matmul<E: Numeric, EA: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    // The geometry a sink cannot be asked for, taken off the tensor behind it.
    let geometry =
        RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(c.lines::<E, Const<1>>(), 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<1>>(c.lines::<E, Const<1>>());
    let c = GlobalOperand::sink::<E>(
        sink,
        geometry,
        1usize,
        comptime!(space.space().clone()),
        comptime!(c.spec.clone()),
        Write::Replace,
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    let acc = c.accumulator::<EA>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    // The K steps select the one fragment by comptime coordinate, so the walk unrolls.
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
}

/// The same contraction again, this time reading its **lhs** through an erased source.
///
/// The mirror of [`sink_matmul`], and the reason the read path had to become a view: an operand
/// tile reads through `matrix_packed`, composed onto the memory's read view as the drain is
/// onto `write_view`. The leaf asks the same layout for the same coordinates, and a call answers.
#[cube(launch)]
fn source_matmul<E: Numeric, EA: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    // The geometry a source cannot be asked for, taken off the tensor behind it.
    let geometry =
        RuntimeGeometry::of_tensor::<Vector<E, Const<1>>>(a.lines::<E, Const<1>>(), 2usize);
    let source = ErasedTensor::<E, ReadOnly>::of_tensor::<Const<1>>(a.lines::<E, Const<1>>());
    let a = GlobalOperand::source::<E>(
        source,
        geometry,
        1usize,
        comptime!(space.space().clone()),
        comptime!(a.spec.clone()),
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    let b = b.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<EA>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    // The K steps select the one fragment by comptime coordinate, so the walk unrolls.
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
}

/// Which backing the contraction under test is given.
#[derive(Clone, Copy)]
enum Backed {
    /// Every operand and the destination in memory: the control.
    Buffer,
    /// The destination erased: fuse-on-write.
    Sink,
    /// The lhs erased: fuse-on-read.
    Source,
}

/// `K` walked in four steps above a one-block leaf: every step returns to the same promoted
/// accumulator, so the destination is touched exactly once, on the drain.
fn matmul_space() -> Launcher {
    let (m, n, k, edge) = (4usize, 4usize, 16usize, 4usize);
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    )
}

fn run_matmul(backed: Backed) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = matmul_space();

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held, and a drain
    // that folded the destination in instead of writing it would show up as the poison.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    match backed {
        Backed::Sink => sink_matmul::launch(
            &client,
            count,
            dim,
            a.arg(&launcher, 1),
            b.arg(&launcher, 1),
            c.arg(&launcher, 1),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            dtype,
            dtype,
        ),
        Backed::Source => source_matmul::launch(
            &client,
            count,
            dim,
            a.arg(&launcher, 1),
            b.arg(&launcher, 1),
            c.arg(&launcher, 1),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            dtype,
            dtype,
        ),
        Backed::Buffer => buffer_matmul::launch(
            &client,
            count,
            dim,
            a.arg(&launcher, 1),
            b.arg(&launcher, 1),
            c.arg(&launcher, 1),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            dtype,
        ),
    }
    HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32)
}

/// A contraction whose accumulator is promoted stores its result through a sink exactly as it
/// stores it through a buffer.
#[test]
fn a_register_accumulator_drains_into_a_sink() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let through_sink = run_matmul(Backed::Sink);
    let through_buffer = run_matmul(Backed::Buffer);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum();
            assert!(
                (through_buffer.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the buffer contraction is wrong at [{i}, {j}], so the comparison means nothing"
            );
            assert!(
                (through_sink.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the sink contraction is wrong at [{i}, {j}]: got {}, want {expected}",
                through_sink.get_f32(&[i, j])
            );
        }
    }
}

/// The read half of the same contract: an operand read through an erased source contracts to
/// what the same operand read out of memory contracts to.
///
/// The other direction of fusion. A sink lets a kernel hand its result to a generated epilogue;
/// a source lets it take its input from a generated producer, and the leaf must not be able to
/// tell: same space, same spec, same coordinates, a call where the load was.
#[test]
fn a_contraction_reads_its_lhs_through_a_source() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let through_source = run_matmul(Backed::Source);
    let through_buffer = run_matmul(Backed::Buffer);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum();
            assert!(
                (through_buffer.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the buffer contraction is wrong at [{i}, {j}], so the comparison means nothing"
            );
            assert!(
                (through_source.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the source contraction is wrong at [{i}, {j}]: got {}, want {expected}",
                through_source.get_f32(&[i, j])
            );
        }
    }
}

// ===========================================================================
// A masked, vectorized store
// ===========================================================================

/// Five rows over a leaf of two: the last block hangs off the end, so the store is masked.
const MASKED_ROWS: usize = 5;

/// The rows overhang their leaf and the columns are served two at a time.
///
/// The two properties the aligned scalar spaces above cannot reach. Masking puts a guard between
/// the walk and the write, and a served width of two makes the tile count its innermost extent in
/// lines and re-express every coarser stride as `stride / 2`, on numbers nobody read off a tensor.
///
/// The columns stay exact and in bounds, since a vectorized innermost axis that can leave the
/// buffer is refused outright.
fn masked_space(form: Form) -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(ROW, MASKED_ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, 2)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        form,
    )
}

/// [`buffer_kernel`] at a served width of two.
///
/// The source is a bound operand rather than [`Position`]: a procedural recipe is evaluated once
/// per *line*, so at a width of two both components of a line would carry one value and the test
/// could not tell a masked store from a store one unit wide.
#[cube(launch)]
fn wide_buffer_kernel(input: &TileArg<'_>, out: &TileArg<'_>, space: Partitioning) {
    let src = input.tile(&space);
    let mut dst = out.tile(&space);
    dst.copy_from(&src);
}

/// [`sink_kernel`] at a served width of two, over the same masked nest.
#[cube(launch)]
fn wide_sink_kernel<E: Float>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(&space);
    let geometry =
        RuntimeGeometry::of_tensor::<Vector<E, Const<2>>>(out.lines::<E, Const<2>>(), 2usize);
    let sink = ErasedTensor::<E, WriteOnly>::of_tensor::<Const<2>>(out.lines::<E, Const<2>>());
    let mut dst = GlobalOperand::sink::<E>(
        sink,
        geometry,
        2usize,
        comptime!(space.space().clone()),
        comptime!(out.spec.clone()),
        Write::Replace,
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    dst.copy_from(&src);
}

/// [`wide_buffer_kernel`] with the *input* reached through an erased source, over the same
/// masked space: the read-side twin of [`wide_sink_kernel`].
#[cube(launch)]
fn wide_source_kernel<E: Float>(
    input: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let geometry =
        RuntimeGeometry::of_tensor::<Vector<E, Const<2>>>(input.lines::<E, Const<2>>(), 2usize);
    let source = ErasedTensor::<E, ReadOnly>::of_tensor::<Const<2>>(input.lines::<E, Const<2>>());
    let src = GlobalOperand::source::<E>(
        source,
        geometry,
        2usize,
        comptime!(space.space().clone()),
        comptime!(input.spec.clone()),
    )
    .tile::<E>(comptime!(space.levels().to_vec()));
    let mut dst = out.tile(&space);
    dst.copy_from(&src);
}

/// Which side of the masked, vectorized copy is erased.
#[derive(Clone, Copy)]
enum Erased {
    /// Neither: the control.
    Neither,
    /// The destination.
    Sink,
    /// The input.
    Source,
}

fn run_masked(erased: Erased) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = masked_space(Form::Dynamic);
    let input = TestInput::builder(client.clone(), shape![MASKED_ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![MASKED_ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    // Bound both ways: the width and the armed check are the launcher's derivation, so the sink
    // walks the tile the buffer kernel walks rather than one this test talked it into.
    let src = launcher
        .arg(input.binding(), dtype)
        .axes(&[ROW, COL])
        .vectorize(2)
        .build()
        .unwrap();
    let out = launcher
        .arg(output.clone().binding(), dtype)
        .axes(&[ROW, COL])
        .vectorize(2)
        .build()
        .unwrap();
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    match erased {
        Erased::Sink => wide_sink_kernel::launch(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.partitioning_arg(),
            dtype,
        ),
        Erased::Source => wide_source_kernel::launch(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.partitioning_arg(),
            dtype,
        ),
        Erased::Neither => wide_buffer_kernel::launch(
            &client,
            count,
            dim,
            src.arg(),
            out.arg(),
            launcher.partitioning_arg(),
        ),
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// A masked store through a sink writes the cells a masked store through a buffer writes.
///
/// The guard is the whole question: a sink's write ends in a call, so an overhanging unit a buffer
/// would have clipped has nothing to clip it, and a mask dropped between the walk and `write_view`
/// hands the epilogue coordinates off the product's end. The `stride / 2` width is the other half.
#[test]
fn a_masked_vectorized_sink_stores_where_a_buffer_stores() {
    let through_sink = run_masked(Erased::Sink);
    let through_buffer = run_masked(Erased::Neither);
    for row in 0..MASKED_ROWS {
        for col in 0..COLS {
            let expected = (row * COLS + col) as f32;
            assert_eq!(
                through_buffer.get_f32(&[row, col]),
                expected,
                "the buffer store is wrong at [{row}, {col}], so the comparison means nothing"
            );
            assert_eq!(
                through_sink.get_f32(&[row, col]),
                expected,
                "the masked sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

/// A masked, vectorized read through a source reads the cells a buffer read reads.
///
/// The read half of the same question, and not the same code: a buffer's masked read is the
/// slice's, while [`ErasedTensor`]'s folds an out-of-bounds index to zero, reads *that* cell, and
/// selects the mask value after.
///
/// So a guard dropped between the walk and the call does not fault here either; it returns the
/// wrong cell's value, which the `row * COLS + col` fill makes visible as another cell rather than
/// as a near miss. The width is the other half, on a geometry stated rather than read.
#[test]
fn a_masked_vectorized_source_reads_where_a_buffer_reads() {
    let through_source = run_masked(Erased::Source);
    let through_buffer = run_masked(Erased::Neither);
    for row in 0..MASKED_ROWS {
        for col in 0..COLS {
            let expected = (row * COLS + col) as f32;
            assert_eq!(
                through_buffer.get_f32(&[row, col]),
                expected,
                "the buffer read is wrong at [{row}, {col}], so the comparison means nothing"
            );
            assert_eq!(
                through_source.get_f32(&[row, col]),
                expected,
                "the masked source read the wrong value at [{row}, {col}]"
            );
        }
    }
}
