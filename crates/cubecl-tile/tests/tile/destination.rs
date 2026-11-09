//! An output this crate does not know, plugged in through [`Destination`]: a kernel written over
//! `O: Destination` stores into [`Output`](cubecl_tile::launch::Output) ([`Buffered`]) or into a
//! caller's own argument with the same body.
//!
//! The caller's destination here is everything a fused epilogue is, built from this file alone: a
//! launch argument of its own, a geometry *stated* at the launch rather than read off a buffer
//! (the one [`Launcher::unbound`] settles), and a backing that runs a function at the store. The
//! function doubles each value on its way into the tensor behind it, so a store that landed
//! anywhere but through the sink shows up as an undoubled value.
//!
//! The backing declares [`WritesLines`] and nothing else, and a tile over a write-only backing
//! refuses a read while the kernel is being defined. A kernel that read its output would never
//! run, and its destination would stay as it was, so every doubled value asserted here is also
//! the proof that the kernel only ever stored into it.

use core::marker::PhantomData;

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, TileInput, hands_off_between_cubes};
use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, WriteOnly, WritesLines,
};
use cubecl::{prelude::*, unexpanded, zspace::shape};
use cubecl_tile::kind::{GlobalOperand, Write};
use cubecl_tile::launch::{Buffered, Destination, DestinationLaunch, Unbound};
use cubecl_tile::layout::RuntimeGeometry;
use cubecl_tile::*;

// ===========================================================================
// The caller's destination
// ===========================================================================

/// A destination whose store doubles each value into the tensor behind it: a fused epilogue in
/// miniature.
struct Doubled;

/// [`Doubled`]'s launch argument: where the doubled values land, and the output's geometry and
/// spec, which a sink has no address to answer for.
#[derive(CubeType, CubeLaunch)]
struct DoubledArg<'a, E: Numeric> {
    values: &'a Tensor<E>,
    shape: Sequence<u32>,
    strides: Sequence<u32>,
    #[cube(comptime)]
    rank: usize,
    #[cube(comptime)]
    spec: TileSpec,
}

#[cube]
impl Destination for Doubled {
    type Arg<E: Numeric> = DoubledArg<'static, E>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for axis in 0..comptime!(arg.rank) {
            geometry.push(*arg.shape.index(axis), *arg.strides.index(axis));
        }
        let sink = doubling_sink::<E, V>(arg.values);
        // Read at expansion, not as a Rust constant: a launch-time `Size` has no `value()` until
        // the kernel is being defined.
        let width = V::value();
        GlobalOperand::sink::<E>(
            sink,
            geometry,
            width,
            comptime!(partitioning.space().clone()),
            comptime!(arg.spec.clone()),
            Write::Replace,
        )
        .tile::<E>(comptime!(partitioning.levels().to_vec()))
    }

    fn finish<E: Numeric>(_arg: &Self::Arg<E>) {}
}

/// What [`Doubled`] launches from: the tensor the values land in, and what the launcher derived
/// for an output with no tensor to bind.
struct DoubledOperand {
    values: TensorArg,
    sink: Unbound,
}

impl DestinationLaunch for Doubled {
    type Operand = DoubledOperand;

    fn arg<E: Numeric>(operand: DoubledOperand) -> DoubledArgLaunch<'static, E> {
        let Unbound { spec, geometry, .. } = operand.sink;
        DoubledArgLaunch::new(
            operand.values,
            geometry.shape().iter().map(|&e| e as u32).collect(),
            geometry.strides().iter().map(|&s| s as u32).collect(),
            geometry.rank(),
            spec,
        )
    }
}

/// The write-only erased tensor over `values`, doubling each line of `N` it is handed.
#[allow(clippy::extra_unused_type_parameters)]
fn doubling_sink<E: Numeric, N: Size>(_values: &Tensor<E>) -> ErasedTensor<E, WriteOnly> {
    unexpanded!()
}

mod doubling_sink {
    use super::*;

    pub fn expand<E: Numeric, N: Size>(
        _scope: &Scope,
        values: &<Tensor<E> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(Doubling::<E, N> {
            values: ExpandTypeClone::clone_unchecked(values),
            _n: PhantomData,
        })
    }
}

/// The backing: writes, never reads. Only [`WritesLines`] is declared, so the read half is the
/// trait's default, which panics at expansion.
struct Doubling<E: Numeric, N: Size> {
    values: <Tensor<E> as CubeType>::ExpandType,
    _n: PhantomData<N>,
}

impl<E: Numeric, N: Size> ErasedTensorOperationsExpand<E> for Doubling<E, N> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        <N as Size>::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.values.__expand_len_method(scope);
        let width = <N as Size>::__expand_value(scope).__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, width)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        store_doubled::expand::<E, N>(scope, &mut self.values, index, value.into());
    }
}

impl<E: Numeric, N: Size> WritesLines<E> for Doubling<E, N> {}

/// One line, doubled, at the line's own offset: `N` scalars.
#[cube]
fn store_doubled<E: Numeric, N: Size>(values: &mut Tensor<E>, index: usize, value: Vector<E, N>) {
    let doubled = value + value;
    let base = index * N::value();
    #[unroll]
    for lane in 0..N::value() {
        values[base + lane] = doubled.extract(lane);
    }
}

// ===========================================================================
// A store, generic over its destination
// ===========================================================================

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;
/// Two values a line, so the sink is handed lines rather than scalars.
const WIDTH: usize = 2;

/// One kernel for every destination: the launch picks `O`.
///
/// The source is bound rather than procedural: a recipe is evaluated once per *line*, so at a
/// width of two both values of a line would be one, and a store a line off would go unseen.
#[cube(launch)]
fn store<E: Float, V: Size, O: Destination>(
    input: &TileTensor<'_>,
    out: &O::Arg<E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(&space);
    let mut dst = O::tile::<E, V>(out, &space);
    dst.copy_from(&src);
    O::finish::<E>(out);
}

/// Cut so the store is not one contiguous run.
fn store_space() -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, 3 * WIDTH)])
                .walk_every(&[ROW, COL])
                .build(),
        ),
        Form::Static,
    )
}

fn run_store(sink: bool) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = store_space();
    // `row * COLS + col`: a touch one cell over shows up as another cell's value.
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let src = launcher
        .arg(input.binding(), dtype)
        .axes(&[ROW, COL])
        .vectorize(WIDTH)
        .build()
        .unwrap();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    match sink {
        true => {
            let derived = launcher
                .unbound(&Geometry::new(&[(ROWS, COLS), (COLS, 1)]))
                .axes(&[ROW, COL])
                .vectorize(WIDTH)
                .build_spec()
                .unwrap();
            let operand = DoubledOperand {
                values: output.clone().binding().into_tensor_arg(),
                sink: derived,
            };
            store::launch::<Doubled>(
                &client,
                count,
                dim,
                WIDTH,
                src.arg(),
                Doubled::arg(operand),
                launcher.partitioning_arg(),
                dtype,
            )
        }
        false => {
            let bound = launcher
                .arg(output.clone().binding(), dtype)
                .axes(&[ROW, COL])
                .vectorize(WIDTH)
                .build()
                .unwrap();
            store::launch::<Buffered>(
                &client,
                count,
                dim,
                WIDTH,
                src.arg(),
                Buffered::arg((bound, Write::Replace)),
                launcher.partitioning_arg(),
                dtype,
            )
        }
    }
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// The same kernel stores through [`Buffered`] and through the caller's sink, the sink's epilogue
/// running on every value it is handed.
#[test]
fn one_kernel_stores_into_a_buffer_or_a_caller_sink() {
    let through_buffer = run_store(false);
    let through_sink = run_store(true);
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
                2.0 * expected,
                "the sink stored the wrong value at [{row}, {col}]"
            );
        }
    }
}

// ===========================================================================
// A contraction draining into the sink
// ===========================================================================

const M: Axis = Axis(2);
const N: Axis = Axis(3);
const K: Axis = Axis(4);
const BLOCK: RegisterBlock = RegisterBlock::new(16);

/// The promoted matmul over any destination: `K` sums in registers and the drain is the only
/// touch of the output, which is the shape of a kernel a fused epilogue is handed to.
#[cube(launch)]
fn contract<E: Numeric, EA: Numeric, O: Destination>(
    a: &TileTensor<'_>,
    b: &TileTensor<'_>,
    out: &O::Arg<E>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = O::tile::<E, Const<1>>(out, &space);
    let acc = c.accumulator::<EA>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
    O::finish::<E>(out);
}

fn run_contract(sink: bool) -> HostData {
    let (m, n, k, edge) = (4usize, 4usize, 16usize, 4usize);
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: a drain that summed the destination in would show the poison.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    let level = launcher.partitioning().level(0);
    match sink {
        true => {
            let derived = launcher
                .unbound(&Geometry::new(&[(m, n), (n, 1)]))
                .axes(&[M, N])
                .build_spec()
                .unwrap();
            let operand = DoubledOperand {
                values: c.handle().binding().into_tensor_arg(),
                sink: derived,
            };
            contract::launch::<Doubled>(
                &client,
                count,
                dim,
                a.arg(&launcher, 1),
                b.arg(&launcher, 1),
                Doubled::arg(operand),
                launcher.partitioning_arg(),
                level,
                dtype,
                dtype,
            )
        }
        false => {
            let bound = launcher
                .arg(c.handle().binding(), dtype)
                .axes(&[M, N])
                .build()
                .unwrap();
            contract::launch::<Buffered>(
                &client,
                count,
                dim,
                a.arg(&launcher, 1),
                b.arg(&launcher, 1),
                Buffered::arg((bound, Write::Replace)),
                launcher.partitioning_arg(),
                level,
                dtype,
                dtype,
            )
        }
    }
    HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32)
}

/// A register accumulator drains into the caller's sink as it drains into a buffer, the epilogue
/// applied once to each finished cell and the poisoned destination never read.
#[test]
fn a_contraction_drains_into_a_caller_sink() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let through_buffer = run_contract(false);
    let through_sink = run_contract(true);
    // Row-major arange operands: lhs(i, p) = i·k + p, rhs(p, j) = p·n + j.
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum();
            assert!(
                (through_buffer.get_f32(&[i, j]) - expected).abs() < 1e-3,
                "the buffer contraction is wrong at [{i}, {j}], so the comparison means nothing"
            );
            assert!(
                (through_sink.get_f32(&[i, j]) - 2.0 * expected).abs() < 1e-3,
                "the sink contraction is wrong at [{i}, {j}]: got {}, want {}",
                through_sink.get_f32(&[i, j]),
                2.0 * expected
            );
        }
    }
}

// ===========================================================================
// A destination that finishes
// ===========================================================================

/// [`Doubled`] that also totals what the launch stored once every unit has finished.
struct Totaled;

/// [`Totaled`]'s launch argument: the doubling destination, the units that have arrived, and the
/// cell the total lands in. The total is read from the one binding the stores go through.
#[derive(CubeType, CubeLaunch)]
struct TotaledArg<'a, E: Numeric> {
    doubled: DoubledArg<'a, E>,
    arrivals: &'a [Atomic<u32>],
    total: &'a [Atomic<u32>],
}

/// Every unit counts itself in after publishing its stores. The cube holding the last unit to
/// count in acquires and sums, the protocol of a merge by the last arrival.
#[cube]
impl Destination for Totaled {
    type Arg<E: Numeric> = TotaledArg<'static, E>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        Doubled::tile::<E, V>(&arg.doubled, partitioning)
    }

    fn finish<E: Numeric>(arg: &Self::Arg<E>) {
        sync_storage();
        let before = arg.arrivals[0].fetch_add(1);
        let mut last = Shared::<[u32]>::new_slice(1usize);
        if UNIT_POS == 0 {
            last[0] = 0;
        }
        sync_cube();
        if before as usize + 1 == CUBE_COUNT * CUBE_DIM as usize {
            last[0] = 1;
        }
        sync_cube();
        // Uniform across the cube: only the cube holding the last unit sums.
        if last[0] == 1 {
            sync_storage();
            if UNIT_POS == 0 {
                let mut sum = E::from_int(0);
                for i in 0..arg.doubled.values.len() {
                    sum += arg.doubled.values[i];
                }
                arg.total[0].store(u32::cast_from(sum));
            }
        }
    }
}

/// What [`Totaled`] launches from.
struct TotaledOperand {
    doubled: DoubledOperand,
    arrivals: BufferArg,
    total: BufferArg,
}

impl DestinationLaunch for Totaled {
    type Operand = TotaledOperand;

    fn arg<E: Numeric>(operand: TotaledOperand) -> TotaledArgLaunch<'static, E> {
        TotaledArgLaunch::new(
            Doubled::arg(operand.doubled),
            operand.arrivals,
            operand.total,
        )
    }
}

/// What [`Totaled`] left after [`store`] over [`store_space`] with `extra` cubes beyond the one
/// its grid needs: the stored values, the units that arrived, the units launched and the total;
/// `None` where the device cannot hand one cube's stores to another. The walk has no cube level,
/// so every extra cube repeats the whole store.
fn run_totaled(extra: u32) -> Option<(HostData, u32, u32, u32)> {
    let client = cubecl::test_device().client();
    if !hands_off_between_cubes(&client) {
        return None;
    }
    let dtype = f32::elem_type_native();
    let launcher = store_space();
    let CubeCount::Static(x, y, z) = launcher.cube_count() else {
        panic!("a static space spawns a static grid");
    };
    let dim = launcher.cube_dim();
    let units = (x + extra) * y * z * dim.num_elems();
    let input = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let src = launcher
        .arg(input.binding(), dtype)
        .axes(&[ROW, COL])
        .vectorize(WIDTH)
        .build()
        .unwrap();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let derived = launcher
        .unbound(&Geometry::new(&[(ROWS, COLS), (COLS, 1)]))
        .axes(&[ROW, COL])
        .vectorize(WIDTH)
        .build_spec()
        .unwrap();
    let arrivals = client.create_from_slice(u32::as_bytes(&[0u32]));
    let total = client.create_from_slice(u32::as_bytes(&[0u32]));
    let out = Totaled::arg(TotaledOperand {
        doubled: DoubledOperand {
            values: output.clone().binding().into_tensor_arg(),
            sink: derived,
        },
        arrivals: unsafe { BufferArg::from_raw_parts(arrivals.clone(), 1) },
        total: unsafe { BufferArg::from_raw_parts(total.clone(), 1) },
    });
    store::launch::<Totaled>(
        &client,
        CubeCount::Static(x + extra, y, z),
        dim,
        WIDTH,
        src.arg(),
        out,
        launcher.partitioning_arg(),
        dtype,
    );
    let stored = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let arrived = u32::from_bytes(&client.read_one_unchecked(arrivals))[0];
    let total = u32::from_bytes(&client.read_one_unchecked(total))[0];
    Some((stored, arrived, units, total))
}

/// Every unit of every cube, extra cubes included, reaches [`Destination::finish`], and exactly
/// one cube finds itself holding the last unit and totals the output. Every cube stores every
/// value here, so it does not prove a store is visible across cubes.
#[test]
fn every_unit_reaches_finish_and_the_last_cube_totals() {
    let cells = ROWS * COLS;
    let want = (cells * (cells - 1)) as u32;
    for extra in [0, 3] {
        let Some((stored, arrived, units, total)) = run_totaled(extra) else {
            continue;
        };
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(
                    stored.get_f32(&[row, col]),
                    2.0 * (row * COLS + col) as f32,
                    "{extra} extra cubes: the store is wrong at [{row}, {col}]"
                );
            }
        }
        assert_eq!(arrived, units, "{extra} extra cubes: units that finished");
        assert_eq!(total, want, "{extra} extra cubes: the total");
    }
}
