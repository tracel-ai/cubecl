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
use crate::support::{HostData, HostDataType, TestInput, TileInput};
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
    input: &TileArg<'_>,
    out: &O::Arg<E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = input.tile(&space);
    let mut dst = O::tile::<E, V>(out, &space);
    dst.copy_from(&src);
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

/// The promoted matmul over any destination: `K` folds in registers and the drain is the only
/// touch of the output, which is the shape of a kernel a fused epilogue is handed to.
#[cube(launch)]
fn contract<E: Numeric, EA: Numeric, O: Destination>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &O::Arg<E>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = O::tile::<E, Const<1>>(c, &space);
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
    // Poisoned, not zeroed: a drain that folded the destination in would show the poison.
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
