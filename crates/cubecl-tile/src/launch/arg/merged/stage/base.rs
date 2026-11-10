//! The stage a store passes through: it sees each stored vector with its place and returns the vector
//! to store.

use cubecl::prelude::*;

/// Where a stored vector sits in a rank 2 output. Its row and column are worked out only by a stage
/// that asks, so a stage that ignores them costs no division.
#[derive(CubeType, Clone, Copy)]
pub struct VectorAt {
    first: usize,
    row_stride: usize,
}

#[cube]
impl VectorAt {
    pub(crate) fn new(first: usize, row_stride: usize) -> VectorAt {
        VectorAt { first, row_stride }
    }

    /// The row of the vector's first cell.
    pub fn row(&self) -> usize {
        self.first / self.row_stride
    }

    /// The column of the vector's first cell.
    pub fn column(&self) -> usize {
        self.first % self.row_stride
    }
}

/// What happens to each vector a [`Merged`](crate::launch::Merged) destination's merge writes out, so a stage
/// sees each cell's final value once. A vector never crosses a row, and vectors arrive in no particular
/// order and in divergent control flow, so a stage may not use plane operations or synchronize.
#[cube]
pub trait VectorStage: Send + Sync + 'static {
    /// The launch argument the stage reads and writes through, at element `E`.
    type Arg<E: Numeric>: LaunchArg + CubeType;

    /// The vector to store in place of `value`, `N` cells of one row at `at`.
    fn vector<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        at: &VectorAt,
        value: Vector<E, N>,
    ) -> Vector<E, N>;
}

/// [`VectorStage`]'s host half: how what the launch is given becomes the argument.
pub trait VectorStageLaunch: VectorStage {
    type Operand;

    fn arg<E: Numeric>(operand: Self::Operand) -> <Self::Arg<E> as LaunchArg>::RuntimeArg;
}

/// The identity stage.
#[cube]
impl VectorStage for () {
    type Arg<E: Numeric> = ();

    fn vector<E: Numeric, N: Size>(
        _arg: &Self::Arg<E>,
        _at: &VectorAt,
        value: Vector<E, N>,
    ) -> Vector<E, N> {
        value
    }
}

impl VectorStageLaunch for () {
    type Operand = ();

    fn arg<E: Numeric>(_operand: ()) {}
}

/// `A` then `B`.
#[cube]
impl<A: VectorStage, B: VectorStage> VectorStage for (A, B) {
    type Arg<E: Numeric> = (A::Arg<E>, B::Arg<E>);

    fn vector<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        at: &VectorAt,
        value: Vector<E, N>,
    ) -> Vector<E, N> {
        let first = A::vector::<E, N>(&arg.0, at, value);
        B::vector::<E, N>(&arg.1, at, first)
    }
}

impl<A: VectorStageLaunch, B: VectorStageLaunch> VectorStageLaunch for (A, B) {
    type Operand = (A::Operand, B::Operand);

    fn arg<E: Numeric>((first, second): Self::Operand) -> <Self::Arg<E> as LaunchArg>::RuntimeArg {
        (A::arg::<E>(first), B::arg::<E>(second))
    }
}
