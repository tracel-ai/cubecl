//! The stage a store passes through: it sees each stored line with its place and returns the line
//! to store.

use cubecl::prelude::*;

/// Where a stored line sits in a rank 2 output. Its row and column are worked out only by a stage
/// that asks, so a stage that ignores them costs no division.
#[derive(CubeType, Clone, Copy)]
pub struct LineAt {
    first: usize,
    row_stride: usize,
}

#[cube]
impl LineAt {
    pub(crate) fn new(first: usize, row_stride: usize) -> LineAt {
        LineAt { first, row_stride }
    }

    /// The row of the line's first cell.
    pub fn row(&self) -> usize {
        self.first / self.row_stride
    }

    /// The column of the line's first cell.
    pub fn column(&self) -> usize {
        self.first % self.row_stride
    }
}

/// What happens to each line a [`Merged`](crate::launch::Merged) destination's merge writes out, so a stage
/// sees each cell's final value once. A line never crosses a row, and lines arrive in no particular
/// order and in divergent control flow, so a stage may not use plane operations or synchronize.
#[cube]
pub trait LineStage: Send + Sync + 'static {
    /// The launch argument the stage reads and writes through, at element `E`.
    type Arg<E: Numeric>: LaunchArg + CubeType;

    /// The line to store in place of `value`, a line of `N` cells at `at`.
    fn line<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        at: &LineAt,
        value: Vector<E, N>,
    ) -> Vector<E, N>;
}

/// [`LineStage`]'s host half: how what the launch is given becomes the argument.
pub trait LineStageLaunch: LineStage {
    type Operand;

    fn arg<E: Numeric>(operand: Self::Operand) -> <Self::Arg<E> as LaunchArg>::RuntimeArg;
}

/// The identity stage.
#[cube]
impl LineStage for () {
    type Arg<E: Numeric> = ();

    fn line<E: Numeric, N: Size>(
        _arg: &Self::Arg<E>,
        _at: &LineAt,
        value: Vector<E, N>,
    ) -> Vector<E, N> {
        value
    }
}

impl LineStageLaunch for () {
    type Operand = ();

    fn arg<E: Numeric>(_operand: ()) {}
}

/// `A` then `B`.
#[cube]
impl<A: LineStage, B: LineStage> LineStage for (A, B) {
    type Arg<E: Numeric> = (A::Arg<E>, B::Arg<E>);

    fn line<E: Numeric, N: Size>(
        arg: &Self::Arg<E>,
        at: &LineAt,
        value: Vector<E, N>,
    ) -> Vector<E, N> {
        let first = A::line::<E, N>(&arg.0, at, value);
        B::line::<E, N>(&arg.1, at, first)
    }
}

impl<A: LineStageLaunch, B: LineStageLaunch> LineStageLaunch for (A, B) {
    type Operand = (A::Operand, B::Operand);

    fn arg<E: Numeric>((first, second): Self::Operand) -> <Self::Arg<E> as LaunchArg>::RuntimeArg {
        (A::arg::<E>(first), B::arg::<E>(second))
    }
}
