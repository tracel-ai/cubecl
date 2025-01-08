use cubecl_core as cubecl;
use cubecl_core::prelude::*;

pub trait Reduce: Send + Sync + 'static + std::fmt::Debug {
    type Instruction<In: Numeric>: ReduceInstruction<In>;
}

/// An instruction for a reduce algorithm that works with [`Line`].
///
/// See a provided implementation, such as [`Sum`] or [`ArgMax`] for an example how to implement
/// this trait for a custom instruction.
///
/// A reduction works at three levels. First, it takes input data of type `In` and reduce them
/// with their coordinate into an `AccumulatorItem`. Then, multiple `AccumulatorItem` are possibly fused
/// together into a single accumulator that is converted to the expected output type.
#[cube]
pub trait ReduceInstruction<In: Numeric>: Send + Sync + 'static + std::fmt::Debug {
    /// The intermediate state into which we accumulate new input elements.
    /// This is most likely a `Line<T>` or a struct or tuple of lines.
    type AccumulatorItem: CubeType;

    /// When multiple agents are collaborating to reduce a single slice,
    /// we need a share accumulator to store multiple `AccumulatorItem`.
    /// This is most likely a `SharedMemory<Line<T>>` or a struct or tuple of lined shared memories.
    type SharedAccumulator: SharedAccumulator<In, Item = Self::AccumulatorItem>;

    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(#[comptime] line_size: u32) -> Line<In>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()` always returns
    /// is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(#[comptime] line_size: u32) -> Self::AccumulatorItem;

    /// Assign the value of `source` into `destination`.
    /// In spirit, this is equivalent to `destination = source;`,
    /// but this syntax is not currently supported by CubeCL.
    fn assign_accumulator(destination: &mut Self::AccumulatorItem, source: &Self::AccumulatorItem);

    /// If `use_planes` is `true`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// Else, reduce the given `item` and `coordinate` into the accumulator.
    fn reduce(
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem;

    /// Reduce two accumulators into a single accumulator.
    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem;

    /// Reduce all elements of the accumulator into a single output element of type `Out`.
    fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out;

    /// Convert each element of the accumulator into the expected output element of type `Out`.
    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}

/// A simple trait that abstract over a single or multiple shared memory.
#[cube]
pub trait SharedAccumulator<In: Numeric>: CubeType + Send + Sync + 'static {
    type Item: CubeType;

    fn allocate(#[comptime] length: u32, #[comptime] line_size: u32) -> Self;

    fn read(accumulator: &Self, index: u32) -> Self::Item;

    fn write(accumulator: &mut Self, index: u32, item: Self::Item);
}

#[cube]
impl<In: Numeric> SharedAccumulator<In> for SharedMemory<Line<In>> {
    type Item = Line<In>;

    fn allocate(#[comptime] length: u32, #[comptime] line_size: u32) -> Self {
        SharedMemory::new_lined(length, line_size)
    }

    fn read(accumulator: &Self, index: u32) -> Self::Item {
        accumulator[index]
    }

    fn write(accumulator: &mut Self, index: u32, item: Self::Item) {
        accumulator[index] = item;
    }
}

/// A pair of shared memory used for [`ArgMax`] and [`ArgMin`].
#[derive(CubeType)]
pub struct ArgAccumulator<N: Numeric> {
    pub elements: SharedMemory<Line<N>>,
    pub args: SharedMemory<Line<u32>>,
}

#[cube]
impl<In: Numeric> SharedAccumulator<In> for ArgAccumulator<In> {
    type Item = (Line<In>, Line<u32>);

    fn allocate(#[comptime] length: u32, #[comptime] line_size: u32) -> Self {
        ArgAccumulator::<In> {
            elements: SharedMemory::new_lined(length, line_size),
            args: SharedMemory::new_lined(length, line_size),
        }
    }

    fn read(accumulator: &Self, index: u32) -> Self::Item {
        (accumulator.elements[index], accumulator.args[index])
    }

    fn write(accumulator: &mut Self, index: u32, item: Self::Item) {
        accumulator.elements[index] = item.0;
        accumulator.args[index] = item.1;
    }
}

#[cube]
pub fn reduce_inplace<In: Numeric, R: ReduceInstruction<In>>(
    accumulator: &mut R::AccumulatorItem,
    item: Line<In>,
    coordinate: Line<u32>,
    #[comptime] use_planes: bool,
) {
    let reduction = &R::reduce(accumulator, item, coordinate, use_planes);
    R::assign_accumulator(accumulator, reduction);
}

#[cube]
pub fn reduce_shared_inplace<In: Numeric, R: ReduceInstruction<In>>(
    accumulator: &mut R::SharedAccumulator,
    index: u32,
    item: Line<In>,
    coordinate: Line<u32>,
    #[comptime] use_planes: bool,
) {
    let acc_item = R::SharedAccumulator::read(accumulator, index);
    let reduction = R::reduce(&acc_item, item, coordinate, use_planes);
    R::SharedAccumulator::write(accumulator, index, reduction);
}

#[cube]
pub fn fuse_accumulator_inplace<In: Numeric, R: ReduceInstruction<In>>(
    accumulator: &mut R::SharedAccumulator,
    destination: u32,
    origin: u32,
) {
    let fused = R::fuse_accumulators(
        R::SharedAccumulator::read(accumulator, destination),
        R::SharedAccumulator::read(accumulator, origin),
    );
    R::SharedAccumulator::write(accumulator, destination, fused);
}
