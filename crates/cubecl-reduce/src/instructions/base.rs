use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Reduce<In: Numeric>: Send + Sync + 'static {
    type AccumulatorItem: CubeType;
    type SharedAccumulator: SharedAccumulator<In, Item = Self::AccumulatorItem>;

    fn null_input(#[comptime] line_size: u32) -> Line<In>;

    fn null_accumulator(#[comptime] line_size: u32) -> Self::AccumulatorItem;

    fn update_accumulator(destination: &mut Self::AccumulatorItem, source: &Self::AccumulatorItem);

    fn reduce(
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem;

    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem;

    fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out;

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}

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
pub fn reduce_inplace<In: Numeric, R: Reduce<In>>(
    accumulator: &mut R::AccumulatorItem,
    item: Line<In>,
    coordinate: Line<u32>,
    #[comptime] use_planes: bool,
) {
    let reduction = &R::reduce(accumulator, item, coordinate, use_planes);
    R::update_accumulator(accumulator, reduction);
}

#[cube]
pub fn reduce_shared_inplace<In: Numeric, R: Reduce<In>>(
    accumulator: &mut R::SharedAccumulator,
    index: u32,
    item: Line<In>,
    coordinate: Line<u32>,
    #[comptime] use_planes: bool,
) {
    let reduction = R::reduce(
        &R::SharedAccumulator::read(accumulator, index),
        item,
        coordinate,
        use_planes,
    );
    R::SharedAccumulator::write(accumulator, index, reduction);
}

#[cube]
pub fn fuse_accumulator_inplace<In: Numeric, R: Reduce<In>>(
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
