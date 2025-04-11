use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{
    ArgMax, ArgMin, Max, MaxAbs, Mean, Min, Prod, ReduceCoordinate, ReduceFamily,
    ReduceInstruction, ReduceRequirements, SharedAccumulator, Sum,
};

#[derive(Debug, CubeType, Clone)]
pub enum ReduceFn {
    Sum(Sum),
    Prod(Prod),
    Mean(Mean),
    MaxAbs(MaxAbs),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Max(Max),
    Min(Min),
}

#[derive_cube_comptime]
pub enum ReduceFnConfig {
    Sum,
    Prod,
    Mean,
    MaxAbs,
    ArgMax,
    ArgMin,
    Max,
    Min,
}

impl ReduceFamily for ReduceFn {
    type Instruction<In: Numeric> = Self;
    type Config = ReduceFnConfig;
}

#[derive(CubeType)]
pub struct DynamicAccumulator<N: Numeric> {
    pub elements: SharedMemory<Line<N>>,
    pub args: CubeOption<SharedMemory<Line<u32>>>,
}

#[derive(CubeType)]
pub struct DynamicAccumulatorItem<N: Numeric> {
    pub elements: Line<N>,
    pub args: CubeOption<Line<u32>>,
}

#[cube]
impl<In: Numeric> SharedAccumulator<In> for DynamicAccumulator<In> {
    type Item = DynamicAccumulatorItem<In>;

    fn allocate(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
        #[comptime] coordinate: bool,
    ) -> Self {
        let elements = SharedMemory::new_lined(length, line_size);
        let args = if comptime![coordinate] {
            let args = SharedMemory::new_lined(length, line_size);
            CubeOption::new_Some(args)
        } else {
            CubeOption::new_None()
        };

        DynamicAccumulator::<In> { elements, args }
    }

    fn read(accumulator: &Self, index: u32) -> Self::Item {
        let elements = accumulator.elements[index];
        let args = match accumulator.args {
            CubeOption::Some(args) => CubeOption::new_Some(args[index]),
            CubeOption::None => CubeOption::new_None(),
        };

        DynamicAccumulatorItem::<In> { elements, args }
    }

    fn write(accumulator: &mut Self, index: u32, item: Self::Item) {
        accumulator.elements[index] = item.elements;

        let args = &mut accumulator.args;
        match args {
            CubeOption::Some(args) => {
                args[index] = item.args.unwrap();
            }
            CubeOption::None => {}
        };
    }
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for ReduceFn {
    type AccumulatorItem = DynamicAccumulatorItem<In>;
    type SharedAccumulator = DynamicAccumulator<In>;
    type Config = ReduceFnConfig;

    fn requirements(this: &Self) -> ReduceRequirements {
        let coordinates = match this {
            ReduceFn::Sum(..) => comptime![false],
            ReduceFn::Prod(..) => comptime![false],
            ReduceFn::Mean(..) => comptime![false],
            ReduceFn::MaxAbs(..) => comptime![false],
            ReduceFn::ArgMax(..) => comptime![true],
            ReduceFn::ArgMin(..) => comptime![true],
            ReduceFn::Max(..) => comptime![false],
            ReduceFn::Min(..) => comptime![false],
        };
        ReduceRequirements {
            coordinates: comptime! {coordinates},
        }
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        match config {
            ReduceFnConfig::Sum => ReduceFn::new_Sum(Sum {}),
            ReduceFnConfig::Prod => ReduceFn::new_Prod(Prod {}),
            ReduceFnConfig::Mean => ReduceFn::new_Mean(Mean { sum: Sum {} }),
            ReduceFnConfig::MaxAbs => ReduceFn::new_MaxAbs(MaxAbs {}),
            ReduceFnConfig::ArgMax => ReduceFn::new_ArgMax(ArgMax {}),
            ReduceFnConfig::ArgMin => ReduceFn::new_ArgMin(ArgMin {}),
            ReduceFnConfig::Max => ReduceFn::new_Max(Max {}),
            ReduceFnConfig::Min => ReduceFn::new_Min(Min {}),
        }
    }

    fn null_input(this: &Self, #[comptime] line_size: u32) -> Line<In> {
        match this {
            ReduceFn::Sum(sum) => Sum::null_input(sum, line_size),
            ReduceFn::Prod(prod) => Prod::null_input(prod, line_size),
            ReduceFn::Mean(mean) => Mean::null_input(mean, line_size),
            ReduceFn::MaxAbs(maxabs) => MaxAbs::null_input(maxabs, line_size),
            ReduceFn::ArgMax(argmax) => ArgMax::null_input(argmax, line_size),
            ReduceFn::ArgMin(argmin) => ArgMin::null_input(argmin, line_size),
            ReduceFn::Max(max) => Max::null_input(max, line_size),
            ReduceFn::Min(min) => Min::null_input(min, line_size),
        }
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        match this {
            ReduceFn::Sum(sum) => {
                let elements = Sum::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Mean(sum) => {
                let elements = Mean::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Prod(sum) => {
                let elements = Prod::null_accumulator(sum, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::MaxAbs(maxabs) => {
                let elements = MaxAbs::null_accumulator(maxabs, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::ArgMax(argmax) => {
                let (elements, args) = ArgMax::null_accumulator(argmax, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::ArgMin(argmin) => {
                let (elements, args) = ArgMin::null_accumulator(argmin, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::Max(max) => {
                let elements = Max::null_accumulator(max, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Min(min) => {
                let elements = Min::null_accumulator(min, line_size);

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
        }
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        destination.elements = source.elements;
        let args = &mut destination.args;
        match args {
            CubeOption::Some(val) => *val = source.args.unwrap(),
            CubeOption::None => {}
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        match this {
            ReduceFn::Sum(sum) => {
                let elements =
                    Sum::reduce(sum, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Prod(sum) => {
                let elements =
                    Prod::reduce(sum, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Mean(sum) => {
                let elements =
                    Mean::reduce(sum, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::MaxAbs(maxabs) => {
                let elements =
                    MaxAbs::reduce(maxabs, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::ArgMax(argmax) => {
                let (elements, args) = ArgMax::reduce(
                    argmax,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    use_planes,
                );

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::ArgMin(argmin) => {
                let (elements, args) = ArgMin::reduce(
                    argmin,
                    &(accumulator.elements, accumulator.args.unwrap()),
                    item,
                    coordinate,
                    use_planes,
                );

                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::Max(max) => {
                let elements =
                    Max::reduce(max, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Min(min) => {
                let elements =
                    Min::reduce(min, &accumulator.elements, item, coordinate, use_planes);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
        }
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        match this {
            ReduceFn::Sum(sum) => {
                let elements = Sum::fuse_accumulators(sum, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Prod(prod) => {
                let elements = Prod::fuse_accumulators(prod, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Mean(mean) => {
                let elements = Mean::fuse_accumulators(mean, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::MaxAbs(maxabs) => {
                let elements = MaxAbs::fuse_accumulators(maxabs, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::ArgMax(argmax) => {
                let (elements, args) = ArgMax::fuse_accumulators(
                    argmax,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::ArgMin(argmin) => {
                let (elements, args) = ArgMin::fuse_accumulators(
                    argmin,
                    (lhs.elements, lhs.args.unwrap()),
                    (rhs.elements, rhs.args.unwrap()),
                );
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_Some(args),
                }
            }
            ReduceFn::Max(max) => {
                let elements = Max::fuse_accumulators(max, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
            ReduceFn::Min(min) => {
                let elements = Min::fuse_accumulators(min, lhs.elements, rhs.elements);
                DynamicAccumulatorItem::<In> {
                    elements,
                    args: CubeOption::new_None(),
                }
            }
        }
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Dynamic can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_line<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Out {
        match this {
            ReduceFn::Sum(sum) => {
                Sum::merge_line::<Out>(sum, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Prod(prod) => {
                Prod::merge_line::<Out>(prod, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Mean(mean) => {
                Mean::merge_line::<Out>(mean, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::MaxAbs(maxabs) => {
                MaxAbs::merge_line::<Out>(maxabs, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::ArgMax(argmax) => ArgMax::merge_line::<Out>(
                argmax,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceFn::ArgMin(argmin) => ArgMin::merge_line::<Out>(
                argmin,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceFn::Max(max) => {
                Max::merge_line::<Out>(max, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Min(min) => {
                Min::merge_line::<Out>(min, accumulator.elements, shape_axis_reduce)
            }
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        match this {
            ReduceFn::Sum(sum) => {
                Sum::to_output_perpendicular::<Out>(sum, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Prod(prod) => {
                Prod::to_output_perpendicular::<Out>(prod, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Mean(mean) => {
                Mean::to_output_perpendicular::<Out>(mean, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::MaxAbs(maxabs) => MaxAbs::to_output_perpendicular::<Out>(
                maxabs,
                accumulator.elements,
                shape_axis_reduce,
            ),
            ReduceFn::ArgMax(args) => ArgMax::to_output_perpendicular::<Out>(
                args,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceFn::ArgMin(args) => ArgMin::to_output_perpendicular::<Out>(
                args,
                (accumulator.elements, accumulator.args.unwrap()),
                shape_axis_reduce,
            ),
            ReduceFn::Max(max) => {
                Max::to_output_perpendicular::<Out>(max, accumulator.elements, shape_axis_reduce)
            }
            ReduceFn::Min(min) => {
                Min::to_output_perpendicular::<Out>(min, accumulator.elements, shape_axis_reduce)
            }
        }
    }
}
