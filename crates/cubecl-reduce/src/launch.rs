use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::instructions::*;
use crate::primitives::*;
use crate::{LineMode, ReduceConfig, ReduceStrategy};

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
pub(crate) fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: ReduceInstruction<In>>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    let settings = ReduceParams {
        shared: strategy.shared.then(|| {
            if strategy.use_planes {
                config.cube_dim.y
            } else {
                config.cube_dim.num_elems()
            }
        }),
        use_planes: strategy.use_planes,
        line_size: config.line_size,
        line_mode: config.line_mode,
        bound_checks: config.bound_checks,
    };
    unsafe {
        reduce_kernel::launch_unchecked::<In, Out, Rd, Run>(
            client,
            config.cube_count,
            config.cube_dim,
            input.as_tensor_arg(config.line_size as u8),
            output.as_tensor_arg(1),
            ScalarArg::new(axis),
            settings,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReduceParams {
    shared: Option<u32>, // shared if Some(x) where x is the accumulator size.
    use_planes: bool,
    line_size: u32,
    line_mode: LineMode,
    bound_checks: bool,
}

mod reduce_kernel {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
        context: &mut cubecl::prelude::CubeContext,
        input: <Tensor<Line<In>> as cubecl::prelude::CubeType>::ExpandType,
        output: <Tensor<Out> as cubecl::prelude::CubeType>::ExpandType,
        axis_reduce: <u32 as cubecl::prelude::CubeType>::ExpandType,
        params: ReduceParams,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            let reduce_index = {
                let _arg_0 = params.clone();
                cubecl::frontend::debug_call_expand(context, "get_reduce_index", |context| {
                    get_reduce_index::expand(context, _arg_0.into())
                })
            };
            {
                let _cond = {
                    let _lhs = params.clone().bound_checks.clone();
                    let _rhs = {
                        let _lhs = reduce_index.clone();
                        let _rhs = {
                            let _arg_0 = {
                                cubecl::frontend::debug_call_expand(
                                    context,
                                    "output.clone().len",
                                    |context| output.clone().__expand_len_method(context),
                                )
                            };
                            let _arg_1 = params.clone();
                            cubecl::frontend::debug_call_expand(
                                context,
                                "get_reduce_count",
                                |context| {
                                    get_reduce_count::expand(context, _arg_0.into(), _arg_1.into())
                                },
                            )
                        };
                        cubecl::frontend::ge::expand(context, _lhs, _rhs)
                    };
                    cubecl::frontend::and::expand(context, _lhs, _rhs)
                };
                cubecl::frontend::branch::if_expand(context, _cond.into(), |context| {
                    cubecl::frontend::branch::return_expand(context);
                    ()
                });
            };
            let range = {
                let _arg_0 = reduce_index.clone();
                let _arg_1 = input.clone();
                let _arg_2 = output.clone();
                let _arg_3 = axis_reduce.clone();
                let _arg_4 = params.clone().line_size;
                let _arg_5 = params.clone().line_mode;
                cubecl::frontend::debug_call_expand(
                    context,
                    "ReduceRange::new :: < In, Out >",
                    |context| {
                        ReduceRange::__expand_new::<In, Out>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                            _arg_3.into(),
                            _arg_4.into(),
                            _arg_5.into(),
                        )
                    },
                )
            };
            let accumulator = match comptime!((params.shared, params.use_planes)) {
                (Some(accumulator_size), use_planes) => {
                    let mut accumulator = {
                        let _init = {
                            let _arg_0 = {
                                cubecl::frontend::debug_call_expand(
                                    context,
                                    "input.clone().to_slice",
                                    |context| input.clone().__expand_to_slice_method(context),
                                )
                            };
                            let _arg_1 = range.clone();
                            let _arg_2 = accumulator_size;
                            let _arg_3 = params.clone().line_size;
                            let _arg_4 = params.clone().line_mode;
                            let _arg_5 = use_planes;
                            cubecl::frontend::debug_call_expand(
                                context,
                                "reduce_slice_shared",
                                |context| {
                                    reduce_slice_shared::expand::<In, R>(
                                        context,
                                        _arg_0.into(),
                                        _arg_1.into(),
                                        _arg_2.into(),
                                        _arg_3.into(),
                                        _arg_4.into(),
                                        _arg_5.into(),
                                    )
                                },
                            )
                        };
                        cubecl::frontend::Init::init(_init, context)
                    };
                    {
                        cubecl::frontend::debug_call_expand(context, "sync_units", |context| {
                            sync_units::expand(context)
                        })
                    };
                    {
                        let _arg_0 = accumulator;
                        let _arg_1 = accumulator_size;
                        cubecl::frontend::debug_call_expand(context, "reduce_tree", |context| {
                            reduce_tree::expand::<In, R>(context, _arg_0.into(), _arg_1.into())
                        })
                    }
                }
                (None, true) => {
                    let _arg_0 = {
                        cubecl::frontend::debug_call_expand(
                            context,
                            "input.clone().to_slice",
                            |context| input.clone().__expand_to_slice_method(context),
                        )
                    };
                    let _arg_1 = range.clone();
                    let _arg_2 = params.clone().line_size;
                    let _arg_3 = params.clone().line_mode;
                    cubecl::frontend::debug_call_expand(context, "reduce_slice_plane", |context| {
                        reduce_slice_plane::expand::<In, R>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                            _arg_3.into(),
                        )
                    })
                }
                (None, false) => {
                    let _arg_0 = {
                        cubecl::frontend::debug_call_expand(
                            context,
                            "input.clone().to_slice",
                            |context| input.clone().__expand_to_slice_method(context),
                        )
                    };
                    let _arg_1 = range;
                    let _arg_2 = params.clone().line_size;
                    let _arg_3 = params.clone().line_mode;
                    cubecl::frontend::debug_call_expand(context, "reduce_slice", |context| {
                        reduce_slice::expand::<In, R>(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                            _arg_2.into(),
                            _arg_3.into(),
                        )
                    })
                }
            };
            {
                let _cond = {
                    let _arg_0 = params.clone();
                    cubecl::frontend::debug_call_expand(context, "elected_writer", |context| {
                        elected_writer::expand(context, _arg_0.into())
                    })
                };
                cubecl::frontend::branch::if_expand(context, _cond.into(), |context| {
                    {
                        let _arg_0 = output;
                        let _arg_1 = accumulator;
                        let _arg_2 = reduce_index;
                        let _arg_3 = {
                            let _arg_0 = axis_reduce;
                            cubecl::frontend::debug_call_expand(context, "input.shape", |context| {
                                input.__expand_shape_method(context, _arg_0.into())
                            })
                        };
                        let _arg_4 = params.clone();
                        cubecl::frontend::debug_call_expand(context, "write_to_output", |context| {
                            write_to_output::expand::<In, Out, R>(
                                context,
                                _arg_0.into(),
                                _arg_1.into(),
                                _arg_2.into(),
                                _arg_3.into(),
                                _arg_4.into(),
                            )
                        })
                    };
                    ()
                });
            };
            ()
        }
    }
    ///reduce_kernel Kernel
    pub struct ReduceKernel<
        In: Numeric,
        Out: Numeric,
        R: ReduceInstruction<In>,
        __R: cubecl::prelude::Runtime,
    > {
        settings: cubecl::prelude::KernelSettings,
        input: <Tensor<Line<In>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        axis_reduce: <u32 as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        output: <Tensor<Out> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        params: ReduceParams,
        __ty: ::core::marker::PhantomData<(Out, In, __R, R)>,
    }
    #[allow(clippy::too_many_arguments)]
    impl<In: Numeric, Out: Numeric, R: ReduceInstruction<In>, __R: cubecl::prelude::Runtime>
        ReduceKernel<In, Out, R, __R>
    {
        pub fn new(
            settings: cubecl::prelude::KernelSettings,
            input: <Tensor<Line<In>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
            axis_reduce: <u32 as cubecl::prelude::LaunchArgExpand>::CompilationArg,
            output: <Tensor<Out> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
            params: ReduceParams,
        ) -> Self {
            Self {
                settings: settings.kernel_name({
                    let shorten = |p: &'static str| {
                        if let Some((_, last)) = p.rsplit_once("::") {
                            last
                        } else {
                            p
                        }
                    };
                    let mut name = format!("{}", "reduce_kernel");
                    {
                        let type_name = shorten(core::any::type_name::<In>());
                        name.push_str(&format!("_{type_name}"));
                    }
                    {
                        let type_name = shorten(core::any::type_name::<Out>());
                        name.push_str(&format!("_{type_name}"));
                    }
                    name
                }),
                input,
                axis_reduce,
                output,
                params,
                __ty: ::core::marker::PhantomData,
            }
        }
        fn to_fast<InF, OutF, RF>(&self) -> ReduceKernel<InF, OutF, RF, __R>
        where
            InF: Numeric,
            OutF: Numeric,
            RF: ReduceInstruction<InF>,
        {
            ReduceKernel {
                settings: self.settings.clone(),
                input: self.input.clone(),
                axis_reduce: self.axis_reduce.clone(),
                output: self.output.clone(),
                params: self.params.clone(),
                __ty: std::marker::PhantomData,
            }
        }
    }
    impl<R: ReduceInstruction<NumericExpand<0>>, __R: cubecl::prelude::Runtime>
        ReduceKernel<NumericExpand<0>, NumericExpand<1>, R, __R>
    {
        fn define_fast(&self, mut builder: KernelBuilder) -> cubecl::prelude::KernelDefinition {
            let input =
                <Tensor<Line<NumericExpand<0u8>>> as cubecl::prelude::LaunchArgExpand>::expand(
                    &self.input,
                    &mut builder,
                );
            let axis_reduce =
                <u32 as cubecl::prelude::LaunchArgExpand>::expand(&self.axis_reduce, &mut builder);
            let output =
                <Tensor<NumericExpand<1u8>> as cubecl::prelude::LaunchArgExpand>::expand_output(
                    &self.output,
                    &mut builder,
                );
            expand::<NumericExpand<0u8>, NumericExpand<1u8>, R>(
                &mut builder.context,
                input.clone(),
                output.clone(),
                axis_reduce.clone(),
                self.params.clone(),
            );
            builder.build(self.settings.clone())
        }
    }

    impl<In: Numeric, Out: Numeric, R: ReduceInstruction<In>, __R: cubecl::prelude::Runtime>
        cubecl::Kernel for ReduceKernel<In, Out, R, __R>
    {
        fn define(&self) -> cubecl::prelude::KernelDefinition {
            let mut builder = cubecl::prelude::KernelBuilder::with_local_allocator(
                <<__R as cubecl::prelude::Runtime>::Compiler as cubecl::Compiler>::local_allocator(
                ),
            );
            builder
                .context
                .register_type::<NumericExpand<0u8>>(In::as_elem_native_unchecked());
            builder
                .context
                .register_type::<NumericExpand<1u8>>(Out::as_elem_native_unchecked());
            let aa = self.to_fast::<NumericExpand<0>, NumericExpand<1>, R>();
            // aa.define_fast(builder)
        }
        fn id(&self) -> cubecl::KernelId {
            let cube_dim = self.settings.cube_dim.clone();
            cubecl::KernelId::new::<Self>().info((
                cube_dim,
                self.params.clone(),
                self.input.clone(),
                self.axis_reduce.clone(),
                self.output.clone(),
            ))
        }
    }
    #[allow(clippy::too_many_arguments)]
    ///Launch the kernel [reduce_kernel()] on the given runtime
    pub unsafe fn launch_unchecked<
        'kernel,
        In: Numeric,
        Out: Numeric,
        R: ReduceInstruction<In>,
        __R: cubecl::prelude::Runtime,
    >(
        __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
        __cube_count: cubecl::prelude::CubeCount,
        __cube_dim: cubecl::prelude::CubeDim,
        input: cubecl::RuntimeArg<'kernel, Tensor<Line<In>>, __R>,
        output: cubecl::RuntimeArg<'kernel, Tensor<Out>, __R>,
        axis_reduce: cubecl::RuntimeArg<'kernel, u32, __R>,
        params: ReduceParams,
    ) -> () {
        use cubecl::frontend::ArgSettings as _;
        let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
        let input_arg_0 =
            <Tensor<Line<In>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&input);
        let input_arg_1 = <u32 as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&axis_reduce);
        let output_arg_0 =
            <Tensor<Out> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&output);
        let __kernel = ReduceKernel::<In, Out, R, __R>::new(
            __settings,
            input_arg_0,
            input_arg_1,
            output_arg_0,
            params,
        );
        let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
        input.register(&mut launcher);
        axis_reduce.register(&mut launcher);
        output.register(&mut launcher);
        launcher.launch_unchecked(__cube_count, __kernel, __client);
    }
}

// #[cube(launch_unchecked, debug)]
// fn reduce_kernel<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
//     input: &Tensor<Line<In>>,
//     output: &mut Tensor<Out>,
//     axis_reduce: u32,
//     #[comptime] params: ReduceParams,
// ) {
//     let reduce_index = get_reduce_index(params);
//
//     if params.bound_checks && reduce_index >= get_reduce_count(output.len(), params) {
//         return;
//     }
//
//     let range = ReduceRange::new::<In, Out>(
//         reduce_index,
//         input,
//         output,
//         axis_reduce,
//         params.line_size,
//         params.line_mode,
//     );
//
//     let accumulator = match comptime!((params.shared, params.use_planes)) {
//         (Some(accumulator_size), use_planes) => {
//             let mut accumulator = reduce_slice_shared::<In, R>(
//                 input.to_slice(),
//                 range,
//                 accumulator_size,
//                 params.line_size,
//                 params.line_mode,
//                 use_planes,
//             );
//             sync_units();
//             reduce_tree::<In, R>(&mut accumulator, accumulator_size)
//         }
//         (None, true) => {
//             reduce_slice_plane::<In, R>(input.to_slice(), range, params.line_size, params.line_mode)
//         }
//         (None, false) => {
//             reduce_slice::<In, R>(input.to_slice(), range, params.line_size, params.line_mode)
//         }
//     };
//
//     if elected_writer(params) {
//         write_to_output::<In, Out, R>(
//             output,
//             accumulator,
//             reduce_index,
//             input.shape(axis_reduce),
//             params,
//         );
//     }
// }

#[cube]
fn get_reduce_index(#[comptime] params: ReduceParams) -> u32 {
    if params.shared.is_some() {
        CUBE_POS
    } else if params.use_planes {
        CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y
    } else {
        ABSOLUTE_POS
    }
}

#[cube]
fn get_reduce_count(output_size: u32, #[comptime] params: ReduceParams) -> u32 {
    match comptime!(params.line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / params.line_size,
    }
}

#[cube]
fn elected_writer(#[comptime] settings: ReduceParams) -> bool {
    if settings.shared.is_some() {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true
    }
}

#[cube]
fn write_to_output<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
    output: &mut Tensor<Out>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            output[reduce_index] = R::merge_line::<Out>(accumulator, shape_axis_reduce)
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(accumulator, shape_axis_reduce);

            #[unroll]
            for k in 0..settings.line_size {
                output[settings.line_size * reduce_index + k] = out[k];
            }
        }
    }
}
