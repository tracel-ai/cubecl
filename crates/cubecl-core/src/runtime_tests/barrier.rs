use crate::{self as cubecl, Feature, as_bytes, prelude::barrier::BarrierLevel};
use barrier::Barrier;
use cubecl::prelude::*;

pub mod async_copy_test_ {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<F: Float>(
        context: &mut cubecl::prelude::Scope,
        input: <Array<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
        output: <Array<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            let barrier = {
                let __init = {
                    let _arg_0 = { BarrierLevel::__expand_unit(context) };
                    Barrier::<F>::__expand_new(context, _arg_0.into())
                };
                __init
            };
            let mut smem = {
                let __init = {
                    let _init = {
                        let _arg_0 = 1u32;
                        let _arg_1 = 1u32;
                        SharedMemory::<F>::__expand_new_lined(context, _arg_0.into(), _arg_1.into())
                    };
                    cubecl::frontend::Init::init(_init, context)
                };
                __init
            };
            let source = {
                let __init = {
                    let _arg_0 = 2;
                    let _arg_1 = 3;
                    input.__expand_slice_method(context, _arg_0.into(), _arg_1.into())
                };
                __init
            };
            let mut destination = {
                let __init = {
                    let _init = {
                        let _arg_0 = 0;
                        let _arg_1 = 1;
                        smem.clone().__expand_slice_mut_method(
                            context,
                            _arg_0.into(),
                            _arg_1.into(),
                        )
                    };
                    cubecl::frontend::Init::init(_init, context)
                };
                __init
            };
            {
                let _arg_0 = source;
                let _arg_1 = destination;
                barrier
                    .clone()
                    .__expand_memcpy_async_method(context, _arg_0.into(), _arg_1.into())
            };
            {
                barrier.__expand_arrive_and_wait_method(context)
            };
            {
                let _array = output;
                let _index = cubecl::frontend::ExpandElementTyped::from_lit(context, 0);
                let _value = {
                    let _array = smem;
                    let _index = cubecl::frontend::ExpandElementTyped::from_lit(context, 0);
                    cubecl::frontend::index::expand(context, _array, _index.into())
                };
                cubecl::frontend::index_assign::expand(
                    context,
                    _array,
                    _index.into(),
                    _value.into(),
                )
            };
            ()
        }
    }
    ///async_copy_test Kernel
    pub struct AsyncCopyTest<F: Float, __R: cubecl::prelude::Runtime> {
        settings: cubecl::prelude::KernelSettings,
        input: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        output: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        __ty: ::core::marker::PhantomData<(__R, F)>,
    }
    #[allow(clippy::too_many_arguments)]
    impl<F: Float, __R: cubecl::prelude::Runtime> AsyncCopyTest<F, __R> {
        pub fn new(
            settings: cubecl::prelude::KernelSettings,
            input: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
            output: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
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
                    let mut name = "async_copy_test".to_string();
                    {
                        let type_name = shorten(core::any::type_name::<F>());
                        name.push_str(&format!("_{type_name}"));
                    }
                    name
                }),
                input,
                output,
                __ty: ::core::marker::PhantomData,
            }
        }
    }
    impl<F: Float, __R: cubecl::prelude::Runtime> cubecl::Kernel for AsyncCopyTest<F, __R> {
        fn define(&self) -> cubecl::prelude::KernelDefinition {
            let mut builder = cubecl::prelude::KernelBuilder::default();
            builder
                .scope
                .register_elem::<FloatExpand<0u8>>(F::as_elem_native_unchecked());
            let input = <Array<Line<FloatExpand<0u8>>> as cubecl::prelude::LaunchArgExpand>::expand(
                &self.input.dynamic_cast(),
                &mut builder,
            );
            let output =
                <Array<Line<FloatExpand<0u8>>> as cubecl::prelude::LaunchArgExpand>::expand_output(
                    &self.output.dynamic_cast(),
                    &mut builder,
                );
            expand::<FloatExpand<0u8>>(&mut builder.scope, input.clone(), output.clone());
            builder.build(self.settings.clone())
        }
        fn id(&self) -> cubecl::KernelId {
            let cube_dim = self.settings.cube_dim;
            cubecl::KernelId::new::<Self>().info((
                cube_dim,
                self.input.clone(),
                self.output.clone(),
            ))
        }
    }
    #[allow(clippy::too_many_arguments)]
    ///Launch the kernel [async_copy_test()] on the given runtime
    pub fn launch<'kernel, F: Float, __R: cubecl::prelude::Runtime>(
        __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
        __cube_count: cubecl::prelude::CubeCount,
        __cube_dim: cubecl::prelude::CubeDim,
        input: cubecl::RuntimeArg<'kernel, Array<Line<F>>, __R>,
        output: cubecl::RuntimeArg<'kernel, Array<Line<F>>, __R>,
    ) {
        use cubecl::frontend::ArgSettings as _;
        let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
        let input_arg_0 =
            <Array<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&input);
        let output_arg_0 =
            <Array<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&output);
        let __kernel = AsyncCopyTest::<F, __R>::new(__settings, input_arg_0, output_arg_0);
        let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
        input.register(&mut launcher);
        output.register(&mut launcher);
        launcher.launch(__cube_count, __kernel, __client);
    }
}

#[cube(launch)]
pub fn async_copy_test<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>) {
    let barrier = Barrier::<F>::new(BarrierLevel::unit());
    let mut smem = SharedMemory::<F>::new_lined(1u32, 1u32);

    let source = input.slice(2, 3);
    let mut destination = smem.slice_mut(0, 1);

    barrier.memcpy_async(&source, &mut destination);

    barrier.arrive_and_wait();
    output[0] = smem[0];
}

pub fn test_async_copy<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let input = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        async_copy_test::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<F>(&input, 5, 1),
            ArrayArg::from_raw_parts::<F>(&output, 1, 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

#[cube(launch)]
fn one_load<F: Float>(lhs: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(4u32, 1u32);

    let barrier = Barrier::<F>::new(BarrierLevel::cube_manual(0u32));
    sync_units();

    // Can't use lhs.to_slice() because then generated input_length will not exist
    barrier.memcpy_async(&lhs.slice(0u32, 4u32), &mut lhs_smem.to_slice_mut());

    barrier.arrive_and_wait();

    let start = UNIT_POS_X * 2u32;
    let end = start + 2u32;
    for i in start..end {
        output[i] = lhs_smem[i];
    }
}

#[cube(launch)]
fn two_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32, // should be even
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let barrier = Barrier::<F>::new(BarrierLevel::cube_manual(0u32));
    sync_units();

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    barrier.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    barrier.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));

    barrier.arrive_and_wait();
    let mut dot = Line::cast_from(0u32);
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X] = dot;
}

#[cube(launch)]
fn two_independent_loads<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    #[comptime] num_data: u32,
) {
    let mut lhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);
    let mut rhs_smem = SharedMemory::<F>::new_lined(num_data, 1u32);

    let barrier_0 = barrier::Barrier::new(BarrierLevel::cube_manual(0u32));
    let barrier_1 = barrier::Barrier::new(BarrierLevel::cube_manual(0u32));
    // At the Cube level, we must sync after barrier creation to make sure they
    // exist for all units
    sync_units();

    let start = UNIT_POS_X * num_data / 2;
    let end = start + num_data / 2;

    for i in start..end {
        lhs_smem[i] = Line::cast_from(0u32);
        rhs_smem[i] = Line::cast_from(0u32);
        output[i] = Line::cast_from(0u32);
    }

    barrier_0.memcpy_async(&lhs.slice(start, end), &mut lhs_smem.slice_mut(start, end));
    barrier_1.memcpy_async(&rhs.slice(start, end), &mut rhs_smem.slice_mut(start, end));

    let mut dot = Line::cast_from(0u32);

    barrier_0.arrive_and_wait();
    barrier_1.arrive_and_wait();
    for i in start..end {
        dot += lhs_smem[i] * rhs_smem[i];
    }

    output[UNIT_POS_X] = dot;
}

pub fn test_memcpy_one_load<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let lhs = client.create(as_bytes![F: 10., 11., 12., 13.]);
    let output = client.empty(4 * core::mem::size_of::<F>());

    unsafe {
        one_load::launch::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            TensorArg::from_raw_parts::<F>(&lhs, &[4, 1], &[4, 4], 1),
            TensorArg::from_raw_parts::<F>(&output, &[4, 1], &[4, 4], 1),
        )
    };

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);
    let expected = [F::new(10.0), F::new(11.0), F::new(12.0), F::new(13.0)];

    assert_eq!(actual, expected);
}

pub fn test_memcpy_two_loads<R: Runtime, F: Float + CubeElement>(
    independent: bool,
    client: ComputeClient<R::Server, R::Channel>,
) {
    if !client.properties().feature_enabled(Feature::Barrier) {
        // We can't execute the test, skip.
        return;
    }

    let num_data = 4;
    let lhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();
    let rhs_data: Vec<F> = (0..num_data).map(|i| F::new(i as f32)).collect();

    let lhs = client.create(F::as_bytes(&lhs_data));
    let rhs = client.create(F::as_bytes(&rhs_data));
    let output = client.empty(2 * core::mem::size_of::<F>());

    if independent {
        unsafe {
            two_independent_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    } else {
        unsafe {
            two_loads::launch::<F, R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(2, 1, 1),
                TensorArg::from_raw_parts::<F>(&lhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&rhs, &[1], &[num_data], 1),
                TensorArg::from_raw_parts::<F>(&output, &[1], &[2], 1),
                num_data as u32,
            )
        };
    }

    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    let middle = num_data / 2;
    let expected = [
        dot(&lhs_data[..middle], &rhs_data[..middle]),
        dot(&lhs_data[middle..], &rhs_data[middle..]),
    ];

    assert_eq!(actual, expected);
}

fn dot<F: Float>(vec1: &[F], vec2: &[F]) -> F {
    let mut sum = F::from_int(0);
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_barrier {
    () => {
        use super::*;

        #[test]
        fn test_barrier_async_copy() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_async_copy::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_barrier_memcpy_async_one_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_one_load::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_barrier_memcpy_async_two_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                false, client,
            );
        }

        #[test]
        fn test_barrier_memcpy_async_two_independent_loads() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::barrier::test_memcpy_two_loads::<TestRuntime, FloatType>(
                true, client,
            );
        }
    };
}
