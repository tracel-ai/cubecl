use crate::{self as cubecl, as_bytes};
use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_line_index<F: Float>(output: &mut Array<F>, #[comptime] line_size: u32) {
    if UNIT_POS == 0 {
        let line = Line::empty(line_size).fill(F::new(5.0));
        output[0] = line[0];
    }
}

pub fn test_line_index<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native().unwrap()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_index::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, line_size as usize, 1),
                line_size as u32,
            );
        }
        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); line_size as usize];
        expected[0] = F::new(5.0);

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_index_assign<F: Float>(output: &mut Array<Line<F>>) {
    if UNIT_POS == 0 {
        let mut line = RuntimeCell::<Line<F>>::new(output[0]);
        line.store_at(0, F::new(5.0));
        output[0] = line.consume();
    }
}

pub fn test_line_index_assign<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native().unwrap()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_index_assign::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, 1, line_size),
            );
        }

        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let mut expected = vec![F::new(0.0); line_size as usize];
        expected[0] = F::new(5.0);

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

#[cube(launch_unchecked)]
pub fn kernel_line_loop_unroll<F: Float>(output: &mut Array<Line<F>>, #[comptime] line_size: u32) {
    if UNIT_POS == 0 {
        let mut line = output[0];
        #[unroll]
        for k in 0..line_size {
            line[k] += F::cast_from(k);
        }
        output[0] = line;
    }
}

pub mod kernel_line_loop_unrollaa {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<F: Float>(
        scope: &mut cubecl::prelude::Scope,
        output: <Array<Line<F>> as cubecl::prelude::CubeType>::ExpandType,
        line_size: u32,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _cond = {
                    let _lhs = UNIT_POS::expand(scope);
                    let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(scope, 0);
                    cubecl::frontend::eq::expand(scope, _lhs.into(), _rhs.into())
                };
                cubecl::frontend::branch::if_expand(scope, _cond.into(), |scope| {
                    let mut line = {
                        let __init = {
                            let _init = {
                                let _array = output.clone();
                                let _index =
                                    cubecl::frontend::ExpandElementTyped::from_lit(scope, 0);
                                cubecl::frontend::index::expand(scope, _array, _index.into())
                            };
                            cubecl::frontend::Init::init(_init, scope)
                        };
                        __init
                    };
                    {
                        let _range = {
                            let _start = 0;
                            let _end = line_size.clone();
                            cubecl::frontend::RangeExpand::new(_start.into(), _end.into(), false)
                        };
                        let _unroll = true;
                        cubecl::frontend::branch::for_expand(scope, _range, _unroll, |scope, k| {
                            {
                                let _array = line.clone();
                                let _index = k.clone();
                                let _value = {
                                    let _arg_0 = k;
                                    F::__expand_cast_from(scope, _arg_0.into())
                                };
                                cubecl::frontend::add_assign_array_op::expand(
                                    scope,
                                    _array,
                                    _index.into(),
                                    _value.into(),
                                )
                            };
                            ()
                        });
                    };
                    {
                        let _array = output;
                        let _index = cubecl::frontend::ExpandElementTyped::from_lit(scope, 0);
                        let _value = line;
                        cubecl::frontend::index_assign::expand(
                            scope,
                            _array,
                            _index.into(),
                            _value.into(),
                        )
                    };
                    ()
                });
            };
            ()
        }
    }
    ///kernel_line_loop_unroll Kernel
    pub struct KernelLineLoopUnroll<F: Float, __R: cubecl::prelude::Runtime> {
        settings: cubecl::prelude::KernelSettings,
        output: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
        line_size: u32,
        __ty: ::core::marker::PhantomData<(__R, F)>,
    }
    #[allow(clippy::too_many_arguments)]
    impl<F: Float, __R: cubecl::prelude::Runtime> KernelLineLoopUnroll<F, __R> {
        pub fn new(
            settings: cubecl::prelude::KernelSettings,
            output: <Array<Line<F>> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
            line_size: u32,
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
                    let mut name = format!("{}", "kernel_line_loop_unroll");
                    {
                        let type_name = shorten(core::any::type_name::<F>());
                        name.push_str(&format!("_{type_name}"));
                    }
                    name
                }),
                output,
                line_size,
                __ty: ::core::marker::PhantomData,
            }
        }
    }
    impl<F: Float, __R: cubecl::prelude::Runtime> cubecl::Kernel for KernelLineLoopUnroll<F, __R> {
        fn define(&self) -> cubecl::prelude::KernelDefinition {
            let mut builder = cubecl::prelude::KernelBuilder::default();
            builder
                .scope
                .register_elem::<FloatExpand<0u8>>(F::as_elem_native_unchecked());
            let output =
                <Array<Line<FloatExpand<0u8>>> as cubecl::prelude::LaunchArgExpand>::expand_output(
                    &self.output.dynamic_cast(),
                    &mut builder,
                );
            expand::<FloatExpand<0u8>>(&mut builder.scope, output.clone(), self.line_size.clone());
            builder.build(self.settings.clone())
        }
        fn id(&self) -> cubecl::KernelId {
            let cube_dim = self.settings.cube_dim.clone();
            cubecl::KernelId::new::<Self>().info((
                cube_dim,
                self.line_size.clone(),
                self.output.clone(),
            ))
        }
    }
    #[allow(clippy::too_many_arguments)]
    ///Launch the kernel [kernel_line_loop_unroll()] on the given runtime
    pub unsafe fn launch_unchecked<'kernel, F: Float, __R: cubecl::prelude::Runtime>(
        __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
        __cube_count: cubecl::prelude::CubeCount,
        __cube_dim: cubecl::prelude::CubeDim,
        output: cubecl::RuntimeArg<'kernel, Array<Line<F>>, __R>,
        line_size: u32,
    ) -> () {
        use cubecl::frontend::ArgSettings as _;
        let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
        let output_arg_0 =
            <Array<Line<F>> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&output);
        let __kernel = KernelLineLoopUnroll::<F, __R>::new(__settings, output_arg_0, line_size);
        let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
        output.register(&mut launcher);
        launcher.launch_unchecked(__cube_count, __kernel, __client);
    }
}

pub fn test_line_loop_unroll<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R::Server, R::Channel>,
) {
    for line_size in R::line_size_elem(&F::as_elem_native_unchecked()) {
        let handle = client.create(F::as_bytes(&vec![F::new(0.0); line_size as usize]));
        unsafe {
            kernel_line_loop_unroll::launch_unchecked::<F, R>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                ArrayArg::from_raw_parts::<F>(&handle, 1, line_size),
                line_size as u32,
            );
        }

        let actual = client.read_one(handle.binding());
        let actual = F::from_bytes(&actual);

        let expected = (0..line_size as i64)
            .map(|x| F::from_int(x))
            .collect::<Vec<_>>();

        assert_eq!(&actual[..line_size as usize], expected);
    }
}

macro_rules! impl_line_comparison {
    ($cmp:ident, $expected:expr) => {
        ::paste::paste! {
            #[cube(launch)]
            pub fn [< kernel_line_ $cmp >]<F: Float>(
                lhs: &Array<Line<F>>,
                rhs: &Array<Line<F>>,
                output: &mut Array<Line<u32>>,
            ) {
                if UNIT_POS == 0 {
                    output[0] = Line::cast_from(lhs[0].$cmp(rhs[0]));
                }
            }

            pub fn [< test_line_ $cmp >] <R: Runtime, F: Float + CubeElement>(
                client: ComputeClient<R::Server, R::Channel>,
            ) {
                let lhs = client.create(as_bytes![F: 0.0, 1.0, 2.0, 3.0]);
                let rhs = client.create(as_bytes![F: 0.0, 2.0, 1.0, 3.0]);
                let output = client.empty(16);

                unsafe {
                    [< kernel_line_ $cmp >]::launch::<F, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(1, 1, 1),
                        ArrayArg::from_raw_parts::<F>(&lhs, 1, 4),
                        ArrayArg::from_raw_parts::<F>(&rhs, 1, 4),
                        ArrayArg::from_raw_parts::<u32>(&output, 1, 4),
                    )
                };

                let actual = client.read_one(output.binding());
                let actual = u32::from_bytes(&actual);

                assert_eq!(actual, $expected);
            }
        }
    };
}

impl_line_comparison!(equal, [1, 0, 0, 1]);
impl_line_comparison!(not_equal, [0, 1, 1, 0]);
impl_line_comparison!(less_than, [0, 1, 0, 0]);
impl_line_comparison!(greater_than, [0, 0, 1, 0]);
impl_line_comparison!(less_equal, [1, 1, 0, 1]);
impl_line_comparison!(greater_equal, [1, 0, 1, 1]);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_line {
    () => {
        use super::*;

        #[test]
        fn test_line_index() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_index::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_index_assign() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_index_assign::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_loop_unroll() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_loop_unroll::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_equal::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_not_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_not_equal::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_less_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_less_than::<TestRuntime, FloatType>(client);
        }

        #[test]
        fn test_line_greater_than() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_greater_than::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_less_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_less_equal::<TestRuntime, FloatType>(
                client,
            );
        }

        #[test]
        fn test_line_greater_equal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::line::test_line_greater_equal::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
