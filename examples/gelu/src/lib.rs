use cubecl::prelude::*;

// #[cube(launch_unchecked, debug)]
// fn gelu_array<F: Float>(input: &Array<F>, output: &mut Array<F>) {
//     if ABSOLUTE_POS < input.len() {
//         output[ABSOLUTE_POS] = gelu_scalar::<F>(input[ABSOLUTE_POS]);
//     }
// }
mod gelu_array {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<F: Float>(
        context: &mut cubecl::prelude::CubeContext,
        input: <Array<F> as cubecl::prelude::CubeType>::ExpandType,
        output: <Array<F> as cubecl::prelude::CubeType>::ExpandType,
    ) -> <() as cubecl::prelude::CubeType>::ExpandType {
        use cubecl::prelude::IntoRuntime as _;
        {
            {
                let _cond = {
                    let _lhs = ABSOLUTE_POS::expand(context);
                    let _rhs = { input.clone().__expand_len_method(context) };
                    cubecl::frontend::lt::expand(context, _lhs, _rhs)
                };
                cubecl::frontend::branch::if_expand(context, _cond.into(), |context| {
                    {
                        let _array = output;
                        let _index = ABSOLUTE_POS::expand(context);
                        let _value = {
                            let _arg_0 = {
                                let _array = input;
                                let _index = ABSOLUTE_POS::expand(context);
                                cubecl::frontend::index::expand(context, _array, _index)
                            };
                            gelu_scalar::expand::<F>(context, _arg_0.into())
                        };
                        cubecl::frontend::index_assign::expand(context, _array, _index, _value)
                    };
                    ()
                });
            };
            ()
        }
    }
    ///gelu_array Kernel
    pub struct GeluArray<F: Float, __R: cubecl::prelude::Runtime> {
        settings: cubecl::prelude::KernelSettings,
        input: <Array<F> as LaunchArgExpand>::CompilationArg,
        output: <Array<F> as LaunchArgExpand>::CompilationArg,
        __ty: ::core::marker::PhantomData<(__R, F)>,
    }
    impl<F: Float, __R: cubecl::prelude::Runtime> GeluArray<F, __R> {
        pub fn new(
            settings: cubecl::prelude::KernelSettings,
            input: <Array<F> as LaunchArgExpand>::CompilationArg,
            output: <Array<F> as LaunchArgExpand>::CompilationArg,
        ) -> Self {
            Self {
                settings,
                input,
                output,
                __ty: ::core::marker::PhantomData,
            }
        }
    }
    impl<F: Float, __R: cubecl::prelude::Runtime> cubecl::Kernel for GeluArray<F, __R> {
        fn define(&self) -> cubecl::prelude::KernelDefinition {
            let mut builder = cubecl::prelude::KernelBuilder::default();
            let mut inputs: ::std::collections::BTreeMap<
                usize,
                std::sync::Arc<dyn core::any::Any>,
            > = std::collections::BTreeMap::new();
            let mut outputs: ::std::collections::BTreeMap<
                usize,
                std::sync::Arc<dyn core::any::Any>,
            > = std::collections::BTreeMap::new();
            #[allow(unused)]
            let register_input = |builder: &mut cubecl::prelude::KernelBuilder,
                                  settings: &cubecl::prelude::KernelSettings,
                                  position: usize|
             -> ::std::sync::Arc<dyn ::core::any::Any> {
                match position {
                    0usize => ::std::sync::Arc::new(
                        <Array<F> as cubecl::prelude::LaunchArgExpand>::expand(
                            self.input,
                            builder,
                            settings.vectorization_input(0usize),
                        ),
                    ),
                    _ => {
                        panic!("Input {position} is invalid");
                    }
                }
            };
            #[allow(unused)]
            let register_output = |builder: &mut cubecl::prelude::KernelBuilder,
                                   settings: &cubecl::prelude::KernelSettings,
                                   position: usize|
             -> ::std::sync::Arc<dyn ::core::any::Any> {
                match position {
                    0usize => ::std::sync::Arc::new(
                        <Array<F> as cubecl::prelude::LaunchArgExpand>::expand_output(
                            self.output,
                            builder,
                            settings.vectorization_output(0usize),
                        ),
                    ),
                    _ => {
                        panic!("Input {position} is invalid");
                    }
                }
            };
            for i in 0..1usize {
                inputs.insert(i, register_input(&mut builder, &self.settings, i));
            }
            for mapping in &self.settings.mappings {
                let input = inputs.get(&mapping.pos_input).unwrap();
                outputs.insert(mapping.pos_output, input.clone());
            }
            for i in 0..1usize {
                if !outputs.contains_key(&i) {
                    outputs.insert(i, register_output(&mut builder, &self.settings, i));
                }
            }
            let input: &<Array<F> as cubecl::prelude::CubeType>::ExpandType = inputs
                   .get(&0usize)
                   .unwrap()
                   .downcast_ref()
                   .expect(
                       "Output type should be correct. It could be caused by an invalid kernel input/output alias.",
                   );
            let output: &<Array<F> as cubecl::prelude::CubeType>::ExpandType = outputs
                   .get(&0usize)
                   .unwrap()
                   .downcast_ref()
                   .expect(
                       "Output type should be correct. It could be caused by an invalid kernel input/output alias.",
                   );
            expand::<F>(&mut builder.context, input.clone(), output.clone());
            builder.build(self.settings.clone())
        }
        fn id(&self) -> cubecl::KernelId {
            cubecl::KernelId::new::<Self>().info(self.settings.clone())
        }
    }
    #[allow(clippy::too_many_arguments)]
    ///Launch the kernel [gelu_array()] on the given runtime
    pub unsafe fn launch_unchecked<'kernel, F: Float, __R: cubecl::prelude::Runtime>(
        __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
        __cube_count: cubecl::prelude::CubeCount<__R::Server>,
        __cube_dim: cubecl::prelude::CubeDim,
        input: cubecl::RuntimeArg<'kernel, Array<F>, __R>,
        output: cubecl::RuntimeArg<'kernel, Array<F>, __R>,
    ) -> () {
        use cubecl::frontend::ArgSettings as _;
        let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
        __settings =
            cubecl::prelude::ArgSettings::<__R>::configure_input(&input, 0usize, __settings);
        __settings =
            cubecl::prelude::ArgSettings::<__R>::configure_output(&output, 0usize, __settings);
        let input_arg = <Array<F> as LaunchArg>::compilation_arg(&input);
        let output_arg = <Array<F> as LaunchArg>::compilation_arg(&output);
        let kernel = GeluArray::<F, __R>::new(__settings, input_arg, output_arg);
        let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
        input.register(&mut launcher);
        output.register(&mut launcher);
        launcher.launch_unchecked(__cube_count, kernel, __client);
    }
}

#[cube]
fn gelu_scalar<F: Float>(x: F) -> F {
    x * F::erf(x / F::new(2.0f32.sqrt()) + F::new(1.0)) / F::new(2.0)
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            ArrayArg::from_raw_parts(&input_handle, input.len(), 1),
            ArrayArg::from_raw_parts(&output_handle, input.len(), 1),
        )
    };

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("Executed gelu with runtime {:?} => {output:?}", R::name());
}
