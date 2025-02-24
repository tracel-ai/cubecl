use crate as cubecl;
use cubecl::prelude::*;

#[cube(launch)]
pub fn kernel_option_scalar(array: &mut Array<i32>, value: Option<i32>) {
    if UNIT_POS == 0 {
        match comptime!(value) {
            Some(value) => array[0] = value,
            None => {}
        }
    }
}

// pub mod kernel_option_scalar {
//     use super::*;
//     #[allow(unused, clippy::all)]
//     pub fn expand(
//         context: &mut cubecl::prelude::Scope,
//         array: <Array<i32> as cubecl::prelude::CubeType>::ExpandType,
//         value: <Option<i32> as cubecl::prelude::CubeType>::ExpandType,
//     ) -> <() as cubecl::prelude::CubeType>::ExpandType {
//         cubecl::frontend::debug_source_expand(
//             context,
//             "kernel_option_scalar",
//             file!(),
//             "",
//             line!(),
//             column!(),
//         );
//         cubecl::frontend::CubeDebug::set_debug_name(&array, context, "array");
//         cubecl::frontend::CubeDebug::set_debug_name(&value, context, "value");
//         use cubecl::prelude::IntoRuntime as _;
//         {
//             {
//                 let _cond = {
//                     let _lhs = UNIT_POS::expand(context);
//                     let _rhs = cubecl::frontend::ExpandElementTyped::from_lit(context, 0);
//                     cubecl::frontend::spanned_expand(context, line!(), column!(), |context| {
//                         cubecl::frontend::eq::expand(context, _lhs.into(), _rhs.into())
//                     })
//                 };
//                 cubecl::frontend::branch::if_expand(context, _cond.into(), |context| {
//                     match comptime!(value) {
//                         Some(value) => {
//                             let _array = array;
//                             let _index = cubecl::frontend::ExpandElementTyped::from_lit(context, 0);
//                             let _value = value;
//                             cubecl::frontend::index_assign::expand(
//                                 context,
//                                 _array.into(),
//                                 _index.into(),
//                                 _value.into(),
//                             )
//                         }
//                         None => {}
//                     }
//                 });
//             }
//         }
//     }
//     ///kernel_option_scalar Kernel
//     pub struct KernelOptionScalar<__R: cubecl::prelude::Runtime> {
//         settings: cubecl::prelude::KernelSettings,
//         value: <Option<i32> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         array: <Array<i32> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         __ty: ::core::marker::PhantomData<(__R)>,
//     }
//     #[allow(clippy::too_many_arguments)]
//     impl<__R: cubecl::prelude::Runtime> KernelOptionScalar<__R> {
//         pub fn new(
//             settings: cubecl::prelude::KernelSettings,
//             value: <Option<i32> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//             array: <Array<i32> as cubecl::prelude::LaunchArgExpand>::CompilationArg,
//         ) -> Self {
//             println!("NEW VALUE: {value:?}");
//             Self {
//                 settings: settings.kernel_name("kernel_option_scalar"),
//                 value,
//                 array,
//                 __ty: ::core::marker::PhantomData,
//             }
//         }
//     }
//     impl<__R: cubecl::prelude::Runtime> cubecl::Kernel for KernelOptionScalar<__R> {
//         fn define(&self) -> cubecl::prelude::KernelDefinition {
//             let mut builder = cubecl::prelude::KernelBuilder::default();
//             println!("BEFORE CAST: {:?}", self.value);
//             // println!(
//             //     "AFTER CAST: {:?}",
//             //     self.value
//             //         .dynamic_cast::<<Option<i32> as LaunchArgExpand>::CompilationArg>()
//             // );

//             let value = <Option<i32> as cubecl::prelude::LaunchArgExpand>::expand(
//                 // &self.value.dynamic_cast(),
//                 &self.value,
//                 &mut builder,
//             );
//             println!("VALUE IN DEFINE: {value:?}");
//             let array = <Array<i32> as cubecl::prelude::LaunchArgExpand>::expand_output(
//                 &self.array.dynamic_cast(),
//                 &mut builder,
//             );
//             expand(&mut builder.context, array.clone(), value.clone());
//             builder.build(self.settings.clone())
//         }
//         fn id(&self) -> cubecl::KernelId {
//             let cube_dim = self.settings.cube_dim.clone();
//             cubecl::KernelId::new::<Self>().info((cube_dim, self.value.clone(), self.array.clone()))
//         }
//     }
//     #[allow(clippy::too_many_arguments)]
//     ///Launch the kernel [kernel_option_scalar()] on the given runtime
//     pub fn launch<'kernel, __R: cubecl::prelude::Runtime>(
//         __client: &cubecl::prelude::ComputeClient<__R::Server, __R::Channel>,
//         __cube_count: cubecl::prelude::CubeCount,
//         __cube_dim: cubecl::prelude::CubeDim,
//         array: cubecl::RuntimeArg<'kernel, Array<i32>, __R>,
//         value: cubecl::RuntimeArg<'kernel, Option<i32>, __R>,
//     ) -> () {
//         use cubecl::frontend::ArgSettings as _;
//         let mut __settings = cubecl::prelude::KernelSettings::default().cube_dim(__cube_dim);
//         let input_arg_0 =
//             <Option<i32> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&value);
//         println!("INPUT ARG 0 {input_arg_0:?}");
//         let output_arg_0 =
//             <Array<i32> as cubecl::prelude::LaunchArg>::compilation_arg::<__R>(&array);
//         let __kernel = KernelOptionScalar::<__R>::new(__settings, input_arg_0, output_arg_0);
//         let mut launcher = cubecl::prelude::KernelLauncher::<__R>::default();
//         println!("VALUE {value:?}");
//         value.register(&mut launcher);
//         array.register(&mut launcher);
//         launcher.launch(__cube_count, __kernel, __client);
//     }
// }

pub fn test_option_scalar_none<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5]));

    unsafe {
        kernel_option_scalar::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 1, 1),
            None,
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 5);
}

pub fn test_option_scalar_some<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5]));

    unsafe {
        kernel_option_scalar::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 1, 1),
            Some(ScalarArg::new(10)),
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 10);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_option {
    () => {
        use super::*;

        #[test]
        fn test_option_scalar_none() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_scalar_none::<TestRuntime>(client);
        }

        #[test]
        fn test_option_scalar_some() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_scalar_some::<TestRuntime>(client);
        }
    };
}
