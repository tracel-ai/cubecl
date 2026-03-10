use core::marker::PhantomData;

use crate::{self as cubecl, IntoRuntime, as_bytes};
use cubecl::prelude::*;
use cubecl_macros::CubeTypeMut;

#[derive_cube_comptime]
#[derive(CubeLaunch, CubeType)]
pub enum TestEnum<T: LaunchArg> {
    A(i32, u32),
    B(BStruct),
    C(T),
    D,
    E { x: i32 },
    F { x: i32, y: u32 },
}

#[derive_cube_comptime]
#[derive(CubeLaunch, CubeType, CubeTypeMut)]
#[cube(runtime_variants)]
pub enum RuntimeEnumEmpty {
    A,
    B,
    C,
}

#[derive_cube_comptime]
#[derive(CubeLaunch, CubeType, IntoRuntime)]
#[cube(runtime_variants)]
pub enum RuntimeEnumSingleValue {
    A,
    B(BStruct),
    C,
}

#[derive_cube_comptime]
#[derive(CubeLaunch, CubeType, Default, IntoRuntime)]
pub struct BStruct {
    x: i32,
    y: u32,
}

#[allow(clippy::derivable_impls)]
impl Default for BStructCompilationArg {
    fn default() -> Self {
        Self { x: (), y: () }
    }
}

impl<R: Runtime> Default for BStructLaunch<R> {
    fn default() -> Self {
        Self {
            _phantom_runtime: PhantomData,
            x: 0,
            y: 0,
        }
    }
}

// We just check that it compiles for the syntax.
#[allow(unused_variables)]
#[allow(clippy::needless_match)]
#[cube(launch)]
pub fn kernel_comptime_values(#[comptime] test: TestEnum<i32>) {
    let test2 = comptime! {
        match test {
            TestEnum::A(x, y) => TestEnum::A(x, y),
            TestEnum::B(x) => TestEnum::B(x),
            TestEnum::C(x) => TestEnum::C(x),
            TestEnum::D => TestEnum::D,
            TestEnum::E {x} => TestEnum::E { x },
            TestEnum::F {x, ..} => TestEnum::F { x, y: 2 }
        }
    };
}

// We just check that it compiles for the syntax.
#[allow(unused_variables)]
#[allow(clippy::needless_match)]
#[cube(launch)]
pub fn kernel_runtime_values(test: TestEnum<i32>) {
    let test2: TestEnum<i32> = match test {
        TestEnum::A(x, y) => TestEnum::new_A(x, y),
        TestEnum::B(x) => TestEnum::new_B(x),
        TestEnum::C(x) => TestEnum::new_C(x),
        TestEnum::D => TestEnum::new_D(),
        TestEnum::E { x } => TestEnum::new_E(x),
        TestEnum::F { x, .. } => TestEnum::new_F(x, 2),
    };
}

#[cube(launch)]
pub fn kernel_scalar_enum(test: TestEnum<i32>, output: &mut Array<f32>) {
    match test {
        TestEnum::A(x, y) | TestEnum::F { x, y } => {
            output[0] = f32::cast_from(x) + f32::cast_from(y);
        }
        TestEnum::B(b) => {
            output[0] = f32::cast_from(b.x) + f32::cast_from(b.y);
        }
        TestEnum::C(x) | TestEnum::E { x } => {
            output[0] = f32::cast_from(x);
        }
        TestEnum::D => {
            output[0] = 999.0;
        }
    };
}

#[cube(launch_unchecked)]
pub fn kernel_runtime_variants_empty(test: u32, output: &mut Array<f32>) {
    let test = if test == 0 {
        RuntimeEnumEmpty::new_A()
    } else if test == 1 {
        RuntimeEnumEmpty::new_B()
    } else {
        RuntimeEnumEmpty::new_C()
    };
    match test {
        RuntimeEnumEmpty::B => {
            output[0] = 20.0;
        }
        RuntimeEnumEmpty::C => {
            output[0] = 30.0;
        }
        RuntimeEnumEmpty::A => {
            output[0] = 10.0;
        }
    };
}

#[cube(launch_unchecked)]
pub fn kernel_runtime_variants_empty_wildcard(test: RuntimeEnumEmpty, output: &mut Array<f32>) {
    match test {
        RuntimeEnumEmpty::B => {
            output[0] = 20.0;
        }
        _ => {
            output[0] = 10.0;
        }
    };
}

#[cube(launch_unchecked)]
pub fn kernel_runtime_variants_value(test: RuntimeEnumSingleValue, output: &mut Array<f32>) {
    let value: i32 = match test {
        RuntimeEnumSingleValue::B(bstruct) => bstruct.x,
        RuntimeEnumSingleValue::A => 0i32,
        RuntimeEnumSingleValue::C => 1i32,
    };
    output[0] = value as f32;
}

pub fn test_scalar_enum<R: Runtime>(client: ComputeClient<R>) {
    let array = client.empty(core::mem::size_of::<f32>());

    kernel_scalar_enum::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TestEnumArgs::<i32, R>::C(10),
        unsafe { ArrayArg::from_raw_parts(array.clone(), 1) },
    );
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 10.0);
}

pub fn test_runtime_variants_empty<R: Runtime>(client: ComputeClient<R>) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_empty::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            1,
            ArrayArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 20.0);
}

pub fn test_runtime_variants_value<R: Runtime>(client: ComputeClient<R>) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_value::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            RuntimeEnumSingleValueLaunch::Runtime(RuntimeEnumSingleValueArgs::B(
                BStructLaunch::new(5, 5),
            )),
            ArrayArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 5.0);
}

pub fn test_runtime_variants_empty_wildcard<R: Runtime>(client: ComputeClient<R>) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_empty_wildcard::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            RuntimeEnumEmptyLaunch::Runtime(RuntimeEnumEmptyArgs::C),
            ArrayArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 10.0);
}

#[derive(CubeLaunch, CubeType)]
pub enum ArrayFloatInt {
    Float(Array<f32>),
    Int(Array<i32>),
}

#[cube(launch)]
fn kernel_array_float_int(array: &mut ArrayFloatInt) {
    if UNIT_POS == 0 {
        match array {
            ArrayFloatInt::Float(array) => array[0] = 10.0,
            ArrayFloatInt::Int(array) => array[0] = 20,
        }
    }
}

pub fn test_array_float_int<R: Runtime, T: CubePrimitive + CubeElement>(
    client: &ComputeClient<R>,
    expected: T,
) {
    let array = client.empty(core::mem::size_of::<T>());

    kernel_array_float_int::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
            ArrayFloatIntArgs::Float(unsafe { ArrayArg::from_raw_parts(array.clone(), 1) })
        } else {
            ArrayFloatIntArgs::Int(unsafe { ArrayArg::from_raw_parts(array.clone(), 1) })
        },
    );

    let bytes = client.read_one_unchecked(array);
    let actual = T::from_bytes(&bytes);

    assert_eq!(actual[0], expected);
}

#[derive(CubeLaunch, CubeType)]
pub enum SimpleEnum<T: LaunchArg> {
    Variant(T),
}

#[cube(launch)]
fn kernel_tuple_enum(first: &mut SimpleEnum<Array<u32>>, second: SimpleEnum<Array<u32>>) {
    if UNIT_POS == 0 {
        match (first, second) {
            (SimpleEnum::Variant(x), SimpleEnum::Variant(y)) => {
                x[0] = y[0];
            }
        }
    }
}

pub fn test_tuple_enum<R: Runtime>(client: &ComputeClient<R>) {
    let first = client.create_from_slice(as_bytes![u32: 20]);
    let second = client.create_from_slice(as_bytes![u32: 5]);

    kernel_tuple_enum::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        SimpleEnumArgs::<Array<u32>, R>::Variant(unsafe {
            ArrayArg::from_raw_parts(first.clone(), 1)
        }),
        SimpleEnumArgs::<Array<u32>, R>::Variant(unsafe { ArrayArg::from_raw_parts(second, 1) }),
    );

    let bytes = client.read_one_unchecked(first);
    let actual = u32::from_bytes(&bytes);

    assert_eq!(actual[0], 5);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_enums {
    () => {
        mod enums {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn scalar_enum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_scalar_enum::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn runtime_enum_empty() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_empty::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn runtime_enum_value() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_value::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn runtime_enum_empty_wildcard() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_empty_wildcard::<
                    TestRuntime,
                >(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn array_float_int() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, f32>(
                    &client, 10.0,
                );
                cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, i32>(
                    &client, 20,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn tuple_enum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_tuple_enum::<TestRuntime>(&client);
            }
        }
    };
}
