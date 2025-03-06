use crate as cubecl;
use cubecl::prelude::*;

#[derive(CubeLaunch, Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub enum TestEnum<
    T: std::fmt::Debug + CubeType + Clone + IntoRuntime + LaunchArg + std::cmp::Eq + std::hash::Hash,
> {
    A(i32, u32),
    B(BStruct),
    C(T),
    D,
    E { x: i32 },
    F { x: i32, y: u32 },
}

#[derive(CubeLaunch, Clone, Debug, Hash, PartialEq, Eq, Copy)]
pub struct BStruct {
    x: i32,
    y: u32,
}

// We just check that it compiles for the syntax.
#[allow(unused_variables)]
#[allow(clippy::needless_match)]
#[cube(launch)]
pub fn kernel_comptime_variants(#[comptime] test: TestEnum<i32>) {
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
#[allow(clippy::clone_on_copy)]
#[cube(launch)]
pub fn kernel_runtime_variants(test: TestEnum<i32>) {
    let test2: TestEnum<i32> = match test.clone() {
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

pub fn test_scalar_enum<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.empty(std::mem::size_of::<f32>());

    kernel_scalar_enum::launch::<R>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TestEnumArgs::<i32, R>::C(ScalarArg::new(10)),
        unsafe { ArrayArg::<R>::from_raw_parts::<f32>(&array, 1, 1) },
    );
    let bytes = client.read_one(array.binding());
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 10.0);
}

#[derive(CubeLaunch)]
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
    client: &ComputeClient<R::Server, R::Channel>,
    expected: T,
) {
    let array = client.empty(std::mem::size_of::<T>());

    kernel_array_float_int::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            ArrayFloatIntArgs::Float(unsafe { ArrayArg::<R>::from_raw_parts::<f32>(&array, 1, 1) })
        } else {
            ArrayFloatIntArgs::Int(unsafe { ArrayArg::<R>::from_raw_parts::<i32>(&array, 1, 1) })
        },
    );

    let bytes = client.read_one(array.binding());
    let actual = T::from_bytes(&bytes);

    assert_eq!(actual[0], expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_enums {
    () => {
        mod enums {
            use super::*;

            #[test]
            fn scalar_enum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_scalar_enum::<TestRuntime>(client);
            }

            #[test]
            fn array_float_int() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, f32>(
                    &client, 10.0,
                );
                cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, i32>(
                    &client, 20,
                );
            }
        }
    };
}
