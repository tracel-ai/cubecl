use alloc::boxed::Box;
use cubecl_runtime::runtime::Runtime;

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

// Derivable, but the type is generated.
#[allow(clippy::derivable_impls)]
impl Default for BStructLaunch {
    fn default() -> Self {
        Self { x: 0, y: 0 }
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
pub fn kernel_scalar_enum(test: TestEnum<i32>, output: &mut [f32]) {
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
pub fn kernel_runtime_variants_empty(test: u32, output: &mut [f32]) {
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
pub fn kernel_runtime_variants_empty_wildcard(test: RuntimeEnumEmpty, output: &mut [f32]) {
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
pub fn kernel_runtime_variants_value(test: RuntimeEnumSingleValue, output: &mut [f32]) {
    let value: i32 = match test {
        RuntimeEnumSingleValue::B(bstruct) => bstruct.x,
        RuntimeEnumSingleValue::A => 0i32,
        RuntimeEnumSingleValue::C => 1i32,
    };
    output[0] = value as f32;
}

pub fn test_scalar_enum<R: Runtime>(client: Client) {
    let array = client.empty(core::mem::size_of::<f32>());

    kernel_scalar_enum::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TestEnumArgs::<i32>::C(10),
        unsafe { BufferArg::from_raw_parts(array.clone(), 1) },
    );
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 10.0);
}

pub fn test_runtime_variants_empty<R: Runtime>(client: Client) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_empty::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            1,
            BufferArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 20.0);
}

pub fn test_runtime_variants_value<R: Runtime>(client: Client) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_value::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            RuntimeEnumSingleValueLaunch::Runtime(RuntimeEnumSingleValueArgs::B(
                BStructLaunch::new(5, 5),
            )),
            BufferArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 5.0);
}

pub fn test_runtime_variants_empty_wildcard<R: Runtime>(client: Client) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_empty_wildcard::launch_unchecked(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            RuntimeEnumEmptyLaunch::Runtime(RuntimeEnumEmptyArgs::C),
            BufferArg::from_raw_parts(array.clone(), 1),
        )
    };
    let bytes = client.read_one_unchecked(array);
    let actual = f32::from_bytes(&bytes);

    assert_eq!(actual[0], 10.0);
}

#[derive(CubeLaunch, CubeType)]
pub enum ArrayFloatInt {
    Float(Box<[f32]>),
    Int(Box<[i32]>),
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

pub fn test_array_float_int<R: Runtime, T: Scalar + CubeElement>(client: &Client, expected: T) {
    let array = client.empty(core::mem::size_of::<T>());

    kernel_array_float_int::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
            ArrayFloatIntArgs::Float(unsafe { BufferArg::from_raw_parts(array.clone(), 1) })
        } else {
            ArrayFloatIntArgs::Int(unsafe { BufferArg::from_raw_parts(array.clone(), 1) })
        },
    );

    let bytes = client.read_one_unchecked(array);
    let actual = T::from_bytes(&bytes);

    assert_eq!(actual[0], expected);
}

#[derive(CubeLaunch, CubeType)]
pub enum SimpleEnum<T: CubeType> {
    Variant(T),
}

#[cube(launch)]
fn kernel_tuple_enum(first: SimpleEnum<&mut [u32]>, second: SimpleEnum<&[u32]>) {
    if UNIT_POS == 0 {
        match (first, second) {
            (SimpleEnum::Variant(x), SimpleEnum::Variant(y)) => {
                x[0] = y[0];
            }
        }
    }
}

pub fn test_tuple_enum<R: Runtime>(client: &Client) {
    let first = client.create_from_slice(as_bytes![u32: 20]);
    let second = client.create_from_slice(as_bytes![u32: 5]);

    kernel_tuple_enum::launch(
        client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        SimpleEnumArgs::<&mut [u32]>::Variant(unsafe {
            BufferArg::from_raw_parts(first.clone(), 1)
        }),
        SimpleEnumArgs::<&[u32]>::Variant(unsafe { BufferArg::from_raw_parts(second, 1) }),
    );

    let bytes = client.read_one_unchecked(first);
    let actual = u32::from_bytes(&bytes);

    assert_eq!(actual[0], 5);
}

#[cube(launch)]
pub fn kernel_comptime_option_scalar(opt: ComptimeOption<f32>, output: &mut [f32]) {
    // The comptime marker keeps the match on the expand-time discriminant: without it, a match
    // with a single payload arm is classified as a runtime match, which ComptimeOption is not.
    #[comptime]
    let value = match opt {
        ComptimeOption::Some(x) => x,
        ComptimeOption::None => 999.0,
    };
    output[0] = value;
}

pub fn test_comptime_option_scalar<R: Runtime>(client: &Client) {
    let read = |args: ComptimeOptionArgs<f32>| {
        let array = client.empty(core::mem::size_of::<f32>());
        kernel_comptime_option_scalar::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            args,
            unsafe { BufferArg::from_raw_parts(array.clone(), 1) },
        );
        let bytes = client.read_one_unchecked(array);
        f32::from_bytes(&bytes)[0]
    };

    assert_eq!(read(ComptimeOptionArgs::Some(3.5)), 3.5);
    assert_eq!(read(ComptimeOptionArgs::None), 999.0);
}

#[derive(CubeType, Clone, Copy)]
pub enum GuardMode {
    Clamp,
    Nan,
    Extrapolate,
}

/// Match guards on a plain enum argument, mirroring the extrapolation-offset pattern.
#[cube]
fn guarded_offset(mode: GuardMode, query: f32, lo: f32, hi: f32) -> f32 {
    match mode {
        GuardMode::Extrapolate if query < lo || query > hi => query - lo,
        GuardMode::Nan if query < lo => -1.0f32,
        _ => 0.0f32,
    }
}

#[cube(launch)]
pub fn kernel_guard_enum_argument(query: f32, output: &mut [f32]) {
    output[0] = guarded_offset(GuardMode::new_Extrapolate(), query, 0.0, 1.0);
    output[1] = guarded_offset(GuardMode::new_Nan(), query, 0.0, 1.0);
    output[2] = guarded_offset(GuardMode::new_Clamp(), query, 0.0, 1.0);
}

pub fn test_guard_enum_argument<R: Runtime>(client: &Client) {
    let read = |query: f32| {
        let array = client.empty(3 * core::mem::size_of::<f32>());
        kernel_guard_enum_argument::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            query,
            unsafe { BufferArg::from_raw_parts(array.clone(), 3) },
        );
        let bytes = client.read_one_unchecked(array);
        f32::from_bytes(&bytes).to_vec()
    };

    assert_eq!(read(-2.0), [-2.0, -1.0, 0.0]);
    assert_eq!(read(0.5), [0.0, 0.0, 0.0]);
    assert_eq!(read(3.0), [3.0, 0.0, 0.0]);
}

/// A guard reading a binding from its own pattern, falling through to the same variant.
#[cube(launch)]
pub fn kernel_guard_binding(test: TestEnum<i32>, output: &mut [f32]) {
    output[0] = match test {
        TestEnum::C(x) if x > 5 => 1.0f32,
        TestEnum::C(_) => 2.0f32,
        TestEnum::D => 3.0f32,
        _ => 4.0f32,
    };
}

pub fn test_guard_binding<R: Runtime>(client: &Client) {
    let read = |args: TestEnumArgs<i32>| {
        let array = client.empty(core::mem::size_of::<f32>());
        kernel_guard_binding::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            args,
            unsafe { BufferArg::from_raw_parts(array.clone(), 1) },
        );
        let bytes = client.read_one_unchecked(array);
        f32::from_bytes(&bytes)[0]
    };

    assert_eq!(read(TestEnumArgs::C(10)), 1.0);
    assert_eq!(read(TestEnumArgs::C(3)), 2.0);
    assert_eq!(read(TestEnumArgs::D), 3.0);
    assert_eq!(read(TestEnumArgs::A(1, 2)), 4.0);
}

/// A runtime guard on a runtime-variant enum.
#[cube(launch_unchecked)]
pub fn kernel_guard_runtime_variants(test: RuntimeEnumEmpty, flag: u32, output: &mut [f32]) {
    match test {
        RuntimeEnumEmpty::B if flag > 0 => {
            output[0] = 21.0;
        }
        RuntimeEnumEmpty::B => {
            output[0] = 20.0;
        }
        _ => {
            output[0] = 10.0;
        }
    };
}

pub fn test_guard_runtime_variants<R: Runtime>(client: &Client) {
    let read = |variant: RuntimeEnumEmptyArgs, flag: u32| {
        let array = client.empty(core::mem::size_of::<f32>());
        unsafe {
            kernel_guard_runtime_variants::launch_unchecked(
                client,
                CubeCount::new_single(),
                CubeDim::new_single(),
                RuntimeEnumEmptyLaunch::Runtime(variant),
                flag,
                BufferArg::from_raw_parts(array.clone(), 1),
            )
        };
        let bytes = client.read_one_unchecked(array);
        f32::from_bytes(&bytes)[0]
    };

    assert_eq!(read(RuntimeEnumEmptyArgs::B, 1), 21.0);
    assert_eq!(read(RuntimeEnumEmptyArgs::B, 0), 20.0);
    assert_eq!(read(RuntimeEnumEmptyArgs::A, 1), 10.0);
}

/// A runtime guard on an integer match, which lowers to a switch.
#[cube(launch)]
pub fn kernel_guard_numeric(sel: u32, flag: u32, output: &mut [f32]) {
    output[0] = match sel {
        1 if flag > 0 => 11.0f32,
        1 => 10.0f32,
        2 | 3 => 20.0f32,
        _ => 0.0f32,
    };
}

pub fn test_guard_numeric<R: Runtime>(client: &Client) {
    let read = |sel: u32, flag: u32| {
        let array = client.empty(core::mem::size_of::<f32>());
        kernel_guard_numeric::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            sel,
            flag,
            unsafe { BufferArg::from_raw_parts(array.clone(), 1) },
        );
        let bytes = client.read_one_unchecked(array);
        f32::from_bytes(&bytes)[0]
    };

    assert_eq!(read(1, 1), 11.0);
    assert_eq!(read(1, 0), 10.0);
    assert_eq!(read(3, 1), 20.0);
    assert_eq!(read(7, 1), 0.0);
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
            fn comptime_option_scalar() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_comptime_option_scalar::<TestRuntime>(
                    &client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn tuple_enum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_tuple_enum::<TestRuntime>(&client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn guard_enum_argument() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_guard_enum_argument::<TestRuntime>(&client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn guard_binding() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_guard_binding::<TestRuntime>(&client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn guard_runtime_variants() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_guard_runtime_variants::<TestRuntime>(
                    &client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn guard_numeric() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_guard_numeric::<TestRuntime>(&client);
            }
        }
    };
}
