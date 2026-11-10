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

#[cube(launch_unchecked)]
pub fn kernel_runtime_variants_if_let_value(test: RuntimeEnumSingleValue, output: &mut [f32]) {
    if let RuntimeEnumSingleValue::B(bstruct) = test {
        output[0] = bstruct.x as f32;
    } else {
        output[0] = -1.0f32;
    }
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

pub fn test_runtime_variants_if_let_value<R: Runtime>(client: Client) {
    let array = client.empty(core::mem::size_of::<f32>());

    unsafe {
        kernel_runtime_variants_if_let_value::launch_unchecked(
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

/// What a look at a unit's flags finds, built in a nested if/else and returned from a function.
#[derive(CubeType, CubeTypeMut)]
#[cube(runtime_variants)]
pub enum LookState {
    Done,
    Claimed,
    Pending,
    Failed,
}

#[cube]
fn try_claim(claims: &[Atomic<u32>], unit: usize) -> bool {
    claims[unit].fetch_add(1u32) == 0u32
}

#[cube]
fn look(flags: u32, claims: &[Atomic<u32>], unit: usize) -> LookState {
    let failed = flags & 1u32 != 0u32;
    if failed {
        LookState::new_Failed()
    } else {
        let done = flags & 2u32 != 0u32;
        let ready = flags & 4u32 != 0u32;
        if done {
            LookState::new_Done()
        } else if ready && try_claim(claims, unit) {
            LookState::new_Claimed()
        } else {
            LookState::new_Pending()
        }
    }
}

#[cube(launch)]
pub fn kernel_runtime_variants_nested(claims: &[Atomic<u32>], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    match look(UNIT_POS, claims, unit) {
        LookState::Done => {
            output[unit] = 0u32;
        }
        LookState::Claimed => {
            output[unit] = 1u32;
        }
        LookState::Pending => {
            output[unit] = 2u32;
        }
        LookState::Failed => {
            output[unit] = 3u32;
        }
    }
}

#[cube(launch)]
pub fn kernel_runtime_variants_nested_loop(claims: &[Atomic<u32>], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    let mut flags = UNIT_POS;
    let mut rounds = 0u32;
    let mut waiting = true;
    while waiting {
        waiting = false;
        match look(flags, claims, unit) {
            LookState::Done => {
                output[unit] = 10u32 * rounds;
            }
            LookState::Claimed => {
                output[unit] = 1u32 + 10u32 * rounds;
            }
            LookState::Pending => {
                flags |= 4u32;
                rounds += 1u32;
                if rounds < 3u32 {
                    waiting = true;
                } else {
                    output[unit] = 2u32 + 10u32 * rounds;
                }
            }
            LookState::Failed => {
                output[unit] = 3u32 + 10u32 * rounds;
            }
        }
    }
}

#[cube(launch)]
pub fn kernel_runtime_variants_nested_if_let(claims: &[Atomic<u32>], output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    if let LookState::Claimed = look(UNIT_POS, claims, unit) {
        output[unit] = 1u32;
    } else {
        output[unit] = 0u32;
    }
}

/// How a test of [`look`] reads its result.
#[derive(Clone, Copy, Debug)]
pub enum NestedLook {
    /// One `match`.
    Match,
    /// A `match` in a `while` loop that looks again while pending.
    Loop,
    /// An `if let` on the claimed variant.
    IfLet,
}

/// Unit `u` looks with flags `u`: bit 0 failed, bit 1 done, bit 2 ready. Units 8 to 15 start
/// with a free claim. Each look runs `try_claim` at most once, so a scrutinee expanded per arm shows
/// up both in the arm matched and in the claim counts.
pub fn test_runtime_variants_nested<R: Runtime>(client: Client, case: NestedLook) {
    let units = 16usize;
    let initial: alloc::vec::Vec<u32> = (0..units).map(|unit| u32::from(unit < 8)).collect();
    let claims = client.create_from_slice(u32::as_bytes(&initial));
    let output = client.create_from_slice(u32::as_bytes(&alloc::vec![99u32; units]));

    let claims_arg = unsafe { BufferArg::from_raw_parts(claims.clone(), units) };
    let output_arg = unsafe { BufferArg::from_raw_parts(output.clone(), units) };
    let cube_dim = CubeDim::new_1d(units as u32);
    let count = CubeCount::new_single();
    let (expected, expected_claims): ([u32; 16], [u32; 16]) = match case {
        NestedLook::Match => {
            kernel_runtime_variants_nested::launch(
                &client, count, cube_dim, claims_arg, output_arg,
            );
            (
                [2, 3, 0, 3, 2, 3, 0, 3, 2, 3, 0, 3, 1, 3, 0, 3],
                [1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            )
        }
        NestedLook::Loop => {
            kernel_runtime_variants_nested_loop::launch(
                &client, count, cube_dim, claims_arg, output_arg,
            );
            (
                [32, 3, 0, 3, 32, 3, 0, 3, 11, 3, 0, 3, 1, 3, 0, 3],
                [3, 1, 1, 1, 4, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
            )
        }
        NestedLook::IfLet => {
            kernel_runtime_variants_nested_if_let::launch(
                &client, count, cube_dim, claims_arg, output_arg,
            );
            (
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            )
        }
    };
    let actual = client.read_one_unchecked(output);
    assert_eq!(u32::from_bytes(&actual), &expected, "matched arms");
    let actual_claims = client.read_one_unchecked(claims);
    assert_eq!(u32::from_bytes(&actual_claims), &expected_claims, "claims");
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
            fn runtime_enum_if_let_value() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_if_let_value::<TestRuntime>(
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
            fn runtime_enum_call_scrutinee() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_nested::<TestRuntime>(
                    client,
                    cubecl_core::runtime_tests::enums::NestedLook::Match,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn runtime_enum_call_scrutinee_loop() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_nested::<TestRuntime>(
                    client,
                    cubecl_core::runtime_tests::enums::NestedLook::Loop,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn runtime_enum_call_scrutinee_if_let() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::enums::test_runtime_variants_nested::<TestRuntime>(
                    client,
                    cubecl_core::runtime_tests::enums::NestedLook::IfLet,
                );
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
        }
    };
}
