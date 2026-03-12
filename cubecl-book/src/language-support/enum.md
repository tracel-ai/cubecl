# Enum Support

CubeCL provides robust support for Rust enums, enabling you to express variant-based logic in your
GPU kernels. Enums can be used as kernel arguments, returned from kernels, or as intermediate types
within your GPU code. This allows you to write expressive, idiomatic Rust code that maps efficiently
to GPU kernels.

CubeCL supports two types of enums:

- comptime variants with optional runtime values
- runtime variants with up to one runtime value

## Runtime variant restrictions

Because of limitations in the backend compilers, runtime-variant enums have certain limitations:

- they must be valueless, or have exactly one tuple-style value (i.e. `Option`)
- to construct them the value must implement `Default + IntoRuntime`, or a custom "empty" value must
  be provided. For `Vector`, the provided empty value _must_ have the same size as the non-empty
  value.
- to construct them based on a runtime condition, they must implement `Assign`/`CubeTypeMut`.

## Defining comptime-variant enums

To use a comptime-variant enum in a CubeCL kernel, simply derive the required traits on the enum you
want to use:

- `CubeType` enables the enum to be used as a CubeCL type in a kernel.
- `CubeLaunch` allows the enum to be used as a kernel argument.

Enums can also have data associated with their variants, as long as all fields implement the
required CubeCL traits, here's an example that is available in cubecl-std:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub enum ComptimeOption<T: CubeLaunch + CubeType> {
    Some(T),
    None,
}
```

## Defining runtime-variant enums

For runtime enums, the derive should have an additional `#[cube(runtime_variants)]` attribute. To
actually use them, you likely also need to derive `CubeTypeMut` for the assign implementation.

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeTypeMut, CubeLaunch)]
#[cube(runtime_variants)]
pub enum RuntimeOption<T: CubeLaunch + CubeType> {
    Some(T),
    None,
}
```

## Using enums in kernels

Enums can be passed as kernel arguments or used as local variables:

```rust,ignore
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub enum Function {
    AffineTransformation { a: f32, b: f32 },
    Cos,
    DivideScalar(f32),
}

#[cube(launch_unchecked)]
pub fn kernel_enum_example(
    input: &Array<Vector<f32>>,
    output: &mut Array<Vector<f32>>,
    function: Function,
) {
    output[UNIT_POS] = match function {
        Function::AffineTransformation { a, b } => Vector::new(a) * input[UNIT_POS] + Vector::new(b),
        Function::Cos => Vector::cos(input[UNIT_POS]),
        Function::DivideScalar(coef) => input[UNIT_POS] / Vector::new(coef),
    }
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#     let input = client.create(f32::as_bytes(&[1.0, -2.0, 0.5]));
#     let output = client.empty(3 * core::mem::size_of::<f32>());
#     unsafe {
#         kernel_enum_example::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(3),
#             ArrayArg::from_raw_parts::<f32>(&input, 3, 2),
#             ArrayArg::from_raw_parts::<f32>(&output, 3, 2),
#             FunctionArgs::AffineTransformation {
#                 a: 1.0,
#                 b: 2.0,
#             },
#         )
#     };
#     println!(
#         "Executed kernel_enum_example with runtime {:?} => {:?}",
#         R::name(&client),
#         f32::from_bytes(&client.read_one(output.binding()))
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
# }
```

You can also use enums with data in pattern matching:

```rust,ignore
# use cubecl::prelude::*;
#
# #[derive(CubeType, CubeLaunch)]
# pub enum ComptimeOption<T: CubeType> {
#     Some(T),
#     None,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_enum_option(input: &Array<f32>, output: &mut Array<f32>, maybe: ComptimeOption<Array<f32>>) {
    #[comptime]
    output[UNIT_POS] = match maybe {
        ComptimeOption::Some(val) => input[UNIT_POS] + val[UNIT_POS],
        ComptimeOption::None => input[UNIT_POS],
    };
}
```

Note the `#[comptime]` above the `match` statement. The macro will try to detect whether a match is
comptime or runtime based on the constraints of runtime-variable enums, but detection may be
incorrect for enums that could be runtime but aren't (i.e. `ComptimeOption`). To override the
detection and force a match to be comptime-variant, simply add `#[comptime]` above it. The same
applies to `if let`.

## Adding methods to enums

You can implement methods for enums using the `#[cube]` attribute on the `impl` block:

```rust,ignore
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub enum Function {
    AffineTransformation { a: f32, b: f32 },
    Cos,
    DivideScalar(f32),
}

#[cube]
impl Function {
    pub fn apply(self, x: Vector<f32>) -> Vector<f32> {
        match self {
            Function::AffineTransformation { a, b } => Vector::new(a) * x + Vector::new(b),
            Function::Cos => Vector::cos(x),
            Function::DivideScalar(coef) => x / Vector::new(coef),
        }
    }
}

#[cube(launch_unchecked)]
pub fn kernel_enum_example(
    input: &Array<Vector<f32>>,
    output: &mut Array<Vector<f32>>,
    function: Function,
    bias: Option<f32>,
) {
    let mut value = function.apply(input[UNIT_POS]);
    // Runtime selected. Use `ComptimeOption` for things like optional tensors.
    if let Some(bias) = bias {
        value += Vector::cast_from(bias);
    }
    output[UNIT_POS] = value;
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#     let input = client.create(f32::as_bytes(&[1.0, -2.0, 0.5]));
#     let output = client.empty(3 * core::mem::size_of::<f32>());
#     unsafe {
#         kernel_enum_example::launch_unchecked(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new_1d(3),
#             ArrayArg::from_raw_parts::<f32>(&input, 3, 2),
#             ArrayArg::from_raw_parts::<f32>(&output, 3, 2),
#             FunctionArgs::AffineTransformation {
#                 a: 1.0,
#                 b: 2.0,
#             },
#             OptionArg::Some(1.2),
#         )
#     };
#     println!(
#         "Executed kernel_enum_example with runtime {:?} => {:?}",
#         R::name(&client),
#         f32::from_bytes(&client.read_one(output.binding()))
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
# }
```
