# Enum Support

CubeCL provides robust support for Rust enums, enabling you to express variant-based logic in your GPU kernels. Enums can be used as kernel arguments, returned from kernels, or as intermediate types within your GPU code. This allows you to write expressive, idiomatic Rust code that maps efficiently to GPU kernels.

## Defining enums

To use an enum in a CubeCL kernel, simply derive the required traits on the enum you want to use:

- `CubeType` enables the enum to be used as a CubeCL type in a kernel.
- `CubeLaunch` allows the enum to be used as a kernel argument or return type.

Enums can also have data associated with their variants, as long as all fields implement the required CubeCL traits, here's an example that is available in cubecl-std:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub enum CubeOption<T: CubeLaunch + CubeType> {
    Some(T),
    None,
}
```

## Using enums in kernels

Enums can be passed as kernel arguments, returned from kernels, or used as local variables:

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
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    function: Function,
) {
    output[UNIT_POS] = match function {
        Function::AffineTransformation { a, b } => Line::new(a) * input[UNIT_POS] + Line::new(b),
        Function::Cos => Line::cos(input[UNIT_POS]),
        Function::DivideScalar(coef) => input[UNIT_POS] / Line::new(coef),
    }
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#     let input = client.create(f32::as_bytes(&[1.0, -2.0, 0.5]));
#     let output = client.empty(3 * core::mem::size_of::<f32>());
#     unsafe {
#         kernel_enum_example::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(3, 1, 1),
#             ArrayArg::from_raw_parts::<f32>(&input, 3, 2),
#             ArrayArg::from_raw_parts::<f32>(&output, 3, 2),
#             FunctionArgs::AffineTransformation {
#                 a: ScalarArg::new(1.0),
#                 b: ScalarArg::new(2.0),
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
# pub enum CubeOption<T: CubeType> {
#     Some(T),
#     None,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_enum_option(input: &Array<f32>, output: &mut Array<f32>, maybe: CubeOption<Array<f32>>) {
    output[UNIT_POS] = match maybe {
        CubeOption::Some(val) => input[UNIT_POS] + val[UNIT_POS],
        CubeOption::None => input[UNIT_POS],
    };
}
```

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
    pub fn apply(self, x: Line<f32>) -> Line<f32> {
        match self {
            Function::AffineTransformation { a, b } => Line::new(a) * x + Line::new(b),
            Function::Cos => Line::cos(x),
            Function::DivideScalar(coef) => x / Line::new(coef),
        }
    }
}

#[cube(launch_unchecked)]
pub fn kernel_enum_example(
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    function: Function,
) {
    output[UNIT_POS] = function.apply(input[UNIT_POS]);
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#     let input = client.create(f32::as_bytes(&[1.0, -2.0, 0.5]));
#     let output = client.empty(3 * core::mem::size_of::<f32>());
#     unsafe {
#         kernel_enum_example::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(3, 1, 1),
#             ArrayArg::from_raw_parts::<f32>(&input, 3, 2),
#             ArrayArg::from_raw_parts::<f32>(&output, 3, 2),
#             FunctionArgs::AffineTransformation {
#                 a: ScalarArg::new(1.0),
#                 b: ScalarArg::new(2.0),
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
