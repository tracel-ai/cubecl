# Struct Support

CubeCL provides robust support for Rust structs, allowing you to organize and modularize your kernel code with zero-cost abstractions. Structs can be used as kernel arguments, returned from kernels, or as intermediate types within your GPU code. This enables you to write idiomatic, maintainable Rust code that maps efficiently to GPU kernels.

## Defining structs

To use a struct in a CubeCL kernel, simply derive the required traits on the struct that you want to use:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub struct Pair<T: CubeLaunch> {
    pub left: T,
    pub right: T,
}
```

- `CubeType` enables the struct to be used as a CubeCL type in a kernel.
- `CubeLaunch` allows the struct to be used as a kernel argument or return type.

Structs can contain other structs, arrays, or generic parameters, as long as all fields implement the required CubeCL traits. Generics are also supported, allowing you to create reusable types that can be instantiated with different types.

## Using structs in kernels

Structs can be passed as kernel arguments if annotated with `CubeLaunch`, returned from kernels, or used as local variables:

```rust,ignore
# use cubecl::prelude::*;
#
# #[derive(CubeType, CubeLaunch)]
# pub struct Pair<T: CubeLaunch> {
#     pub left: T,
#     pub right: T,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_struct_example(pair: &Pair<Array<f32>>, output: &mut Array<f32>) {
    output[UNIT_POS] = pair.left[UNIT_POS] + pair.right[UNIT_POS];
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#
#     let left = [f32::from_int(1)];
#     let left = client.create(f32::as_bytes(&left));
#     let right = [f32::from_int(1)];
#     let right = client.create(f32::as_bytes(&right));
#     let output = client.empty(core::mem::size_of::<f32>());
#
#     unsafe {
#         kernel_struct_example::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(1, 1, 1),
#             PairLaunch::new(
#                 ArrayArg::from_raw_parts::<f32>(&left, 1, 1),
#                 ArrayArg::from_raw_parts::<f32>(&right, 1, 1),
#             ),
#             ArrayArg::from_raw_parts::<f32>(&output, 1, 1),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_example with runtime {:?} => {:?}",
#         R::name(&client),
#         f32::from_bytes(&client.read_one(output.binding()))
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
# }
```

You can also mutate struct fields if the struct is passed as a mutable reference:

```rust,ignore
# use cubecl::prelude::*;
#
# #[derive(CubeType, CubeLaunch)]
# pub struct Pair<T: CubeLaunch> {
#     pub left: T,
#     pub right: T,
# }
#
#[cube(launch_unchecked)]
pub fn kernel_struct_mut(output: &mut Pair<Array<f32>>) {
    output.left[UNIT_POS] = 42.0;
    output.right[UNIT_POS] = 3.14;
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#
#     let left = [f32::from_int(1)];
#     let left = client.create(f32::as_bytes(&left));
#     let right = [f32::from_int(1)];
#     let right = client.create(f32::as_bytes(&right));
#
#     unsafe {
#         kernel_struct_mut::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(1, 1, 1),
#             PairLaunch::new(
#                 ArrayArg::from_raw_parts::<f32>(&left, 1, 1),
#                 ArrayArg::from_raw_parts::<f32>(&right, 1, 1),
#             ),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_mut with runtime {:?} => ({:?}, {:?})",
#         R::name(&client),
#         f32::from_bytes(&client.read_one(left.binding())),
#         f32::from_bytes(&client.read_one(right.binding())),
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
# }
```

## Comptime fields

You can mark struct fields as comptime, which means their values are known at kernel compilation time and can be used for specialization:

```rust,ignore
# use cubecl::prelude::*;
#
#[derive(CubeType, CubeLaunch)]
pub struct TaggedArray {
    pub array: Array<f32>,
    #[cube(comptime)]
    pub tag: String,
}

#[cube(launch_unchecked)]
pub fn kernel_with_tag(output: &mut TaggedArray) {
    if UNIT_POS == 0 {
        if comptime! {&output.tag == "zero"} {
            output.array[0] = 0.0;
        } else {
            output.array[0] = 1.0;
        }
    }
}
#
# pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
#     let client = R::client(device);
#
#     let output = client.empty(core::mem::size_of::<F>());
#
#     unsafe {
#         kernel_with_tag::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(1, 1, 1),
#             TaggedArrayLaunch::new(
#                 ArrayArg::<R>::from_raw_parts::<F>(&output, 1, 1),
#                 &"not_zero".to_string(),
#             ),
#         )
#     };
#
#     println!(
#         "Executed kernel_with_tag with runtime {:?} => {:?}",
#         R::name(&client),
#         F::from_bytes(&client.read_one(output.binding()))
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime, f32>(&Default::default());
# }
```

## Adding methods to struct
You can implement methods for structs using the `#[cube]` attribute. Please note that the `#[cube]` attribute must be on the impl block. Here's an example:

```rust,ignore
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub struct Pair<T: CubeLaunch> {
    pub left: T,
    pub right: T,
}

#[cube]
impl Pair<Array<f32>> {
    pub fn sum(&self, index: u32) -> f32 {
        self.left[index] + self.right[index]
    }
}

#[cube(launch_unchecked)]
pub fn kernel_struct_example(pair: &Pair<Array<f32>>, output: &mut Array<f32>) {
    output[UNIT_POS] = pair.sum(UNIT_POS);
}
#
# pub fn launch<R: Runtime>(device: &R::Device) {
#     let client = R::client(device);
#
#     let left = [f32::from_int(1)];
#     let left = client.create(f32::as_bytes(&left));
#     let right = [f32::from_int(1)];
#     let right = client.create(f32::as_bytes(&right));
#     let output = client.empty(core::mem::size_of::<f32>());
#
#     unsafe {
#         kernel_struct_example::launch_unchecked::<R>(
#             &client,
#             CubeCount::Static(1, 1, 1),
#             CubeDim::new(1, 1, 1),
#             PairLaunch::new(
#                 ArrayArg::from_raw_parts::<f32>(&left, 1, 1),
#                 ArrayArg::from_raw_parts::<f32>(&right, 1, 1),
#             ),
#             ArrayArg::from_raw_parts::<f32>(&output, 1, 1),
#         )
#     };
#
#     println!(
#         "Executed kernel_struct_example with runtime {:?} => {:?}",
#         R::name(&client),
#         f32::from_bytes(&client.read_one(output.binding()))
#     );
# }
#
# fn main() {
#     launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
# }
```
