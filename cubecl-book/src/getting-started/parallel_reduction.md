# Parallel Reduction

Before we dive into the code to implement parallel reduction, let's take a look at the constants that CubeCL provides to help us
write efficient parallel reduction kernels.

## CubeCL Constants

CubeCL is designed around - you guessed it - Cubes! More specifically, it's based on cuboids,
because not all axes are the same size. Since all compute APIs need to map to the hardware, which
are tiles that can be accessed using a 3D representation, our topology can easily be mapped to
concepts from other APIs.

<div align="center">

### CubeCL - Topology

<img src="./cubecl.drawio.svg" width="100%"/>
<br />
</div>
<br />

_A cube is composed of units, so a 3x3x3 cube has 27 units that can be accessed by their positions
along the x, y, and z axes. Similarly, a hyper-cube is composed of cubes, just as a cube is composed
of units. Each cube in the hyper-cube can be accessed by its position relative to the hyper-cube
along the x, y, and z axes. Hence, a hyper-cube of 3x3x3 will have 27 cubes. In this example, the
total number of working units would be 27 x 27 = 729._

### Topology Equivalence

Since all topology variables are constant within the kernel entry point, we chose to use the Rust
constant syntax with capital letters. Often when creating kernels, we don't always care about the
relative position of a unit within a cube along each axis, but often we only care about its position
in general. Therefore, each kind of variable also has its own axis-independent variable, which is
often not present in other languages, except WebGPU with `local_invocation_index`.

<br />

| CubeCL         | CUDA        | WebGPU                 |
| -------------- | ----------- | ---------------------- |
| CUBE_COUNT     | N/A         | N/A                    |
| CUBE_COUNT_X   | gridDim.x   | num_workgroups.x       |
| CUBE_COUNT_Y   | gridDim.y   | num_workgroups.y       |
| CUBE_COUNT_Z   | gridDim.z   | num_workgroups.z       |
| CUBE_POS       | N/A         | N/A                    |
| CUBE_POS_X     | blockIdx.x  | workgroup.x            |
| CUBE_POS_Y     | blockIdx.y  | workgroup.y            |
| CUBE_POS_Z     | blockIdx.z  | workgroup.z            |
| CUBE_DIM       | N/A         | N/A                    |
| CUBE_DIM_X     | blockDim.x  | workgroup_size.x       |
| CUBE_DIM_Y     | blockDim.y  | workgroup_size.y       |
| CUBE_DIM_Z     | blockDim.z  | workgroup_size.z       |
| UNIT_POS       | N/A         | local_invocation_index |
| UNIT_POS_X     | threadIdx.x | local_invocation_id.x  |
| UNIT_POS_Y     | threadIdx.y | local_invocation_id.y  |
| UNIT_POS_Z     | threadIdx.z | local_invocation_id.z  |
| PLANE_DIM      | warpSize    | subgroup_size          |
| ABSOLUTE_POS   | N/A         | N/A                    |
| ABSOLUTE_POS_X | N/A         | global_id.x            |
| ABSOLUTE_POS_Y | N/A         | global_id.y            |
| ABSOLUTE_POS_Z | N/A         | global_id.z            |

</details>

## Parallel Reduction Example
Remembering the previous example, we will now implement a parallel reduction using CubeCL. The goal is to reduce a 2D matrix into a 1D vector by summing the elements of each row. Where can we parallelize this operation? We can parallelize the reduction of each row, allowing each thread to compute the sum of a row independently. We need to change the launch parameters to set the CubeDim to launch multiple invocation in parallel and we can just remove the outer loop and use the `UNIT_POS_X` to access the rows of the input tensor. Please note that it is important to worry about data races when parallelizing operations, so we need to ensure that each invocation writes to a different position in the output tensor, because each invocation run in parallel.
```rust,ignore
{{#rustdoc_include src/bin/v4-gpu.rs:31:54}}
```

## The Results
Now that we have improved parallelism, the kernel is up to 70x faster for the first shape and up to 25x faster for the second shape. Why is the first shape faster than the second? Even though the two shape have the same number of elements, the first shape has more rows than the second shape, which means that more invocations can be used to compute the reduction in parallel. The second shape has fewer rows, so fewer invocations are available to compute the reduction in parallel.
```
wgpu<wgsl>-reduction-[512, 8192]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        3.369ms
  Variance    48.000ns
  Median      3.321ms
  Min         3.177ms
  Max         4.011ms
―――――――――――――――――――――――――
wgpu<wgsl>-reduction-[128, 32768]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        8.713ms
  Variance    5.069µs
  Median      8.507ms
  Min         5.963ms
  Max         12.301ms
―――――――――――――――――――――――――
```
