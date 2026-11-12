# CubeCL OpenCL Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) OpenCL runtime: kernels in OpenCL C 3.0, built by the
device's driver.

The OpenCL library (the ICD loader) is loaded at run time, so the crate builds and links on any
platform; a machine without one simply has no OpenCL device.

## Device support

A device takes part when it has OpenCL C 3.0 and coarse-grained shared virtual memory. Beyond
that:

| Feature                | Requires                                                                 |
| :--------------------- | :----------------------------------------------------------------------- |
| Buffers                | Intel's unified shared memory (device allocations), else coarse-grained SVM |
| Planes                 | `cl_intel_required_subgroup_size` and the `cl_khr_subgroup_*` extensions |
| Cooperative matrices   | Intel's DPAS (`cl_intel_subgroup_matrix_multiply_accumulate`)            |
| Atomics                | device-scope atomics; float add / min / max from `cl_ext_float_atomics` |
| `f16`, 64-bit integers | `cl_khr_fp16`, `__opencl_c_int64`                                        |

Validated on Intel GPUs with Intel's compute runtime (NEO).

## Limitations

- No `f64`, no tensor maps, no indirect dispatch (a dynamic cube count is read back first).
- Apple's OpenCL (1.2) is not supported.
