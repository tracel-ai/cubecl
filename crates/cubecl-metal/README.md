# CubeCL Metal Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) Metal runtime.

## Platform support

| Platform   | GPU |
| :--------- | :-: |
| macOS 13+  | Yes |
| iOS 16+    | Yes |

## Limitations

- Metal 3+ required (atomic float and simdgroup matrix support)
- bf16 plane operations (simd_shuffle) not supported by Metal hardware
- f64 not supported by Metal
