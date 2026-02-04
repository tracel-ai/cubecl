# CubeCL Metal Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) Metal runtime.

## Platform support

| Platform   | GPU |
| :--------- | :-: |
| macOS 12+  | Yes |
| iOS 15+    | Yes |

## Limitations

- bf16 plane operations (simd_shuffle) not supported by Metal hardware
- f64 not supported by Metal
