# Metal runtime

Native Metal backend for CubeCL, providing direct access to Apple's Metal API.

## Features

- **BF16 support** - Native bfloat16 operations
- **Full type support** - f16, bf16, f32, i8, i16, i32, i64, u8, u16, u32, u64
- **Simdgroup operations** - Direct access to Metal's SIMD primitives

## Requirements

- macOS 12.0+ or iOS 15.0+

## Usage

```rust
use cubecl_metal::{MetalRuntime, MetalDevice};

let device = MetalDevice::default();
let client = MetalRuntime::client(&device);
```

## Known Limitations

- bf16 plane operations (simd_shuffle) are not supported by Metal hardware
- f64 is not supported by Metal
