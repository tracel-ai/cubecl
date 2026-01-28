# CubeCL Metal Backend

Native Metal backend for CubeCL, providing direct access to Apple's Metal API without going through WGPU.

## Features

- **BF16 support** - Native bfloat16 operations (Metal 3.1+)
- **Vec8 vectorization** - 2x memory bandwidth vs WGPU
- **Simdgroup operations** - Direct access to Metal's warp primitives
- **Lower overhead** - No abstraction layer between CubeCL and Metal
- **Better debugging** - Direct MSL source mapping

## Requirements

- macOS 12.0+ or iOS 15.0+
- Metal 3.0+
- For BF16: Metal 3.1+ (macOS 13.0+, iOS 16.0+)

## Usage

```rust
use cubecl_metal::{MetalRuntime, MetalDevice};

// Get default Metal device
let device = MetalDevice::default();

// Create client
let client = MetalRuntime::client(&device);

// Use with CubeCL as normal
```

## Architecture

This backend uses `objc2-metal` for Metal bindings and leverages the existing `cubecl-cpp` MSL compiler.

**Compilation flow:**
```
CubeCL IR → MslDialect (cubecl-cpp) → MSL source → MTLLibrary → MTLComputePipelineState
```

## Status

🚧 **Work in Progress** - This backend is under active development.

### Implemented
- [ ] Device enumeration
- [ ] Basic buffer allocation
- [ ] Kernel compilation
- [ ] Kernel launch
- [ ] Memory management

### Planned
- [ ] Buffer pooling
- [ ] Residency set management
- [ ] BF16 support
- [ ] Simdgroup matrix operations
- [ ] Multi-stream execution
