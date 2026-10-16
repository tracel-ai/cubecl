# Math Optimizations

## Fast Math Options

Floating point operations have a lot of restrictions required to follow the specification,
especially around special values (`Inf`/`NaN`) and signed zero that are rarely used. CubeCL allows
marking functions with loosened restrictions to accelerate math operations, while trading off some
special handling or precision.

The effect is backend-dependent, but uses a unified API of flags specifying acceptable
optimizations. These `FastMath` flags can be applied per-function, so they can be applied only to
performance-critical sections of the code.

**Example:**

```rust
/// Only the inverse square root has reduced precision/no special handling. Everything else is full
/// precision.
#[cube(launch_unchecked)]
fn run_on_array<F: Float>(input: &[F], alpha: F, epsilon: F, output: &mut [F]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = alpha * fast_rsqrt::<F>(input[ABSOLUTE_POS]) + epsilon;
    }
}

#[cube(fast_math = FastMath::all())]
fn fast_rsqrt<F: Float>(x: F) -> F {
    F::inverse_sqrt(x)
}
```

### Strict Arithmetic

`FastMath::none()` allows nothing: every operation rounds on its own, so a product is never fused
into the add that consumes it, and nothing is assumed about `Inf`, `NaN` or the sign of zero. Use it
where the arithmetic itself is the algorithm, as in compensated summation.

```rust
/// The error term of `a + b`, exact only if neither operation is fused or reassociated.
#[cube(fast_math = FastMath::none())]
fn two_sum<F: Float>(a: F, b: F) -> (F, F) {
    let sum = a + b;
    let b_virtual = sum - a;
    let error = (a - (sum - b_virtual)) + (b - b_virtual);
    (sum, error)
}
```

A function that says nothing gets each backend's default, which usually allows contraction: CubeCL
fuses a product into the add that consumes it, and so may the driver.

To take optimizations away from every kernel, including those in other crates, see
`disabled_fast_math` in the [configuration](./config.md).

### Backend Implementation

#### WGPU with Vulkan Compiler

Forbidden contraction is `NoContraction` on each operation. Other permissions become
`FPFastMathMode` where the device supports `SPV_KHR_float_controls2`. Vulkan may assume there is no
`Inf`, `NaN` or signed zero whatever the flags say, so strict kernels on Vulkan are guaranteed only
the rounding of each operation.

#### CPU, and CUDA/HIP through LLVM

The flags map to LLVM's fast-math flags on each operation.

#### CUDA/HIP C++

Forbidden contraction prints add, subtract, multiply and divide as `__fadd_rn` and its siblings on
CUDA, and as helpers compiled under `#pragma clang fp contract(off)` on HIP. Half precision runs in
`f32` for these, which rounds to the same result.

#### Metal

Metal compiles with `MathMode::Safe`, so it never contracts on its own.

#### WGSL

WGSL has no way to forbid contraction for one operation. CubeCL does not fuse a strict kernel's
operations, but the shader compiler may.

## FastDivmod

A very common operation, especially on GPUs, is applying integer division and modulo with a uniform,
but not constant, divisor (i.e. width). For example:

```rust
#[cube(launch)]
pub fn some_2d_kernel<F: Float>(output: &mut [F], width: u32) {
    let y = ABSOLUTE_POS / width;
    let x = ABSOLUTE_POS % width;
    //...
}

// ...
some_2d_kernel::launch::<F, R>(
    &client,
    // ...,
    matrix.width as u32,
);
```

However, integer division is quite slow, so this might have an impact on runtime. To mitigate the
cost you can use `FastDivmod` to pre-calculate the factors for division using 64-bit
[Barret Reduction](https://en.wikipedia.org/wiki/Barrett_reduction), and pass those instead of the
divisor.  
This is faster even if you only use division or modulo, and _much_ faster if you use both.

**Example:**

```rust
#[cube(launch)]
pub fn some_2d_kernel<F: Float>(output: &mut [F], width: FastDivmod) {
    let (y, x) = width.div_mod(ABSOLUTE_POS);
    //...
}

some_2d_kernel::launch::<F, R>(
    &client,
    // ...,
    FastDivmodArgs::new(&client, matrix.width as u32),
);
```

### Backend Support

This is implemented using efficient extended multiplication on CUDA (`__umulhi`) and Vulkan
(`OpUMulExtended`), and using manual casts and shifts on targets that support `u64`. Targets without
either (possibly `WebGPU`) fall back to normal division.
