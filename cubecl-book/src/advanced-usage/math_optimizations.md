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
fn run_on_array<F: Float>(input: &Array<F>, alpha: F, epsilon: F, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = alpha * fast_rsqrt::<F>(input[ABSOLUTE_POS]) + epsilon;
    }
}

#[cube(fast_math = FastMath::all())]
fn fast_rsqrt<F: Float>(x: F) -> F {
    F::inverse_sqrt(x)
}
```

### Backend Implementation

#### WGPU with Vulkan Compiler

Vulkan supports each flag as a modifier for all floating point operations. The compiler applies all
enabled flags, but the implementation is driver-specific.

#### CUDA/HIP

These targets only expose specific intrinsics. These intrinsics are used when all their required
flags are present. Only `f32` is supported for these intrinsics, other float types are not affected
by math flags on CUDA/HIP. Note that some of these are guesswork, because CUDA lacks documentation
on special value handling.

| CubeCL Function   | Intrinsic           | Required Flags                                                  |
| ----------------- | ------------------- | --------------------------------------------------------------- |
| `a / b`           | `__fdividef(a, b)`  | `AllowReciprocal \| ReducedPrecision \| UnsignedZero \| NotInf` |
| `exp(a)`          | `__expf(a)`         | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `log(a)`          | `__logf(a)`         | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `sin(a)`          | `__sinf(a)`         | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `cos(a)`          | `__cosf(a)`         | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `tanh(a)`         | `__tanhf(a)`        | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `powf(a)`         | `__powf(a)`         | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `sqrt(a)`         | `__fsqrt_rn(a)`     | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `inverse_sqrt(a)` | `__frsqrt_rn(a)`    | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `recip(a)`        | `__frcp_rn(a)`      | `AllowReciprocal \| ReducedPrecision \| UnsignedZero \| NotInf` |
| `normalize(a)`    | n/a (`__frsqrt_rn`) | `ReducedPrecision \| NotNaN \| NotInf`                          |
| `magnitude(a)`    | n/a (`__fsqrt_rn`)  | `ReducedPrecision \| NotNaN \| NotInf`                          |

#### Other Backends

Other backends currently don't support any of these optimizations.

## FastDivmod

A very common operation, especially on GPUs, is applying integer division and modulo with a uniform,
but not constant, divisor (i.e. width). For example:

```rust
#[cube(launch)]
pub fn some_2d_kernel<F: Float>(output: &mut Array<F>, width: u32) {
    let y = ABSOLUTE_POS / width;
    let x = ABSOLUTE_POS % width;
    //...
}

// ...
some_2d_kernel::launch::<F, R>(
    &client,
    // ...,
    ScalarArg::new(matrix.width as u32),
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
pub fn some_2d_kernel<F: Float>(output: &mut Array<F>, width: FastDivmod) {
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
