# Comptime

CubeCL isn't just a new compute language: though it feels like you are writing GPU kernels, you are,
in fact, writing compiler plugins that you can fully customize! Comptime is a way to modify the
compiler IR at runtime when compiling a kernel for the first time.

This enables a lot of optimizations and flexibility without having to write many separate variants
of the same kernels to ensure maximal performance.

## Loop Unrolling

You can easily unroll loops in CubeCL using the `unroll` attribute on top of a for loop.

```rust
#[cube(launch)]
fn sum<F: Float>(input: &Array<F>, output: &mut Array<F>, #[comptime] end: Option<u32>) {
    let unroll = end.is_some();
    let end = end.unwrap_or_else(|| input.len());
    let mut sum = F::new(0.0);

    #[unroll(unroll)]
    for i in 0..end {
        sum += input[i];
    }

    output[ABSOLUTE_POS] = sum;
}
```

Note that if you provide a variable `end` that can't be determined at compile time, a panic will
arise when trying to execute that kernel.

## Feature Specialization

You could also achieve the sum using plane operations. We will write a kernel that uses that
instruction when available based on a comptime feature flag. When it isn't available, it will fall
back on the previous implementation, essentially making it portable.

```rust
#[cube(launch)]
fn sum_plane<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] plane: bool,
    #[comptime] end: Option<u32>,
) {
    if plane {
        output[UNIT_POS] = plane_sum(input[UNIT_POS]);
    } else {
        sum_basic(input, output, end);
    }
}
```

Note that no branching will actually occur on the GPU, since three different kernels can be
generated from the last code snippet. You can also use the
[trait system](../language-support/trait.md) to achieve a similar behavior.
