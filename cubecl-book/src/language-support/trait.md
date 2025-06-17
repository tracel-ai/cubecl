# Trait Support

CubeCL partially supports traits to modularize your kernel code without any overhead. For now most
features are supported except stateful functions.

```rust
#[cube]
trait MyTrait {
    /// Supported
    fn my_function(x: &Array<f32>) -> f32;
    /// Unsupported
    fn my_function_2(&self, x: &Array<f32>) -> f32;
}
```

The trait system allows you to do specialization quite easily. Let's take the same example as in the
[comptime section](../core-features/comptime.md).

First you can define your trait. Note that if you use your trait from the launch function, you will
need to add `'static + Send + Sync`.

```rust
#[cube]
trait SumKind: 'static + Send + Sync {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] end: Option<u32>) -> F;
}
```

Then we can define some implementations:

```rust
struct SumBasic;
struct SumPlane;

#[cube]
impl SumKind for SumBasic {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] end: Option<u32>) -> F {
        let unroll = end.is_some();
        let end = end.unwrap_or_else(|| input.len());

        let mut sum = F::new(0.0);

        #[unroll(unroll)]
        for i in 0..end {
            sum += input[i];
        }

        sum
    }
}

#[cube]
impl SumKind for SumPlane {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] _end: Option<u32>) -> F {
        plane_sum(input[UNIT_POS])
    }
}
```

[Associated types](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types) are also supported. Let's say you want to create a series from a sum.

```rust
#[cube]
trait CreateSeries: 'static + Send + Sync {
    type SumKind: SumKind;

    fn execute<F: Float>(input: &Slice<F>, #[comptime] end: Option<u32>) -> F;
}
```

You may want to define what kind of series you want to create using an implementation.

```rust
struct SumThenMul<K: SumKind> {
    _p: PhantomData<K>,
}

#[cube]
impl<K: SumKind> CreateSeries for SumThenMul<K> {
    type SumKind = K;

    fn execute<F: Float>(input: &Slice<F>, #[comptime] end: Option<u32>) -> F {
        let val = Self::SumKind::sum(input, end);
        val * input[UNIT_POS]
    }
}
```

It's actually not the best example of using [associated types](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types),
but it shows how they are totally supported with CubeCL.
