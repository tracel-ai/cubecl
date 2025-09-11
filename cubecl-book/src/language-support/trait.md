# Trait Support

CubeCL supports traits to modularize your kernel code without any overhead. For now most features
are supported. However, methods will create a second half representing the JIT compilation template
(called the "expand" trait). It will have the same name as the base trait with `Expand` added, but
will need to be imported separately wherever the trait is used. This is due to how Rust resolves
trait methods, and how CubeCL works behind the scenes.

```rust
#[cube]
trait MyTrait {
    /// Does not get added to an expand trait.
    fn my_function(x: &Array<f32>) -> f32;
    /// Generates an expand copy in the expand trait.
    fn my_function_2(&self, x: &Array<f32>) -> f32;
}
```

# Expansion and the expand trait

There is an important concept in the way CubeCL works that would normally be hidden, but becomes
relevant with traits. This concept is called **expansion**. When CubeCL processes your trait, it
generates a JIT template that "expands" the Rust code to a GPU-compatible intermediate
representation. This expansion will recreate your code, but with placeholders instead of the real
types and values. The reasons this is relevant to traits are four-fold:

1. Methods need to be implemented on the conceptual, placeholder type. This requires a secondary
   trait that needs to be imported separately. The `#[cube]` macro will automatically handle the
   expanded implementation, as long as your expand type is called (or aliased to) `{name}Expand`.
   This will be the case for any types created with `#[derive(CubeType)]`.
2. Because associated versions of the methods need to be defined on the base trait, they will
   forward their body to the expanded methods. This means your trait _must_ extend some version of
   `CubeType<ExpandType: {name}Expand>`. This is done automatically by the macro, but should be kept
   in mind since it introduces a hidden supertrait.
3. Expanded methods will use an owned `self` by default, meaning they cannot be used with dynamic
   dispatch. If you need dynamic dispatch on the expanded values (as seen in
   [`VirtualTensor`](https://docs.rs/cubecl-std/latest/cubecl_std/tensor/virtual/struct.VirtualTensor.html)
   for example), you can override this behaviour by setting the `self_type` to either
   `#[cube(self_type = "ref")]`, or `#[cube(self_type = "ref_mut")]`. This will also need to be
   added to any implementations, since the macro doesn't know type resolution, so can't tell the
   type of `self`.
4. Expanded traits will not inherit any traits from the base. If you need the expanded trait to
   implement base traits (i.e. `Clone`), use the `expand_base_traits` option on the macro.

# Example

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

[Associated types](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
are also supported. Let's say you want to create a series from a sum.

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

It's actually not the best example of using
[associated types](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types),
but it shows how they are totally supported with CubeCL.
