# Benchmarking reduction
Now that we have a basic understanding of how to perform a reduction operation, let's benchmark it to see how it performs in terms of speed and efficiency.

## Benchmarking struct
For benchmarking, we will create a struct that holds the necessary information for the benchmark, such as the input shape, device, and client. This struct will be used to run the benchmark tests and configure the benchmarking environment. Please note that the `Runtime` and `Float` traits are used to make the benchmark generic over different CubeCL runtimes and floating-point types. A `PhantomData` is used to indicate that the struct holds a type parameter `F` without actually storing a value of that type, which is useful for generic programming in Rust, for more information see the [Rust documentation](https://doc.rust-lang.org/std/marker/struct.PhantomData.html) and in our case allows us to easily change the type of float used in the benchmark.
```rust,ignore
{{#include src/bin/v3-gpu.rs:1:11}}
```
## Implementing the benchmark trait
To benchmark a CubeCL kernel, it is recommended to implement the `Benchmark` trait that defines the necessary methods for preparing, executing, and synchronizing the benchmark because GPUs are asynchronous and most benchmarking tools will not wait for the GPU to finish executing the kernel before measuring the time it takes to execute it with a sync.
```rust,ignore
/// Benchmark trait.
pub trait Benchmark {
    /// Benchmark input arguments.
    type Input: Clone;
    /// The benchmark output.
    type Output;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Input;

    /// Execute the benchmark and returns the logical output of the task executed.
    ///
    /// It is important to return the output since otherwise deadcode optimization might optimize
    /// away code that should be benchmarked.
    fn execute(&self, input: Self::Input) -> Self::Output;

    /// Name of the benchmark, should be short and it should match the name
    /// defined in the crate Cargo.toml
    fn name(&self) -> String;

    /// Wait for computation to complete.
    fn sync(&self);
}

```

In the `prepare` method, we will create the input data and return a `GpuTensor` that will be used in the `execute` method. The `execute` method will launch the kernel and the `sync` method will wait for the GPU to finish executing the kernel before measuring the time it takes to execute it. Don't forget to add the function that we want to benchmark.
```rust,ignore
{{#include src/bin/v3-gpu.rs:1:56}}
```

## Running the benchmark
Now that we have implemented the `Benchmark` trait, we can run the benchmark using the `Benchmark::run` method. This method will execute the benchmark and return the time it took to complete it.
```rust,ignore
{{#rustdoc_include src/bin/v3-gpu.rs:58:81}}
```

## The Results
When you run the above code, it will execute the reduction benchmark for two different input shapes and print the results to the console. The output will look something like this:
```
wgpu<wgsl>-reduction-[512, 8192]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        240.730ms
  Variance    1.595µs
  Median      240.310ms
  Min         239.974ms
  Max         244.374ms
―――――――――――――――――――――――――
wgpu<wgsl>-reduction-[128, 32768]

―――――――― Result ―――――――――
  Timing      system
  Samples     10
  Mean        241.018ms
  Variance    1.068µs
  Median      240.943ms
  Min         239.734ms
  Max         243.782ms
―――――――――――――――――――――――――
```
As we will see in the next chapter, our time is not that good, but it is expected because we are using a very simple kernel that does not take advantage of the GPU parallelism. In the next chapter, we will see how to optimize our kernel to take advantage of the GPU parallelism and improve the performance of our reduction operation.
