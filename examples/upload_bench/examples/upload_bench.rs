fn main() {
    #[cfg(feature = "cuda")]
    upload_bench::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(not(feature = "cuda"))]
    eprintln!("Build with --features cuda to run this benchmark.");
}
