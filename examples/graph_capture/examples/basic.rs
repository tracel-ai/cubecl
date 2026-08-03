fn main() -> Result<(), Box<dyn core::error::Error>> {
    #[cfg(feature = "cuda")]
    graph_capture::basic::<cubecl::cuda::CudaRuntime>(&Default::default())?;
    #[cfg(not(feature = "cuda"))]
    println!("enable --features cuda");
    Ok(())
}
