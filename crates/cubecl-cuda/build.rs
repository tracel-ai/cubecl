fn main() {
    if let Ok(version) = std::env::var("CUDARC_CUDA_VERSION") {
        println!("cargo:rustc-cfg=feature=\"cuda-{version}\"");
    }
}
