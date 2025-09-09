fn main() {
    let version = cudarc::driver::sys::CUDA_VERSION;

    if version >= 12080 {
        println!("cargo:rustc-cfg=feature=\"cuda-12080\"");
    }
}
