use cudarc::driver::sys::CUDA_VERSION;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_12050)");
    println!("cargo::rustc-check-cfg=cfg(cuda_12080)");

    if CUDA_VERSION >= 12050 {
        println!("cargo:rustc-cfg=cuda_12050");
    }
    if CUDA_VERSION >= 12080 {
        println!("cargo:rustc-cfg=cuda_12080");
    }
}
