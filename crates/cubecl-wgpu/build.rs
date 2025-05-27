use cfg_aliases::cfg_aliases;
use std::env;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        exclusive_memory_only: { any(feature = "exclusive-memory-only", target_family = "wasm") },
        apple_silicon: { all(target_os = "macos", target_arch = "aarch64") },
    }

    // Check if we are on macOS
    // Errors out on MacOS when the "spirv" feature is enabled and the Vulkan SDK is not installed.
    // To install Vulkan SDK visit https://vulkan.lunarg.com/sdk/home#mac
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SPIRV");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    let is_macos = env::var("CARGO_CFG_TARGET_OS")
        .map(|os| os == "macos")
        .unwrap_or(false);
    if is_macos {
        println!("cargo:rustc-cfg=feature=\"vulkan-portability\"");
    }
    let spirv_feature_enabled = env::var("CARGO_FEATURE_SPIRV").is_ok();
    let vulkan_sdk_installed = env::var("VULKAN_SDK").is_ok();
    if is_macos && spirv_feature_enabled && !vulkan_sdk_installed {
        let msg = "The Vulkan SDK is required on macOS when the 'spirv' feature is enabled. Install the Vulkan SDK and make sure the VULKAN_SDK environment variable is set. Visit https://vulkan.lunarg.com/sdk/home#mac to learn how to install it.";
        println!("cargo:warning={msg}");
        panic!("{msg}");
    }
}
