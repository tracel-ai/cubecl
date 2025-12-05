use cfg_aliases::cfg_aliases;

fn main() {
    // Setup cfg aliases
    cfg_aliases! {
        test_runtime_default: {
            all(
                feature = "test-runtime",
                not(any(feature="cpu", feature = "cuda", feature = "hip", feature="wgpu", feature = "wgpu-msl", feature = "wgpu-spirv"))
            )
        },
    }
}
