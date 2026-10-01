use cfg_aliases::cfg_aliases;

fn main() {
    // A counter is compiled where its feature is on and its platform keeps it.
    cfg_aliases! {
        amdgpu_busy_percent: { all(feature = "amdgpu-busy-percent", target_os = "linux") },
        intel_idle_residency: { all(feature = "intel-idle-residency", target_os = "linux") },
        nvml: { all(feature = "nvml", target_os = "linux") },
        gpu_engine_counters: { all(feature = "gpu-engine-counters", target_os = "windows") },
        io_accelerator: { all(feature = "io-accelerator", target_os = "macos") },
        card_counter: { any(amdgpu_busy_percent, intel_idle_residency, nvml, gpu_engine_counters, io_accelerator) },
    }
}
