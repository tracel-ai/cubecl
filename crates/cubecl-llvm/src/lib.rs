#[macro_use]
extern crate derive_new;

extern crate alloc;

#[cfg(feature = "amdgpu")]
pub mod amdgpu;
pub mod cpu;
#[cfg(feature = "nvptx")]
pub mod nvptx;
pub(crate) mod prelude;
pub mod shared;
pub mod target;

pub use cpu::jit::data::{PlironData, SharedData};
pub use cpu::jit::engine::{KernelRequirements, PlironEngine};
pub use cpu::shared_memory::SharedMemories;
#[cfg(feature = "amdgpu")]
pub use shared::AmdGpuModule;
#[cfg(feature = "nvptx")]
pub use shared::NvptxModule;
pub use shared::{PlironArtifact, PlironCompiler, PlironOptions};
pub use target::LlvmTarget;

/// The parts of `build.rs` that the build script cannot test itself.
#[cfg(test)]
mod build_script {
    include!("../build/frame_pointers.rs");

    #[test]
    fn the_last_force_frame_pointers_flag_wins() {
        let cases = [
            ("", false),
            ("-Ctarget-cpu=native", false),
            ("-Cforce-frame-pointers", true),
            ("-Cforce-frame-pointers=yes", true),
            ("-C\x1fforce-frame-pointers=always", true),
            ("-Cforce-frame-pointers=non-leaf", false),
            (
                "-Cforce-frame-pointers=yes\x1f-C\x1fforce-frame-pointers=no",
                false,
            ),
            (
                "-Cforce-frame-pointers=off\x1f-Copt-level=3\x1f-Cforce-frame-pointers=on",
                true,
            ),
        ];
        for (flags, forced) in cases {
            assert_eq!(forces_frame_pointers(flags), forced, "{flags:?}");
        }
    }
}
