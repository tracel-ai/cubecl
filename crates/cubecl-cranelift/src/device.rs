use std::sync::Arc;

use cranelift_codegen::isa::{self, TargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use target_lexicon::{Architecture, Triple};

#[derive(new, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CraneLiftDevice {
    pub triple: Triple,
    /// The index of the device. *Almost* always 0.
    pub index: usize,
    /// The maximum number of bytes in a SIMD register.
    max_line_size: Option<u8>,
}

impl Default for CraneLiftDevice {
    fn default() -> Self {
        let current_triple: Triple = Triple::host();

        let isa = match isa::lookup(current_triple.clone()) {
            Err(_) => panic!("No Cranelift target found for host"),
            Ok(isa_builder) => isa_builder
                .finish(settings::Flags::new(settings::builder()))
                .unwrap(),
        };
        let flags = isa.isa_flags().iter();
        let max_lane_size = match current_triple.architecture {
            // RISC-V only supports SIMD with the V extension.
            Architecture::Riscv64(_) => {
                if flags
                    .find(|f| f.name == "has_v")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false)
                {
                    Some(16) //placeholder, not familiar with RISC-V
                } else {
                    None
                }
            }
            _ => (true, 8),
            Architecture::X86_64 => {
                if flags
                    .find(|f| f.name == "use_avx512f")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false)
                {
                    Some(64)
                } else if flags
                    .find(|f| f.name == "use_avx2")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false)
                {
                    //fall back to avx2
                    Some(32)
                } else if flags
                    .find(|f| f.name == "use_avx")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false)
                {
                    //fall back to avx
                    Some(16)
                } else {
                    None
                }
            }
            Architecture::Aarch64 => Some(16),
        };

        Self {
            triple: current_triple,
            index: 0,
            max_line_size,
        }
    }
}

impl CraneLiftDevice {
    pub(crate) fn supported_line_sizes(&self) -> &'static [u8] {
        match self.max_line_size {
            Some(16) => &[16, 8, 4, 2, 1],
            Some(32) => &[32, 16, 8, 4, 2, 1],
            Some(64) => &[64, 32, 16, 8, 4, 2, 1],
            _ => &[1],
        }
    }
}
