pub mod reinterpret_slice;
pub mod tensor;
pub mod trigonometry;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_cubecl_std {
            use super::*;
            use half::{bf16, f16};

            cubecl_std::testgen_reinterpret_slice!();
            cubecl_std::testgen_trigonometry!();
        }
    };
}
