pub mod reinterpret_slice;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_cubecl_std {
            use super::*;

            cubecl_std::testgen_reinterpret_slice!();
        }
    };
}
