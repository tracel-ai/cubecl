pub mod reinterpret_list;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_cubecl_std {
            use super::*;

            cubecl_std::testgen_reinterpret_list!();
        }
    };
}
