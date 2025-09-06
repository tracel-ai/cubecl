pub mod simple;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_scan {
    () => {
        mod test_scan {
            use super::*;

            cubecl_scan::testgen_scan_simple!();
        }
    };
}
