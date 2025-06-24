#[macro_export]
macro_rules! testgen_matmul_hypercube {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection_builder: expr) => {
        // TODO test hypercube once PR #746 merged

        $crate::testgen_matmul_problem!($kind, $algorithm, $precision, $selection_builder);
    };
}
