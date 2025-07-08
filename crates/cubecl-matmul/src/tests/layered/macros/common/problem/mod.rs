mod layouts;
mod problem_size;

#[macro_export]
macro_rules! testgen_matmul_problem {
    ($kind: ident, $algorithm: ty, $precision: ty, $selection_builder: expr) => {
        mod _problem_generated {
            use super::*;

            pub fn get_selection() -> MatmulSelection {
                $selection_builder.build()
            }
        }

        $crate::testgen_matmul_layouts!(
            $kind,
            $algorithm,
            $precision,
            _problem_generated::get_selection()
        );
    };
}
