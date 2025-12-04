mod problem_size;

#[macro_export]
macro_rules! testgen_convolution_problem {
    ($algorithm: ty, $precision: ty, $selection_builder: expr) => {
        mod _problem_generated {
            use super::*;
            use cubecl_matmul::components::MatmulSelection;

            pub fn get_selection() -> MatmulSelection {
                $selection_builder.build()
            }
        }

        $crate::testgen_convolution_problem_size!(
            $algorithm,
            $precision,
            _problem_generated::get_selection()
        );
    };
}
