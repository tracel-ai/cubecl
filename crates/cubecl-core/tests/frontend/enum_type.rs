use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[allow(dead_code)]
mod test_compilation {
    use super::*;

    #[derive(CubeType)]
    enum VariantNoInput {
        Add,
        Min,
    }

    #[derive(CubeType)]
    enum SingleVariant {
        Add(u32),
    }

    #[derive(CubeType)]
    enum MultipleVariants {
        Add(u32),
        Min(u32, u32),
    }

    #[derive(CubeType)]
    enum MultipleVariantsNamed {
        Add(u32),
        Min(u32, u32),
        Mul { lhs: u32, rhs: u32 },
    }
}
