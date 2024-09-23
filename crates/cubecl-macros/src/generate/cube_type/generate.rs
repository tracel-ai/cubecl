use proc_macro2::TokenStream;

use crate::parse::cube_type::CubeType;

impl CubeType {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        match self {
            CubeType::Enum(data) => data.generate(with_launch),
            CubeType::Struct(data) => data.generate(with_launch),
        }
    }
}
