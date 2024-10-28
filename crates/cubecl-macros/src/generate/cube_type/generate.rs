use proc_macro2::TokenStream;

use crate::parse::cube_type::CubeType;

impl CubeType {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        match self {
            Self::Enum(data) => data.generate(with_launch),
            Self::Struct(data) => data.generate(with_launch),
        }
    }
}
