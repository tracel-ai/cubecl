use cubecl_cpp::{metal::MslDialect, shared::CppCompiler};

/// Metal shader compiler using the MSL dialect from cubecl-cpp
pub type MetalCompiler = CppCompiler<MslDialect>;
