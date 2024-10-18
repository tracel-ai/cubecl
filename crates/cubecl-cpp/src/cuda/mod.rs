use crate::shared::Dialect;

#[derive(Clone, Debug, Default)]
pub struct Cuda;

impl Dialect for Cuda {
    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_bfp16.h>\n")
    }
    fn include_wmma(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }
    fn namespace_wmma(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("using namespace nvcuda;\n")
    }
}
