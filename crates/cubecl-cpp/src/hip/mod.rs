use crate::shared::Dialect;

#[derive(Clone, Debug, Default)]
pub struct Hip;

impl Dialect for Hip {
    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip_bfp16.h>\n")
    }
    fn include_wmma(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip/hip_runtime.h>\n")
    }
}
