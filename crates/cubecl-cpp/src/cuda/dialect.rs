use std::marker::PhantomData;

use crate::shared::{Dialect, Variable, WmmaCompiler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaDialect<M: WmmaCompiler<Self>> {
    _wmma_compiler: PhantomData<M>,
}

impl<M: WmmaCompiler<Self>> Dialect for CudaDialect<M> {
    type WmmaCompiler = M;

    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_bf16.h>\n")
    }
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_runtime.h>\n")
    }

    fn bfloat16_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("__nv_bfloat16")
    }
    fn bfloat162_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("__nv_bfloat162")
    }

    fn warp_shuffle(input: &Variable<Self>, id: &Variable<Self>) -> String {
        format!("__shfl_sync(-1, {input}, {id})")
    }
    fn warp_shuffle_xor(out: &Variable<Self>) -> String {
        format!("__shfl_xor_sync(-1, {out}, offset)")
    }
    fn warp_shuffle_down(out: &Variable<Self>) -> String {
        format!("__shfl_down_sync(-1, {out}, offset)")
    }
    fn warp_all(out: &Variable<Self>) -> String {
        format!("__all_sync(-1, {out})")
    }
    fn warp_any(out: &Variable<Self>) -> String {
        format!("__any_sync(-1, {out})")
    }
}
