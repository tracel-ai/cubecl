use std::marker::PhantomData;

use crate::shared::{Dialect, Variable, WmmaCompiler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HipDialect<M: WmmaCompiler<Self>> {
    _wmma_compiler: PhantomData<M>,
}

impl<M: WmmaCompiler<Self>> Dialect for HipDialect<M> {
    type WmmaCompiler = M;

    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip/hip_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "hip_bf16.h" triggers redefinition errors during compilation
        f.write_str("#include <hip/hip_bfloat16.h>\n")
    }
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <hip/hip_runtime.h>\n")
    }

    fn bfloat16_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("hip_bfloat16")
    }
    fn bfloat162_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "hip_bfloat16.h" has no "hip_bfloat162" type
        f.write_str("hip_bfloat16")
    }

    fn warp_shuffle(input: &Variable<Self>, id: &Variable<Self>) -> String {
        format!("__shfl({input}, {id})")
    }
    fn warp_shuffle_xor(out: &Variable<Self>) -> String {
        format!("__shfl_xor({out}, offset)")
    }
    fn warp_shuffle_down(out: &Variable<Self>) -> String {
        format!("__shfl_down({out}, offset)")
    }
    fn warp_all(out: &Variable<Self>) -> String {
        format!("__all({out})")
    }
    fn warp_any(out: &Variable<Self>) -> String {
        format!("__any({out})")
    }
}
