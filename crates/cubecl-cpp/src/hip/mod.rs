use crate::shared::{Dialect, IndexedVariable, Variable};
use std::marker::PhantomData;

use crate::shared::{
    wmma_api_base, Dialect, Fragment, FragmentIdent, FragmentLayout, Variable, WmmaCompiler,
    WmmaInstruction,
};

const ROCWMMA_NAMESPACE: &str = "rocwmma";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsic {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hip<M: WmmaCompiler<Self>> {
    _wmma_compiler: PhantomData<M>,
}

impl<M: WmmaCompiler<Self>> Dialect for Hip<M> {
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

    fn warp_shuffle(input: &IndexedVariable<Self>, id: &Variable<Self>) -> String {
        format!("__shfl({input}, {id})")
    }
    fn warp_shuffle_xor(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_xor({out}, offset)")
    }
    fn warp_shuffle_down(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_down({out}, offset)")
    }
    fn warp_all(out: &IndexedVariable<Self>) -> String {
        format!("__all({out})")
    }
    fn warp_any(out: &IndexedVariable<Self>) -> String {
        format!("__any({out})")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaApiHip {}

impl WmmaCompiler<Hip<Self>> for WmmaApiHip {
    fn includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <rocwmma/rocwmma.hpp>\n")
    }

    fn deftypes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_ident(
        ident: &FragmentIdent<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(ROCWMMA_NAMESPACE, ident, f)
    }

    fn compile_fragment_layout(
        layout: &FragmentLayout<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(ROCWMMA_NAMESPACE, layout, f)
    }

    fn compile_fragment(
        fragment: &Fragment<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(ROCWMMA_NAMESPACE, fragment, f)
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(ROCWMMA_NAMESPACE, instruction, f)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsicHip {}

impl WmmaCompiler<Hip<Self>> for WmmaIntrinsicHip {
    fn includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn deftypes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("typedef _Float16 half16 __attribute__((ext_vector_type(16)));\n")?;
        f.write_str("typedef float float8 __attribute__((ext_vector_type(8)));\n")
    }

    fn compile_fragment_ident(
        _ident: &FragmentIdent<Hip<Self>>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // nothing to do
        Ok(())
    }

    fn compile_fragment_layout(
        _layout: &FragmentLayout<Hip<Self>>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // nothing to do
        Ok(())
    }

    fn compile_fragment(
        fragment: &Fragment<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match fragment.ident {
            FragmentIdent::A | FragmentIdent::B => write!(f, "half16"),
            FragmentIdent::Accumulator => write!(f, "float8"),
            FragmentIdent::_Dialect(_) => Ok(()),
        }
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<Hip<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value } => {
                let fill_with_zeros =
                    matches!(value, Variable::ConstantScalar(number, _) if number.is_zero());
                if fill_with_zeros {
                    writeln!(f, "{frag} = {{}};")
                } else {
                    write!(
                        f,
                        "
for (int i = 0; i < 8; ++i) {{
  {frag}[i] = {value};
}}
"
                    )
                }
            }
            WmmaInstruction::Load { frag, value, .. } => {
                write!(
                    f,
                    "
for (int i = 0; i < 16; ++i) {{
    {frag}[i] = {value}[i + (threadIdx.x % 16) * 16];
}}
"
                )
            }
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
            } => writeln!(
                f,
                "{frag_d} = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32({frag_a}, {frag_b}, {frag_c});"
            ),
            WmmaInstruction::Store {
                output,
                frag,
                ..
            } => {
                write!(
                    f,
                    "
for (int i = 0; i < 8; ++i) {{
  const int row = (i * 2 + threadIdx.x / 16);
  {output}[row * 16 + (threadIdx.x % 16)] = {frag}[i];
}}
"
                )
            },
        }
    }
}
