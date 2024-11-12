use crate::{hip::{arch::AMDArchitecture, HipDialect}, shared::{
    Fragment, FragmentIdent, FragmentLayout, SupportedWmmaCombinations, Variable, WmmaCompiler, WmmaInstruction
}, Dialect};
use cubecl_core::ir::{self as gpu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsicCompiler {}

impl WmmaCompiler<HipDialect<Self>> for WmmaIntrinsicCompiler {
    type Architecture = AMDArchitecture;

    fn includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn deftypes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("typedef _Float16 half16 __attribute__((ext_vector_type(16)));\n")?;
        f.write_str("typedef float float8 __attribute__((ext_vector_type(8)));\n")
    }

    fn compile_fragment_ident(
        _ident: &FragmentIdent<HipDialect<Self>>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // nothing to do
        Ok(())
    }

    fn compile_fragment_layout(
        _layout: &FragmentLayout<HipDialect<Self>>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // nothing to do
        Ok(())
    }

    fn compile_fragment(
        fragment: &Fragment<HipDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match fragment.ident {
            FragmentIdent::A | FragmentIdent::B => write!(f, "half16"),
            FragmentIdent::Accumulator => write!(f, "float8"),
            FragmentIdent::_Dialect(_) => Ok(()),
        }
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<HipDialect<Self>>,
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

    fn supported_wmma_combinations(arch: &Self::Architecture) -> SupportedWmmaCombinations {
        // Reference: https://gpuopen.com/learn/wmma_on_rdna3/
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.is_wmma_capable() {
            // Types fully supported.
            let types = vec![
                (
                    gpu::Elem::Float(gpu::FloatKind::F16), // i
                    gpu::Elem::Float(gpu::FloatKind::F16), // o
                    gpu::Elem::Float(gpu::FloatKind::F16), // c
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
            ];
            let combinations: SupportedWmmaCombinations = types
                .into_iter()
                //                           m   n   k
                .map(|(i, o, c)| {(i, o, c, vec![(16, 16, 16)])})
                .collect();
            result.extend(combinations);
        }
        result
    }
}
