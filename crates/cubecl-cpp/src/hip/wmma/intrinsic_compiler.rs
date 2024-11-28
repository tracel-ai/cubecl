use crate::{
    hip::{arch::AMDArchitecture, HipDialect},
    shared::{
        Architecture, Component, Elem, Fragment, FragmentIdent, FragmentLayout,
        SupportedWmmaCombinations, Variable, WmmaCompiler, WmmaInstruction,
    },
};
use cubecl_core::ir::{self as gpu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsicCompiler {}

impl WmmaCompiler<HipDialect<Self>> for WmmaIntrinsicCompiler {
    type Architecture = AMDArchitecture;

    fn includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn deftypes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("typedef _Float16 half8 __attribute__((ext_vector_type(8)));\n")?;
        f.write_str("typedef _Float16 half16 __attribute__((ext_vector_type(16)));\n")?;
        f.write_str("typedef float float8 __attribute__((ext_vector_type(8)));\n")
    }

    fn local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // threads 0-15 and threads 16-31 of the wavefront hold the same fragments respectively
        // in other words fragments are duplicated
        // so lanes 0,16 / 1,17 / ... / 15, 31 are the same
        f.write_str("uint wmmaLane = uint(threadIdx.x % 16);\n")
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
            FragmentIdent::Accumulator => match fragment.elem {
                Elem::F16 => write!(f, "half8"),
                _ => write!(f, "float8"),
            },
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
for (uint i = 0; i < uint(8); ++i) {{
  {frag}[i] = {value};
}}
"
                    )
                }
            }
            WmmaInstruction::Load {
                frag,
                value,
                layout,
                ..
            } => {
                // Matrix A must be in column major layout
                // Matrices B, C and D must be in row major layout
                let item = value.item();
                let mut value_ident = format!("{value}");
                if item.vectorization > 1 {
                    writeln!(
                        f,
                        "__half* {value}_half = reinterpret_cast<__half*>({value});"
                    )?;
                    value_ident = format!("{value}_half");
                }
                let (index, length) = match frag {
                    Variable::WmmaFragment { frag: inner, .. } => {
                        match inner.ident {
                            FragmentIdent::A | FragmentIdent::B => {
                                let index = if (inner.ident == FragmentIdent::A
                                    && inner.layout.unwrap() == FragmentLayout::ColMajor)
                                    || (inner.ident == FragmentIdent::B
                                        && inner.layout.unwrap() == FragmentLayout::RowMajor)
                                {
                                    // correct layout
                                    "i * uint(16) + wmmaLane"
                                } else {
                                    // transpose
                                    "i + wmmaLane * uint(16)"
                                };
                                (index, 16)
                            }
                            FragmentIdent::Accumulator => {
                                let length = 8;
                                // For the acc we check layout of the source
                                // the acc must be in row format which mean that each lane (thread) is in col format
                                // moreover even rows are in 1~16 first thread of the wavefront and odd rows are in 17~32
                                match layout {
                                    Some(FragmentLayout::ColMajor) => {
                                        ("(i * uint(2) + threadIdx.x / uint(16)) + wmmaLane * uint(16)", length)
                                    },
                                    Some(FragmentLayout::RowMajor) => {
                                        ("(i * uint(2) + threadIdx.x / uint(16)) * uint(16) + wmmaLane", length)
                                    },
                                    _ => panic!("cannot load data to an accumulator without knowing the layout of the data "),
                                }
                            }
                            other => panic!("unknown matrix identifier {other}"),
                        }
                    }
                    other => panic!("{other} is not a WMMMA fragment!"),
                };
                write!(
                    f,
                    "for (uint i = 0; i < uint({length}); ++i) {{
  {frag}[i] = {value_ident}[{index}];
}}
"
                )
            }
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                warp_size,
            } => {
                let ab_format = if let Variable::WmmaFragment { frag: inner_a, .. } = frag_a {
                    if let Variable::WmmaFragment { frag: inner_b, .. } = frag_b {
                        if inner_a.elem == inner_b.elem {
                            match inner_a.elem {
                                Elem::F16 => "f16",
                                Elem::BF16 => "bf16",
                                other => {
                                    panic!("{other} format not supported for {frag_a} and {frag_b}")
                                }
                            }
                        } else {
                            panic!("{frag_a} and {frag_b} have different types (respectively {} and {})", inner_a.elem, inner_b.elem)
                        }
                    } else {
                        panic!("{frag_b} is not a WMMA fragment!")
                    }
                } else {
                    panic!("{frag_a} is not a WMMA fragment!")
                };
                let cd_format = if let Variable::WmmaFragment { frag: inner_c, .. } = frag_c {
                    if let Variable::WmmaFragment { frag: inner_d, .. } = frag_d {
                        if inner_c.elem == inner_d.elem {
                            match inner_c.elem {
                                Elem::F32 => "f32",
                                Elem::F16 => "f16",
                                Elem::BF16 => "bf16",
                                other => {
                                    panic!("{other} format not supported for {frag_c} and {frag_d}")
                                }
                            }
                        } else {
                            panic!("{frag_c} and {frag_d} have different types (respectively {} and {})", inner_c.elem, inner_d.elem)
                        }
                    } else {
                        panic!("{frag_d} is not a WMMA fragment!")
                    }
                } else {
                    panic!("{frag_c} is not a WMMA fragment!")
                };
                writeln!(
                    f,
                    "{frag_d} = __builtin_amdgcn_wmma_{cd_format}_16x16x16_{ab_format}_w{warp_size}({frag_a}, {frag_b}, {frag_c});"
                )
            }
            WmmaInstruction::Store {
                output,
                frag,
                layout,
                ..
            } => {
                let item = output.item();
                let mut output_ident = format!("{output}");
                if item.vectorization > 1 {
                    writeln!(
                        f,
                        "float* {output}_float = reinterpret_cast<float*>({output});"
                    )?;
                    output_ident = format!("{output}_float");
                }
                // frag holds a result column where threads 0-15 of the wavefront have the even rows and threads 16-31 the odd rows
                // moreover, since we use OPSEL to false in the Execute instruction in f16 output format, the output elements are
                // stored in even indexes (0, 2, 4, ...) (low 16-bits of the VGPR) in frag
                let frag_idx = match frag {
                    Variable::WmmaFragment { frag: inner, .. } => {
                        match inner.elem {
                            Elem::F16 | Elem::BF16 if inner.ident == FragmentIdent::A || inner.ident == FragmentIdent::B   => "elemIdx * 2",
                            Elem::F16 | Elem::BF16 if inner.ident == FragmentIdent::Accumulator  => "elemIdx",
                            Elem::F32 => "elemIdx",
                            other => panic!("C fragment format cannot be {other}. Only f16, bf16 and f32 are supported."),
                        }
                    },
                    other => panic!("{frag} is not a WMMA fragment (it is a {other})!")
                };
                // FragmentLayout here represents the desired layout of the matrix C
                let output_idx = match layout {
                    FragmentLayout::ColMajor => "wmmaLane * uint(16) + rowIdx",
                    FragmentLayout::RowMajor => "wmmaLane + rowIdx * uint(16)",
                    FragmentLayout::_Dialect(_) => "",
                };
                write!(
                    f,
                    "for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
  const uint rowIdx = elemIdx * uint(2) + threadIdx.x / uint(16);
  {output_ident}[{output_idx}] = {frag}[{frag_idx}];
}}
 "
                )
            }
            WmmaInstruction::Cast { input, output } => {
                write!(
                    f,
                    "for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
  {output}[elemIdx] = {input}[elemIdx];
}}
 "
                )
            }
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
                .map(|(i, o, c)| (i, o, c, vec![(16, 16, 16)]))
                .collect();
            result.extend(combinations);
        }
        result
    }
}
