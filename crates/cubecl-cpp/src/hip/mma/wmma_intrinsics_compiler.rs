use std::fmt::Formatter;

use crate::{
    Dialect,
    hip::{HipDialect, arch::AMDArchitecture},
    shared::{
        Architecture, DialectWmmaCompiler, Elem, Flags, Fragment, FragmentIdent, FragmentLayout,
        SupportedWmmaCombinations, Variable, WmmaInstruction, frag_as_ptr, frag_ident_str,
        frag_layout_str, variable_to_frag, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WmmaIntrinsicCompiler {}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaFill<D: Dialect> {
    frag: Fragment<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaLoad<D: Dialect> {
    frag: Fragment<D>,
    layout: Option<FragmentLayout<D>>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaStore<D: Dialect> {
    frag: Fragment<D>,
    layout: FragmentLayout<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaExecute<D: Dialect> {
    frag_a: Fragment<D>,
    frag_b: Fragment<D>,
    frag_c: Fragment<D>,
    frag_d: Fragment<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct WmmaCast<D: Dialect> {
    frag_input: Fragment<D>,
    frag_output: Fragment<D>,
}

impl<D: Dialect> WmmaFill<D> {
    pub fn fn_name(&self) -> String {
        let layout = frag_layout_str(&self.frag.layout);
        let ident = frag_ident_str(&self.frag.ident);
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);
        let elem = self.frag.elem;

        format!("wmma_fill_{elem}_{ident}_{m}x{n}x{k}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();

        write!(
            f,
            "
// Fill the fragment.
__device__ void {name}({frag}& frag, {elem} value) {{
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {{
      frag[i] = value;
    }}
}}
        "
        )
    }
}

impl<D: Dialect> WmmaLoad<D> {
    pub fn fn_name(&self) -> String {
        let layout_frag = frag_layout_str(&self.frag.layout);
        let layout = frag_layout_str(&self.layout);
        let ident = frag_ident_str(&self.frag.ident);
        let elem = self.frag.elem;
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);

        format!("wmma_load_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}",)
    }

    /// Matrix A must be in column major layout (so fragments correspond to a row)
    /// Matrices B, C and D must be in row major layout (so fragments correspond to a column)
    ///
    /// Each lane is a thread so each column get 8 VGPRs used to store fragments
    /// Here is the layout for C and D matrices and how they map to registers
    ///
    /// Lane index   0      1      2      3      ...     13     14     15     ...     17     18     ...     30     31
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR0      | 1,1  | 1,2  | 1,3  | 1,4  | ...  | 1,13 | 1,14 | 1,15 | ...  | 2,1  | 2,2  | ...  | 2,15 | 2,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR1      | 3,1  | 3,2  | 3,3  | 3,4  | ...  | 3,13 | 3,14 | 3,15 | ...  | 4,1  | 4,2  | ...  | 4,15 | 4,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR2      | 5,1  | 5,2  | 5,3  | 5,4  | ...  | 5,13 | 5,14 | 5,15 | ...  | 6,1  | 6,2  | ...  | 6,15 | 6,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR3      | 7,1  | 7,2  | 7,3  | 7,4  | ...  | 7,13 | 7,14 | 7,15 | ...  | 8,1  | 8,2  | ...  | 8,15 | 8,16 |
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR4      | 9,1  | 9,2  | 9,3  | 9,4  | ...  | 9,13 | 9,14 | 9,15 | ...  | 10,1 | 10,2 | ...  | 10,15| 10,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR5      | 11,1 | 11,2 | 11,3 | 11,4 | ...  | 11,13| 11,14| 11,15| ...  | 12,1 | 12,2 | ...  | 12,15| 12,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR6      | 13,1 | 13,2 | 13,3 | 13,4 | ...  | 13,13| 13,14| 13,15| ...  | 14,1 | 14,2 | ...  | 14,15| 14,16|
    /// --------------------------------------------------------------------------------------------------------------
    /// VGPR7      | 15,1 | 15,2 | 15,3 | 15,4 | ...  | 15,13| 15,14| 15,15| ...  | 16,1 | 16,2 | ...  | 16,15| 16,16|
    /// --------------------------------------------------------------------------------------------------------------
    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();

        let (index_body, length, step) = match frag.ident {
            FragmentIdent::A | FragmentIdent::B => {
                let length = 16;
                let step = 1;
                // fragment a and b are always in half precision and they don't require special attention
                // to how they are stored in memory as matrix A and B are also in half precision
                let index = if (frag.ident == FragmentIdent::A
                    && frag.layout.unwrap() == FragmentLayout::ColMajor)
                    || (frag.ident == FragmentIdent::B
                        && frag.layout.unwrap() == FragmentLayout::RowMajor)
                {
                    "i * stride + wmmaLane".to_string()
                } else {
                    "i + wmmaLane * stride".to_string()
                };
                (index, length, step)
            }
            FragmentIdent::Accumulator => {
                let length = 8;
                let step = get_output_accumulator_index_step(&elem, &frag);
                let index = match self.layout {
                    Some(FragmentLayout::ColMajor) => {
                        "(i * uint(2) + threadIdx.x / uint(16)) + wmmaLane * stride".to_string()
                    }
                    Some(FragmentLayout::RowMajor) => {
                        "(i * uint(2) + threadIdx.x / uint(16)) * stride + wmmaLane".to_string()
                    }
                    _ => panic!(
                        "cannot load data to an accumulator without knowing the layout of the data"
                    ),
                };
                (index, length, step)
            }
            other => panic!("unknown matrix identifier {other}"),
        };

        write!(
            f,
            "
// Load the fragment.
__device__ void {name}({frag}& frag, const {elem}* value_ptr, const uint stride) {{
    const uint wmmaLane = uint(threadIdx.x % 16);

    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
      const uint index = {index_body};
      frag[i * {step}] = value_ptr[index];
    }}
}}
        "
        )
    }
}

impl<D: Dialect> WmmaStore<D> {
    pub fn fn_name(&self) -> String {
        let layout_frag = frag_layout_str(&self.frag.layout);
        let layout_option = Some(self.layout);
        let layout = frag_layout_str(&layout_option);
        let ident = frag_ident_str(&self.frag.ident);
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);
        let elem = self.frag.elem;

        format!("wmma_store_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();
        // frag holds a result column where threads 0-15 of the wavefront have the even rows and threads 16-31 the odd rows
        // moreover, since we use OPSEL to false in the Execute instruction in f16 output format, the output elements are
        // stored in even indexes (0, 2, 4, ...) (low 16-bits of the VGPR) in frag
        let frag_idx = match elem {
            Elem::F16 | Elem::BF16 => "elemIdx * 2",
            Elem::F32 => "elemIdx",
            other => {
                panic!("C fragment format cannot be {other}. Only f16, bf16 and f32 are supported.")
            }
        };
        // FragmentLayout here represents the desired layout of the matrix C
        let output_idx = match self.layout {
            FragmentLayout::ColMajor => "wmmaLane * stride + rowIdx".to_string(),
            FragmentLayout::RowMajor => "wmmaLane + rowIdx * stride".to_string(),
            FragmentLayout::_Dialect(_) => String::new(),
        };

        write!(
            f,
            "
// Store the fragment.
__device__ void {name}({frag}& frag, {elem}* output_ptr, uint stride) {{
    const uint wmmaLane = uint(threadIdx.x % 16);

    #pragma unroll
    for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
      const uint rowIdx = elemIdx * uint(2) + threadIdx.x / uint(16);
      output_ptr[{output_idx}] = frag[{frag_idx}];
    }}
}}
        "
        )
    }
}

impl<D: Dialect> WmmaExecute<D> {
    pub fn fn_name(&self) -> String {
        format!(
            "wmma_execute_16x16x16_{}_{}",
            self.frag_a.elem, self.frag_c.elem
        )
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = self.fn_name();
        let ab_format = match self.frag_a.elem {
            Elem::F32 => "f32",
            Elem::BF16 => "bf16",
            Elem::F16 => "f16",
            _ => panic!(),
        };
        let (cd_format, opsel) = match self.frag_c.elem {
            Elem::F32 => ("f32", ""),
            Elem::BF16 => ("bf16", ", false"),
            Elem::F16 => ("f16", ", false"),
            _ => panic!(),
        };
        let warp_size = 32;
        write!(
            f,
            "
// Execute wmma.
__device__ void {name}({}& frag_a, {}& frag_b, {}& frag_c, {}& frag_d) {{
    frag_d = __builtin_amdgcn_wmma_{cd_format}_16x16x16_{ab_format}_w{warp_size}(frag_a, frag_b, frag_c{opsel});
}}
        ", self.frag_a, self.frag_b, self.frag_c, self.frag_d
        )
    }
}

impl<D: Dialect> WmmaCast<D> {
    pub fn fn_name(&self) -> String {
        let layout = frag_layout_str(&self.frag_input.layout);
        let ident = frag_ident_str(&self.frag_input.ident);
        let (m, n, k) = (self.frag_input.m, self.frag_input.n, self.frag_input.k);
        let elem = self.frag_input.elem;
        let elem_out = self.frag_output.elem;

        format!("wmma_cast_{elem}_to_{elem_out}_{ident}_{m}x{n}x{k}_{layout}",)
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let input = self.frag_input;
        let output = self.frag_output;
        let name = self.fn_name();
        let step = match output.ident {
            FragmentIdent::Accumulator => {
                get_output_accumulator_index_step(&self.frag_input.elem, &output)
            }
            _ => 1,
        };

        write!(
            f,
            "
// Cast the fragment.
__device__ void {name}({input}& input, {output}& output) {{
    #pragma unroll
    for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {{
      output[elemIdx * {step}] = input[elemIdx];
    }}
}}
        "
        )
    }
}

impl DialectWmmaCompiler<HipDialect<Self>> for WmmaIntrinsicCompiler {
    fn compile_wmma_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        flags: &Flags,
    ) -> std::fmt::Result {
        if flags.elem_bf16 {
            f.write_str("typedef __bf16 bhalf8_t __attribute__((ext_vector_type(8)));\n")?;
            f.write_str("typedef __bf16 bhalf16_t __attribute__((ext_vector_type(16)));\n")?;
        }
        if flags.elem_f16 {
            f.write_str("typedef _Float16 half8_t __attribute__((ext_vector_type(8)));\n")?;
            f.write_str("typedef _Float16 half16_t __attribute__((ext_vector_type(16)));\n")?;
        }
        f.write_str("typedef float float8_t __attribute__((ext_vector_type(8)));\n")
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<HipDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<HipDialect<Self>>,
    ) -> std::fmt::Result {
        match fragment.ident {
            FragmentIdent::A | FragmentIdent::B => match fragment.elem {
                Elem::F16 => write!(f, "half16_t"),
                Elem::BF16 => write!(f, "bhalf16_t"),
                other => panic!("unsupported type {other} for {fragment}"),
            },
            FragmentIdent::Accumulator => match fragment.elem {
                Elem::F16 => write!(f, "half16_t"),
                Elem::BF16 => write!(f, "bhalf16_t"),
                Elem::F32 => write!(f, "float8_t"),
                other => panic!("unsupported type {other} for {fragment}"),
            },
            FragmentIdent::_Dialect(_) => Ok(()),
        }
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<HipDialect<Self>>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value } => {
                let extension = WmmaFill::new(match frag {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!(),
                });
                let name = extension.fn_name();
                writeln!(f, "{name}({frag}, {value});")
            }
            WmmaInstruction::Load {
                frag,
                value,
                layout,
                offset,
                stride,
            } => {
                let extension = WmmaLoad::new(variable_to_frag(frag), *layout);
                let name = extension.fn_name();
                let value_ptr = frag_as_ptr(f, value, offset);
                writeln!(f, "{name}({frag}, {value_ptr}, {stride});")
            }
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
                warp_size,
            } => {
                assert_eq!(*warp_size, 32, "Only warp size of 32 supported");

                let extension = WmmaExecute::new(
                    variable_to_frag(frag_a),
                    variable_to_frag(frag_b),
                    variable_to_frag(frag_c),
                    variable_to_frag(frag_d),
                );
                let name = extension.fn_name();
                writeln!(f, "{name}({frag_a}, {frag_b}, {frag_c}, {frag_d});")
            }
            WmmaInstruction::Store {
                output,
                frag,
                layout,
                offset,
                stride,
            } => {
                let extension = WmmaStore::new(variable_to_frag(frag), *layout);
                let name = extension.fn_name();
                let output_ptr = frag_as_ptr(f, output, offset);
                writeln!(f, "{name}({frag}, {output_ptr}, {stride});")
            }
            WmmaInstruction::Cast { input, output } => {
                let extension = WmmaCast::new(variable_to_frag(input), variable_to_frag(output));
                let name = extension.fn_name();
                writeln!(f, "{name}({input}, {output});")
            }
        }
    }

    fn supported_wmma_combinations(arch: &AMDArchitecture) -> SupportedWmmaCombinations {
        // Reference: https://gpuopen.com/learn/wmma_on_rdna3/
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.is_wmma_capable() {
            // Types fully supported.
            let types = vec![
                (
                    gpu::Elem::Float(gpu::FloatKind::F16), // m
                    gpu::Elem::Float(gpu::FloatKind::F16), // n
                    gpu::Elem::Float(gpu::FloatKind::F16), // k
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
                .map(|(m, n, k)| (m, n, k, vec![(16, 16, 16)]))
                .collect();
            result.extend(combinations);
        }
        result
    }
}

fn get_output_accumulator_index_step<D: Dialect>(
    input_elem: &Elem<D>,
    output: &Fragment<D>,
) -> u32 {
    // Each VGPR is 32 bit wide and there is 8 VGPR per lane, an accumulator can then be either:
    // - a vector of 8 float
    // - a vector of 16 half
    // Depending on the precision used for the input, the whole 32 bits per register will be used or
    // just only 16 bits. In such a case we always use the lower 16 bits (opsel set to false) which means
    // that we only assign values to even indexes of the accumulator (0, 2, 4, ...)

    assert_eq!(output.ident, FragmentIdent::<D>::Accumulator);

    match input_elem {
        Elem::F16 | Elem::BF16 | Elem::F32 => {
            match output.elem {
                // loading into accumulator of 16 half precision
                Elem::F16 | Elem::BF16 => 2,
                // loading into accumulator of 8 full precision
                Elem::F32 => 1,
                other => panic!("unsupported format {other} for {output}"),
            }
        }
        other => panic!("unsupported format {other} for {input_elem}"),
    }
}
