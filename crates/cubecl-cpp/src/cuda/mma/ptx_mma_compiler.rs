//! This MMA compiler is a work in progress and shouldn't be used.

use std::fmt::Formatter;

use crate::{
    Dialect,
    cuda::{CudaDialect, arch::CudaArchitecture},
    shared::{
        Architecture, Component, DialectWmmaCompiler, Elem, FmtLeft, Fragment, FragmentIdent,
        FragmentLayout, SupportedWmmaCombinations, Variable, WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MmaSyncCompiler {}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaFill<D: Dialect> {
    frag: Fragment<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaLoad<D: Dialect> {
    frag: Fragment<D>,
    layout: Option<FragmentLayout<D>>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaStore<D: Dialect> {
    frag: Fragment<D>,
    layout: FragmentLayout<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaExecute<D: Dialect> {
    frag_a: Fragment<D>,
    frag_b: Fragment<D>,
    frag_c: Fragment<D>,
    frag_d: Fragment<D>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct MmaCast<D: Dialect> {
    frag_input: Fragment<D>,
    frag_output: Fragment<D>,
}

impl<D: Dialect> MmaFill<D> {
    pub fn fn_name(&self) -> String {
        let layout = frag_layout_str(&self.frag.layout);
        let ident = frag_ident_str(&self.frag.ident);
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);
        let elem = self.frag.elem;

        format!("mma_fill_{elem}_{ident}_{m}x{n}x{k}_{layout}")
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();
        let length = get_fragment_length(&frag);

        write!(
            f,
            "
// Fill the fragment with a scalar value.
__device__ void {name}({frag}& frag, {elem} value) {{
    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
        frag[i] = value;
    }}
}}
        "
        )
    }
}

impl<D: Dialect> MmaLoad<D> {
    pub fn fn_name(&self) -> String {
        let layout_frag = frag_layout_str(&self.frag.layout);
        let layout = frag_layout_str(&self.layout);
        let ident = frag_ident_str(&self.frag.ident);
        let elem = self.frag.elem;
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);

        format!("mma_load_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}")
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();

        let trans = match frag.layout {
            Some(layout) => match layout {
                FragmentLayout::ColMajor => "trans.",
                FragmentLayout::RowMajor => "",
                FragmentLayout::_Dialect(_) => "",
            },
            None => "",
        };

        let (amd_body, args) = match frag.ident {
            FragmentIdent::A => (
                format!(
                    r#"
__device__ void {name}_ams(uint& a0, uint& a1, uint& a2, uint& a3, const {elem}* shared_mem) {{
    asm volatile(
        "ldmatrix.sync.aligned.m16n8.x2.{trans}shared.b16 {{%0, %1, %2, %3}}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(shared_mem)
    );
}}
"#
                ),
                "fr, fr+1, fr+2, fr+3, value_ptr",
            ),
            FragmentIdent::B => (
                format!(
                    r#"
__device__ void {name}_ams(uint& b0, uint& b1, const {elem}* shared_mem) {{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.{trans}shared.b16 {{%0, %1}}, [%2];"
        : "=r"(b0), "=r"(b1)
        : "r"(shared_mem)
    );
}}
"#
                ),
                "fr, fr+1, value_ptr",
            ),
            FragmentIdent::Accumulator => return self.format_extension_acc(f),
            other => panic!("Unknown matrix identifier {other}"),
        };

        write!(
            f,
            "
{amd_body}

// Load the fragment from memory.
__device__ void {name}({frag}& frag, const {elem}* value_ptr, const uint stride) {{
    uint *fr = reinterpret_cast<uint *>(frag);

    {name}_ams({args})
}}
        "
        )
    }

    pub fn format_extension_acc(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();
        let length = get_fragment_length(&frag);

        let index_body = match self.layout {
            Some(FragmentLayout::RowMajor) => "wmmaLane + i * stride".to_string(),
            Some(FragmentLayout::ColMajor) => "i + wmmaLane * stride".to_string(),
            _ => panic!("Accumulator load requires explicit layout"),
        };

        write!(
            f,
            "
// Load the fragment from memory.
__device__ void {name}({frag}& frag, const {elem}* value_ptr, const uint stride) {{
    const uint wmmaLane = threadIdx.x % 32;

    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
        const uint index = {index_body};
        frag[i] = value_ptr[index];
    }}
}}
        "
        )
    }
}

impl<D: Dialect> MmaStore<D> {
    pub fn fn_name(&self) -> String {
        let layout_frag = frag_layout_str(&self.frag.layout);
        let layout_option = Some(self.layout);
        let layout = frag_layout_str(&layout_option);
        let ident = frag_ident_str(&self.frag.ident);
        let (m, n, k) = (self.frag.m, self.frag.n, self.frag.k);
        let elem = self.frag.elem;

        format!("mma_store_{elem}_{ident}_{m}x{n}x{k}_{layout_frag}_{layout}")
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let elem = self.frag.elem;
        let frag = self.frag;
        let name = self.fn_name();
        let length = get_fragment_length(&frag);

        let output_idx = match self.layout {
            FragmentLayout::RowMajor => "wmmaLane + i * stride".to_string(),
            FragmentLayout::ColMajor => "i + wmmaLane * stride".to_string(),
            FragmentLayout::_Dialect(_) => String::new(),
        };

        write!(
            f,
            "
// Store the fragment to memory.
__device__ void {name}({frag}& frag, {elem}* output_ptr, uint stride) {{
    const uint wmmaLane = threadIdx.x % 32;

    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
        output_ptr[{output_idx}] = frag[i];
    }}
}}
        "
        )
    }
}

impl<D: Dialect> MmaExecute<D> {
    pub fn fn_name(&self) -> String {
        let (m, n, k) = (self.frag_a.m, self.frag_a.n, self.frag_a.k);
        format!(
            "mma_execute_{}x{}x{}_{}_{}",
            m, n, k, self.frag_a.elem, self.frag_c.elem
        )
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = self.fn_name();
        // TODO: Support more shapes, only m16n8k16 for now.

        write!(
            f,
            r#"
// Execute mma.sync.
__device__ void {name}({}& a, {}& b, {}& c, {}& d) {{
    uint const *A = reinterpret_cast<uint const *>(a);
    uint const *B = reinterpret_cast<uint const *>(b);
    float const *C = reinterpret_cast<float const *>(c);
    float *D = reinterpret_cast<float *>(d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, "
        "{{%10,%11,%12,%13}};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
}}
        "#,
            self.frag_a, self.frag_b, self.frag_c, self.frag_d
        )
    }
}

impl<D: Dialect> MmaCast<D> {
    pub fn fn_name(&self) -> String {
        let layout = frag_layout_str(&self.frag_input.layout);
        let ident = frag_ident_str(&self.frag_input.ident);
        let (m, n, k) = (self.frag_input.m, self.frag_input.n, self.frag_input.k);
        let elem = self.frag_input.elem;
        let elem_out = self.frag_output.elem;

        format!("mma_cast_{elem}_to_{elem_out}_{ident}_{m}x{n}x{k}_{layout}")
    }

    pub fn format_extension(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let input = self.frag_input;
        let output = self.frag_output;
        let name = self.fn_name();
        let length = get_fragment_length(&output);

        write!(
            f,
            "
// Cast the fragment.
__device__ void {name}({input}& input, {output}& output) {{
    #pragma unroll
    for (uint i = 0; i < {length}; ++i) {{
        output[i] = input[i];
    }}
}}
        "
        )
    }
}

impl DialectWmmaCompiler<CudaDialect<Self>> for MmaSyncCompiler {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "#include <cuda_fp16.h>")?;
        writeln!(f, "#include <cuda_bf16.h>")?;
        Ok(())
    }

    fn compile_wmma_type_definitions(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Define vector types for fragments
        writeln!(f, "typedef __half half4_t[4];")?;
        writeln!(f, "typedef __half half8_t[8];")?;
        writeln!(f, "typedef __nv_bfloat16 bhalf4_t[4];")?;
        writeln!(f, "typedef __nv_bfloat16 bhalf8_t[8];")?;
        writeln!(f, "typedef float float4_t[4];")?;
        Ok(())
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match fragment.ident {
            FragmentIdent::A => match fragment.elem {
                Elem::F16 => write!(f, "half8_t"),
                Elem::BF16 => write!(f, "bhalf8_t"),
                other => panic!("Unsupported type {other} for {fragment}"),
            },
            FragmentIdent::B => match fragment.elem {
                Elem::F16 => write!(f, "half4_t"),
                Elem::BF16 => write!(f, "bhalf4_t"),
                other => panic!("Unsupported type {other} for {fragment}"),
            },
            FragmentIdent::Accumulator => match fragment.elem {
                Elem::F16 => write!(f, "half4_t"),
                Elem::BF16 => write!(f, "bhalf4_t"),
                Elem::F32 => write!(f, "float4_t"),
                other => panic!("Unsupported type {other} for {fragment}"),
            },
            FragmentIdent::_Dialect(_) => Ok(()),
        }
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value } => {
                let extension = MmaFill::new(match frag {
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
                let extension = MmaLoad::new(variable_to_frag(frag), *layout);
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
                let extension = MmaExecute::new(
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
                let extension = MmaStore::new(variable_to_frag(frag), *layout);
                let name = extension.fn_name();
                let output_ptr = frag_as_ptr(f, output, offset);
                writeln!(f, "{name}({frag}, {output_ptr}, {stride});")
            }
            WmmaInstruction::Cast { input, output } => {
                let extension = MmaCast::new(variable_to_frag(input), variable_to_frag(output));
                let name = extension.fn_name();
                writeln!(f, "{name}({input}, {output});")
            }
        }
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedWmmaCombinations {
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.is_wmma_capable() {
            let types = vec![
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
                .map(|(m, n, k)| (m, n, k, vec![(16, 8, 16)]))
                .collect();
            result.extend(combinations);
        }
        result
    }
}

fn get_fragment_length<D: Dialect>(frag: &Fragment<D>) -> u32 {
    match frag.ident {
        FragmentIdent::A => 8,
        FragmentIdent::B => 4,
        FragmentIdent::Accumulator => 4,
        FragmentIdent::_Dialect(_) => 1,
    }
}

fn frag_as_ptr<D: Dialect>(
    f: &mut Formatter<'_>,
    frag: &Variable<D>,
    offset: &Variable<D>,
) -> Variable<D> {
    let item = frag.item();
    let mut frag_ptr = Variable::tmp_ptr(item);
    if frag.is_const() {
        frag_ptr.to_const();
    }
    let frag_ptr_out = frag_ptr.fmt_left();
    writeln!(f, "{frag_ptr_out} = {frag} + {offset};").unwrap();

    if item.vectorization > 1 {
        let mut item_value = item;
        item_value.vectorization = 1;
        frag_ptr.reinterpret_ptr(f, item_value)
    } else {
        frag_ptr
    }
}

fn frag_ident_str<D: Dialect>(frag: &FragmentIdent<D>) -> &str {
    match frag {
        FragmentIdent::A => "a",
        FragmentIdent::B => "b",
        FragmentIdent::Accumulator => "c",
        FragmentIdent::_Dialect(_) => "d",
    }
}

fn frag_layout_str<D: Dialect>(frag: &Option<FragmentLayout<D>>) -> &str {
    match frag {
        Some(layout) => match layout {
            FragmentLayout::ColMajor => "col",
            FragmentLayout::RowMajor => "row",
            FragmentLayout::_Dialect(_) => "",
        },
        None => "",
    }
}

pub(crate) fn variable_to_frag<D: Dialect>(frag: &Variable<D>) -> Fragment<D> {
    match frag {
        Variable::WmmaFragment { frag, .. } => *frag,
        _ => panic!(),
    }
}
