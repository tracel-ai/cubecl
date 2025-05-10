use crate::{
    cuda::{CudaDialect, arch::CudaArchitecture},
    shared::{
        Architecture, Component, DialectWmmaCompiler, Elem, FmtLeft, Fragment, FragmentIdent,
        FragmentLayout, SupportedWmmaCombinations, Variable, WmmaInstruction,
    },
};
use cubecl_core::ir::{self as gpu, ConstantScalarValue};

use super::WMMA_MINIMUM_VERSION;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PtxWmmaCompiler {}

impl DialectWmmaCompiler<CudaDialect<Self>> for PtxWmmaCompiler {
    fn compile_wmma_includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &Variable<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        let frag = match var {
            Variable::WmmaFragment { frag, .. } => *frag,
            _ => panic!("load instruction expects a WmmaFragment"),
        };
        let reg_count = get_fragment_register_total_count(&frag);
        let ty = match frag.elem {
            Elem::F16 | Elem::BF16 | Elem::TF32 => "unsigned int",
            Elem::F32 => "float",
            Elem::F64 => "double",
            _ => panic!("unsupported type"),
        };
        writeln!(f, "{ty} {var}[{reg_count}];")
    }

    fn compile_wwma_fragment_ident(
        _f: &mut std::fmt::Formatter<'_>,
        _ident: &FragmentIdent<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_fragment_layout(
        _f: &mut std::fmt::Formatter<'_>,
        _layout: &FragmentLayout<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_fragment(
        _f: &mut std::fmt::Formatter<'_>,
        _fragment: &Fragment<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag: var, value } => {
                let frag = match var {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!("variable should be WmmaFragment"),
                };
                let reg_count = get_fragment_register_total_count(&frag);
                write!(
                    f,
                    "// fill
for (uint i = 0; i < uint({reg_count}); ++i) {{
  {var}[i] = {value};
}}
 "
                )
            }
            WmmaInstruction::Load {
                frag: var,
                value,
                offset,
                stride,
                layout,
            } => {
                let frag = match var {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!("load instruction expects a WmmaFragment"),
                };
                // Important note: the current frontend has been designed around
                // CUDA wmma which is not optimal in the case of PTX wmma and mma
                // We choose here to use the layout defined in the fragment first,
                // if it is unknown and we look into the layout passed to the instruction.
                let layout = if frag.layout.is_some() {
                    get_fragment_layout_qualifier(var)
                } else if let Some(layout) = layout {
                    get_qualifier_from_layout(layout)
                } else {
                    panic!("unknown matrix layout for wmma load instruction");
                };
                // instruction qualifiers
                let ty = get_type_qualifier(value);
                let matrix = match frag.ident {
                    FragmentIdent::A => "a",
                    FragmentIdent::B => "b",
                    FragmentIdent::Accumulator => "c",
                    FragmentIdent::_Dialect(_) => unreachable!(),
                };
                let value_ty = value.item();
                let opcode = match frag.elem {
                    Elem::F16 | Elem::BF16 | Elem::F32 | Elem::TF32 => format!(
                        "wmma.load.{matrix}.sync.aligned.{layout}.m{}n{}k{}.{ty}",
                        frag.m, frag.n, frag.k,
                    ),
                    other => panic!("{} fragment type not supported", other),
                };
                // constraints
                let mut reg_count = 0;
                let (regs_decl, out_constraints) =
                    get_variable_regs_decl_constraints(var, true, &mut reg_count);
                let buffer_reg = format_reg_and_inc(&mut reg_count);
                let (stride_reg, stride_constraint) =
                    get_variable_regs_decl_constraints(stride, false, &mut reg_count);
                let tmp_ptr = Variable::tmp_ptr(value.item());
                let tmp_ptr_left = tmp_ptr.fmt_left();
                write!(
                    f,
                    r#"// load
{tmp_ptr_left} = ({value_ty}*){value} + {offset};
asm volatile(
    "{opcode} "
    "{{{regs_decl}}}, [{buffer_reg}], {stride_reg};\n"
    : {out_constraints}
    : "l"({tmp_ptr}){stride_constraint}
);
"#
                )
            }
            WmmaInstruction::Execute {
                frag_a: var_a,
                frag_b: var_b,
                frag_c: var_c,
                frag_d: var_d,
                ..
            } => {
                let frag_a = match var_a {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!("variable should be WmmaFragment"),
                };
                let layout_a = get_fragment_layout_qualifier(var_a);
                let layout_b = get_fragment_layout_qualifier(var_b);
                let type_c = get_type_qualifier(var_c);
                let type_d = get_type_qualifier(var_d);
                let opcode = match var_a.elem() {
                    Elem::F16 | Elem::F32 => format!(
                        "wmma.mma.sync.aligned.m{}n{}k{}.{layout_a}.{layout_b}.{type_d}.{type_c}",
                        frag_a.m, frag_a.n, frag_a.k,
                    ),
                    Elem::BF16 => format!(
                        "wmma.mma.sync.aligned.{layout_a}.{layout_b}.m{}n{}k{}.f32.bf16.bf16.f32",
                        frag_a.m, frag_a.n, frag_a.k,
                    ),
                    Elem::TF32 => format!(
                        "wmma.mma.sync.aligned.{layout_a}.{layout_b}.m{}n{}k{}.f32.tf32.tf32.f32",
                        frag_a.m, frag_a.n, frag_a.k,
                    ),
                    other => panic!("{} fragment type not supported", other),
                };
                let mut reg_count = 0;
                // order matters, declare the registers in the same order as the intrinsic
                let (regs_decl_d, out_constraints_d) =
                    get_variable_regs_decl_constraints(var_d, true, &mut reg_count);
                let (regs_decl_a, in_constraints_a) =
                    get_variable_regs_decl_constraints(var_a, false, &mut reg_count);
                let (regs_decl_b, in_constraints_b) =
                    get_variable_regs_decl_constraints(var_b, false, &mut reg_count);
                let (regs_decl_c, in_constraints_c) =
                    get_variable_regs_decl_constraints(var_c, false, &mut reg_count);
                write!(
                    f,
                    r#"// execute
asm volatile(
    "{opcode} "
    "{{{regs_decl_d}}}, "
    "{{{regs_decl_a}}}, "
    "{{{regs_decl_b}}}, "
    "{{{regs_decl_c}}};\n"
    : {out_constraints_d}
    : {in_constraints_a}, {in_constraints_b}, {in_constraints_c}
);
"#
                )
            }
            WmmaInstruction::Store {
                output,
                frag: var,
                stride,
                offset,
                layout,
            } => {
                let frag_acc = match var {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!("variable should be WmmaFragment"),
                };
                // instruction qualifiers
                let layout = match layout {
                    FragmentLayout::ColMajor => "col",
                    FragmentLayout::RowMajor => "row",
                    FragmentLayout::_Dialect(..) => unreachable!(),
                };
                let opcode = match var.elem() {
                    Elem::F16 | Elem::BF16 => format!(
                        // hack because wmma.store does not support bf16
                        // f16 should still work correctly for bf16 as long
                        // as the input registers are in correct format
                        "wmma.store.d.sync.aligned.{layout}.m{}n{}k{}.f16",
                        frag_acc.m, frag_acc.n, frag_acc.k,
                    ),
                    Elem::TF32 | Elem::F32 => format!(
                        // same hack for tf32
                        "wmma.store.d.sync.aligned.{layout}.m{}n{}k{}.f32",
                        frag_acc.m, frag_acc.n, frag_acc.k,
                    ),
                    other => panic!("{} fragment type not supported", other),
                };
                // constraints
                let mut reg_count = 0;
                let buffer_reg = format_reg_and_inc(&mut reg_count);
                // offset and stride can be passed as local const or as const scalar
                // we need to handle both cases correctly in the asm.
                let (stride_reg, stride_constraint) =
                    get_variable_regs_decl_constraints(stride, false, &mut reg_count);
                // we start at 2 because of the buffer address calculation
                let (regs_decl, in_constraints) =
                    get_variable_regs_decl_constraints(var, false, &mut reg_count);
                let tmp_ptr = Variable::tmp_ptr(output.item());
                let tmp_ptr_left = tmp_ptr.fmt_left();
                write!(
                    f,
                    r#"// store
{tmp_ptr_left} = {output} + {offset};
asm volatile(
    "{opcode} "
    "[{buffer_reg}], {{{regs_decl}}}, {stride_reg};\n"
    :
    : "l"({tmp_ptr}),
      {in_constraints}{stride_constraint}
);
"#
                )
            }
            WmmaInstruction::Cast { input, output } => {
                let frag = match input {
                    Variable::WmmaFragment { frag, .. } => *frag,
                    _ => panic!("variable should be WmmaFragment"),
                };
                let reg_count = get_fragment_register_total_count(&frag);
                match output.elem() {
                    Elem::F16 => {
                        write!(
                            f,
                            "// cast
for (int i = 0; i < {reg_count}; ++i) {{
    __half h_lo = __float2half_rn({input}[2*i + 0]);
    __half h_hi = __float2half_rn({input}[2*i + 1]);
    __half2 h2 = __halves2half2(h_lo, h_hi);
    {output}[i] = *reinterpret_cast<unsigned int*>(&h2);
}}
"
                        )
                    }
                    Elem::BF16 => {
                        write!(
                            f,
                            "// cast
for (int i = 0; i < {reg_count}; ++i) {{
    __nv_bfloat16 b_lo = __float2bfloat16({input}[2*i + 0]);
    __nv_bfloat16 b_hi = __float2bfloat16({input}[2*i + 1]);
    __nv_bfloat162 bf2 = __halves2bfloat162(b_lo, b_hi);
    {output}[i] = *reinterpret_cast<unsigned int*>(&bf2);
}}
"
                        )
                    }
                    other => panic!("casting fragment to {other} not supported"),
                }
            }
        }
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedWmmaCombinations {
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.get_version() >= WMMA_MINIMUM_VERSION {
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
            if arch.get_version() >= 80 {
                result.push((
                    gpu::Elem::Float(gpu::FloatKind::TF32),
                    gpu::Elem::Float(gpu::FloatKind::TF32),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                    vec![(16, 16, 8)],
                ));
            }
        }
        result
    }
}

fn get_fragment_register_total_count(frag: &Fragment<CudaDialect<PtxWmmaCompiler>>) -> u8 {
    let m = frag.m as u32;
    let n = frag.n as u32;
    let k = frag.k as u32;
    let elements = match frag.ident {
        FragmentIdent::A => m * k,
        FragmentIdent::B => k * n,
        FragmentIdent::Accumulator => m * n,
        _ => unreachable!(),
    };
    let bits_per_elem = match frag.elem {
        Elem::F16 | Elem::BF16 => 16,
        Elem::F32 | Elem::TF32 => 32,
        _ => panic!("unsupported WMMA element {:?}", frag.elem),
    };
    // TODO: retrieve the warp size from the compiler CompilationOptions
    let lanes_per_reg = 32 / bits_per_elem;
    // choose threads-per-frag:
    // - accumulators always use 32 lanes
    // - A/B use 16 lanes _except_ TF32 (k=8) which also uses 32 lanes
    let threads_per_frag = match frag.ident {
        FragmentIdent::Accumulator => 32,
        FragmentIdent::A | FragmentIdent::B => {
            if frag.elem == Elem::TF32 {
                32
            } else {
                16
            }
        }
        _ => unreachable!(),
    };
    let regs = elements / (lanes_per_reg * threads_per_frag);
    regs as u8
}

fn get_type_qualifier(var: &Variable<CudaDialect<PtxWmmaCompiler>>) -> String {
    match var.elem() {
        Elem::F16 => "f16",
        Elem::BF16 => "bf16",
        Elem::F32 => "f32",
        Elem::TF32 => "tf32",
        Elem::F64 => "f64",
        _ => panic!("unsupported WMMA fragment type"),
    }
    .to_string()
}

fn get_fragment_layout_qualifier(var: &Variable<CudaDialect<PtxWmmaCompiler>>) -> String {
    let frag = match var {
        Variable::WmmaFragment { frag, .. } => *frag,
        _ => panic!("variable should be WmmaFragment"),
    };
    match frag.layout {
        Some(layout) => get_qualifier_from_layout(&layout),
        None => "".to_string(),
    }
}

fn get_qualifier_from_layout(layout: &FragmentLayout<CudaDialect<PtxWmmaCompiler>>) -> String {
    match layout {
        FragmentLayout::ColMajor => "col",
        FragmentLayout::RowMajor => "row",
        FragmentLayout::_Dialect(..) => unreachable!(),
    }
    .to_string()
}

fn get_variable_regs_decl_constraints(
    var: &Variable<CudaDialect<PtxWmmaCompiler>>,
    output: bool,
    reg_count: &mut u8,
) -> (String, String) {
    match var {
        Variable::WmmaFragment { frag, .. } => {
            let reg_total_count = get_fragment_register_total_count(frag);
            let reg_decl = (0..reg_total_count)
                .map(|_| format_reg_and_inc(reg_count))
                .collect::<Vec<_>>()
                .join(",");
            let frag_elem = frag.elem;
            let modifier = format!(
                "{}{}",
                if output { "=" } else { "" },
                match frag_elem {
                    Elem::F32 => "f",
                    Elem::F64 => "d",
                    _ => "r",
                },
            );
            let constraints = (0..reg_total_count)
                .map(|i| format!("\"{modifier}\"({var}[{}])", i))
                .collect::<Vec<_>>()
                .join(", ");
            (reg_decl, constraints)
        }
        Variable::ConstantScalar(number, ..) => match number {
            ConstantScalarValue::UInt(val, ..) => (val.to_string(), "".to_string()),
            _ => panic!("variable should be an unsigned integer"),
        },
        _ => (format_reg_and_inc(reg_count), format!(r#", "r"({var})"#)),
    }
}

fn format_reg_and_inc(count: &mut u8) -> String {
    let res = format!("%{count}");
    *count += 1;
    res
}
