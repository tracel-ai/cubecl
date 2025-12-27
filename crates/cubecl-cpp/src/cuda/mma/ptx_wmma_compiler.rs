use std::fmt::Display;

use crate::{
    Dialect,
    cuda::{
        CudaDialect,
        arch::CudaArchitecture,
        ptx::{comma_separated, ldmatrix_call, stmatrix_call},
    },
    shared::{
        Architecture, Component, DialectWmmaCompiler, Elem, Flags, FmtLeft, Fragment,
        FragmentIdent, FragmentLayout, ManualMma, SupportedMmaCombinations,
        SupportedScaledMmaCombinations, Variable, WmmaInstruction,
    },
};
use cubecl_core::ir::{self as gpu, ConstantValue, Matrix, MatrixIdent};
use cubecl_runtime::{MmaConfig, ScaledMmaConfig};
use itertools::Itertools;

use super::WMMA_MINIMUM_VERSION;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PtxWmmaCompiler {}

impl DialectWmmaCompiler<CudaDialect<Self>> for PtxWmmaCompiler {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
        // We need mma header for conversion
        if flags.elem_tf32 {
            f.write_str("#include <mma.h>\n")?;
        }
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
            Elem::U8 | Elem::I8 | Elem::F16 | Elem::BF16 | Elem::TF32 => "unsigned int",
            Elem::F32 => "float",
            Elem::F64 => "double",
            _ => panic!("unsupported type"),
        };
        writeln!(f, "{ty} {var}[{reg_count}];")
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
                    Elem::U8 | Elem::I8 | Elem::F16 | Elem::BF16 | Elem::F32 | Elem::TF32 => {
                        format!(
                            "wmma.load.{matrix}.sync.aligned.{layout}.m{}n{}k{}.{ty}",
                            frag.m, frag.n, frag.k,
                        )
                    }
                    other => panic!("{other} fragment type not supported"),
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
            WmmaInstruction::LdMatrix {
                output,
                buffer,
                offset,
                line_size,
                factor,
                transpose,
            } => f.write_str(&ldmatrix_call(
                output, buffer, offset, line_size, factor, transpose,
            )),
            WmmaInstruction::StMatrix {
                registers,
                buffer,
                offset,
                line_size,
                factor,
                transpose,
            } => f.write_str(&stmatrix_call(
                registers, buffer, offset, line_size, factor, transpose,
            )),
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
                    Elem::U8 | Elem::I8 | Elem::F16 | Elem::F32 => format!(
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
                    other => panic!("{other} fragment type not supported"),
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
            WmmaInstruction::ExecuteManual {
                shape,
                frag_a,
                frag_b,
                frag_c,
                frag_d,
            } => {
                Self::compile_manual_mma(f, ManualMma::new(*shape, frag_a, frag_b, frag_c, frag_d))
            }
            WmmaInstruction::ExecuteScaled {
                shape,
                frag_a,
                frag_b,
                frag_c,
                frag_d,

                scales_a,
                scales_b,
                scales_factor,
            } => Self::compile_scaled_mma(
                f,
                ManualMma::new(*shape, frag_a, frag_b, frag_c, frag_d),
                *scales_a,
                *scales_b,
                *scales_factor,
            ),
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
                    Elem::I32 => format!(
                        // same hack for tf32
                        "wmma.store.d.sync.aligned.{layout}.m{}n{}k{}.s32",
                        frag_acc.m, frag_acc.n, frag_acc.k,
                    ),
                    other => panic!("{other} fragment type not supported"),
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

    fn compile_manual_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        compile_manual_mma(f, mma)
    }

    fn compile_scaled_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
        scales_a: Variable<CudaDialect<Self>>,
        scales_b: Variable<CudaDialect<Self>>,
        scales_factor: u32,
    ) -> std::fmt::Result {
        compile_scaled_mma(f, mma, scales_a, scales_b, scales_factor)
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        let mut result: SupportedMmaCombinations = vec![];
        if arch.get_version() >= WMMA_MINIMUM_VERSION {
            // Types fully supported.
            let types = vec![
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16), // m
                    gpu::ElemType::Float(gpu::FloatKind::F16), // n
                    gpu::ElemType::Float(gpu::FloatKind::F16), // k
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
            ];
            let combinations: SupportedMmaCombinations = types
                .into_iter()
                .map(|(a, b, cd)| MmaConfig {
                    a_type: a.into(),
                    b_type: b.into(),
                    cd_type: cd.into(),
                    m: 16,
                    n: 16,
                    k: 16,
                })
                .collect();
            result.extend(combinations);
            if arch.get_version() >= 72 {
                result.extend([
                    MmaConfig {
                        a_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                        b_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                        cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                        m: 16,
                        n: 16,
                        k: 16,
                    },
                    MmaConfig {
                        a_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                        b_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                        cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                        m: 16,
                        n: 16,
                        k: 16,
                    },
                ]);
            }
            if arch.get_version() >= 80 {
                result.push(MmaConfig {
                    a_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    b_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                    m: 16,
                    n: 16,
                    k: 8,
                });
            }
        }
        result
    }

    fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        supported_mma_combinations(arch)
    }

    fn supported_scaled_mma_combinations(
        arch: &CudaArchitecture,
    ) -> SupportedScaledMmaCombinations {
        supported_scaled_mma_combinations(arch)
    }
}

fn get_fragment_register_total_count(frag: &Fragment<CudaDialect<PtxWmmaCompiler>>) -> u32 {
    let Fragment {
        ident,
        m,
        n,
        k,
        elem,
        ..
    } = frag;
    let elements = match ident {
        FragmentIdent::A => m * k,
        FragmentIdent::B => k * n,
        FragmentIdent::Accumulator => m * n,
        _ => unreachable!(),
    };
    let bits_per_elem = elem.size_bits() as u32;
    // TODO: retrieve the warp size from the compiler CompilationOptions
    let lanes_per_reg = 32 / bits_per_elem;
    // choose threads-per-frag:
    // - accumulators always use 32 lanes
    // - A/B use 16 lanes _except_ TF32 (k=8) which also uses 32 lanes
    let threads_per_frag = match ident {
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

    elements / (lanes_per_reg * threads_per_frag)
}

fn get_type_qualifier(var: &Variable<CudaDialect<PtxWmmaCompiler>>) -> String {
    match var.elem() {
        Elem::U8 => "u8",
        Elem::I8 => "s8",
        Elem::F16 => "f16",
        Elem::BF16 => "bf16",
        Elem::F32 => "f32",
        Elem::TF32 => "tf32",
        Elem::I32 => "s32",
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
                .map(|i| format!("\"{modifier}\"({var}[{i}])"))
                .collect::<Vec<_>>()
                .join(", ");
            (reg_decl, constraints)
        }
        Variable::Constant(number, ..) => match number {
            ConstantValue::UInt(val, ..) => (val.to_string(), "".to_string()),
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

fn as_ty(var: impl Display, ty: impl Display) -> String {
    format!("reinterpret_cast<{ty}&>({var})")
}

fn as_const_ty(var: impl Display, ty: impl Display) -> String {
    format!("reinterpret_cast<const {ty}&>({var})")
}

pub(super) fn compile_manual_mma<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    mma: ManualMma<D>,
) -> std::fmt::Result {
    let ManualMma {
        shape,
        frag_a,
        frag_b,
        frag_c,
        frag_d,
    } = mma;

    let a_elem = frag_a.elem().unpacked();
    let b_elem = frag_b.elem().unpacked();
    let cd_elem = frag_c.elem().unpacked();

    let ab_ty = match a_elem {
        Elem::F32 => &format!("{}", Elem::<D>::F32),
        _ => &format!("{}", Elem::<D>::U32),
    };
    let cd_ty = match cd_elem {
        Elem::F32 => &format!("{}", Elem::<D>::F32),
        _ => &format!("{}", Elem::<D>::U32),
    };

    let a_elems = shape.num_elems(FragmentIdent::<D>::A) / 32;
    let b_elems = shape.num_elems(FragmentIdent::<D>::B) / 32;
    let cd_elems = shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;

    let a_regs = a_elems as usize / (32 / frag_a.elem().unpacked().size_bits());
    let b_regs = b_elems as usize / (32 / frag_b.elem().unpacked().size_bits());
    let cd_regs = cd_elems as usize / (32 / frag_c.elem().unpacked().size_bits());

    let frag_a = (0..a_regs).map(|i| as_const_ty(format!("{frag_a}[{i}]"), ab_ty));
    let frag_b = (0..b_regs).map(|i| as_const_ty(format!("{frag_b}[{i}]"), ab_ty));

    // C and D fragments are always vectorized for optimal stores and casts, but are taken as separate
    // registers for f32 so we need to unpack it. f16 is taken in packed registers, so use as is.
    let frag_c = match cd_elem.size() {
        4 | 8 => (0..cd_regs)
            .map(|i| as_ty(format!("{frag_c}[{}].i_{}", i / 2, i % 2), cd_ty))
            .collect::<Vec<_>>(),
        2 => (0..cd_regs)
            .map(|i| as_ty(format!("{frag_d}[{i}]"), cd_ty))
            .collect::<Vec<_>>(),
        other => panic!("Found unhandled accumulator elem size {other}"),
    };
    let frag_d = match cd_elem.size() {
        4 | 8 => (0..cd_regs)
            .map(|i| as_ty(format!("{frag_d}[{}].i_{}", i / 2, i % 2), cd_ty))
            .collect::<Vec<_>>(),
        2 => (0..cd_regs)
            .map(|i| as_ty(format!("{frag_d}[{i}]"), cd_ty))
            .collect::<Vec<_>>(),
        other => panic!("Found unhandled accumulator elem size {other}"),
    };
    let args = comma_separated(frag_a.chain(frag_b).chain(frag_c).chain(frag_d));
    write!(
        f,
        "__mma_m16n8k{}_{}_{}_{}({args});",
        shape.k, a_elem, b_elem, cd_elem
    )
}

pub(super) fn compile_scaled_mma<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    mma: ManualMma<D>,
    scales_a: Variable<D>,
    scales_b: Variable<D>,
    scales_factor: u32,
) -> std::fmt::Result {
    let ManualMma {
        shape,
        frag_a,
        frag_b,
        frag_c,
        frag_d,
    } = mma;

    let a_elem = frag_a.elem().unpacked();
    let b_elem = frag_b.elem().unpacked();
    let cd_elem = frag_c.elem().unpacked();

    let ab_ty = &format!("{}", Elem::<D>::U32);
    let cd_ty = &format!("{}", Elem::<D>::F32);

    let a_elems = shape.num_elems(FragmentIdent::<D>::A) / 32;
    let b_elems = shape.num_elems(FragmentIdent::<D>::B) / 32;
    let cd_elems = shape.num_elems(FragmentIdent::<D>::Accumulator) / 32;

    let a_regs = a_elems as usize / (32 / frag_a.elem().unpacked().size_bits());
    let b_regs = b_elems as usize / (32 / frag_b.elem().unpacked().size_bits());
    let cd_regs = cd_elems as usize / (32 / frag_c.elem().unpacked().size_bits());

    let frag_a = (0..a_regs).map(|i| as_const_ty(format!("{frag_a}[{i}]"), ab_ty));
    let frag_b = (0..b_regs).map(|i| as_const_ty(format!("{frag_b}[{i}]"), ab_ty));
    let frag_c = (0..cd_regs).map(|i| as_const_ty(format!("{frag_c}[{i}]"), cd_ty));
    let frag_d = (0..cd_regs).map(|i| as_ty(format!("{frag_d}[{i}]"), cd_ty));
    let fragments = comma_separated(frag_a.chain(frag_b).chain(frag_c).chain(frag_d));
    write!(
        f,
        "__mma_scaled_{scales_factor}x_m16n8k{}_{}_{}_{}({fragments}, reinterpret_cast<uint32&>({scales_a}), reinterpret_cast<uint32&>({scales_b}));",
        shape.k, a_elem, b_elem, cd_elem
    )
}

pub(super) fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
    let mut result: SupportedMmaCombinations = vec![];
    // Higher than WMMA because we only support the newest shapes. Other shapes would make things
    // very complicated.
    // Also only use f32 accumulators for now
    if arch.get_version() >= 80 {
        result.extend([
            MmaConfig {
                a_type: gpu::ElemType::Float(gpu::FloatKind::F16).into(), // a
                b_type: gpu::ElemType::Float(gpu::FloatKind::F16).into(), // b
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(), // cd
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                b_type: gpu::ElemType::Float(gpu::FloatKind::BF16).into(),
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                b_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                m: 16,
                n: 8,
                k: 8,
            },
            MmaConfig {
                a_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                b_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                b_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                b_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: gpu::ElemType::UInt(gpu::UIntKind::U8).into(),
                b_type: gpu::ElemType::Int(gpu::IntKind::I8).into(),
                cd_type: gpu::ElemType::Int(gpu::IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            // TODO: u4/i4/b1, there's no types for them yet
        ]);
    }
    if arch.get_version() >= 89 {
        let f8f6f4_types = [
            gpu::FloatKind::E4M3,
            gpu::FloatKind::E5M2,
            gpu::FloatKind::E3M2,
            gpu::FloatKind::E2M3,
            gpu::FloatKind::E2M1,
        ];
        let combinations = f8f6f4_types.iter().cartesian_product(f8f6f4_types.iter());
        result.extend(combinations.map(|(t1, t2)| MmaConfig {
            a_type: gpu::ElemType::Float(*t1).into(),
            b_type: gpu::ElemType::Float(*t2).into(),
            cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
            m: 16,
            n: 8,
            k: 32,
        }));
    }
    result
}

pub(super) fn supported_scaled_mma_combinations(
    arch: &CudaArchitecture,
) -> SupportedScaledMmaCombinations {
    let mut result: SupportedScaledMmaCombinations = vec![];
    // sm_120f
    if arch.get_version() >= 120 && arch.get_version() < 130 {
        let f8f6f4_types = [
            gpu::FloatKind::E4M3,
            gpu::FloatKind::E5M2,
            gpu::FloatKind::E3M2,
            gpu::FloatKind::E2M3,
            gpu::FloatKind::E2M1,
        ];
        let combinations = f8f6f4_types
            .iter()
            .flat_map(|t1| f8f6f4_types.iter().map(move |t2| (t1, t2)));

        result.extend(combinations.map(|(t1, t2)| ScaledMmaConfig {
            a_type: gpu::ElemType::Float(*t1).into(),
            b_type: gpu::ElemType::Float(*t2).into(),
            cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
            scales_type: gpu::ElemType::Float(gpu::FloatKind::UE8M0).into(),
            m: 16,
            n: 8,
            k: 32,
            scales_factor: 1,
        }));

        result.extend([
            ScaledMmaConfig {
                a_type: gpu::StorageType::Packed(gpu::ElemType::Float(gpu::FloatKind::E2M1), 2),
                b_type: gpu::StorageType::Packed(gpu::ElemType::Float(gpu::FloatKind::E2M1), 2),
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                scales_type: gpu::ElemType::Float(gpu::FloatKind::UE8M0).into(),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 2,
            },
            // Sign of scales is ignored
            ScaledMmaConfig {
                a_type: gpu::StorageType::Packed(gpu::ElemType::Float(gpu::FloatKind::E2M1), 2),
                b_type: gpu::StorageType::Packed(gpu::ElemType::Float(gpu::FloatKind::E2M1), 2),
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                scales_type: gpu::ElemType::Float(gpu::FloatKind::E4M3).into(),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 4,
            },
        ]);
    }
    result
}

pub fn contiguous_elements_cuda(ident: MatrixIdent, matrix: Matrix) -> u32 {
    match ident {
        MatrixIdent::A | MatrixIdent::B => (32 / matrix.storage.size_bits()) as u32,
        MatrixIdent::Accumulator => 2,
    }
}
