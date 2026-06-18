use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    interfaces::{ScalarType, TypedExt},
    types::scalar::*,
};
use pliron::{
    context::Context,
    r#type::{Type, TypeHandle, type_cast},
    value::Value,
};

use crate::{
    cuda::mma::manual::frag_elem,
    shared::{CppValue, ty::TypeExtCPP},
};

#[allow(clippy::too_many_arguments)]
pub fn mma_template(
    ctx: &Context,
    a_elem: TypeHandle,
    b_elem: TypeHandle,
    cd_elem: TypeHandle,
    k: usize,
    n_a_registers: usize,
    n_b_registers: usize,
    n_c_registers: usize,
    n_d_registers: usize,
) -> String {
    let a_elem = a_elem.deref(ctx);
    let b_elem = b_elem.deref(ctx);
    let cd_elem = cd_elem.deref(ctx);

    let a_ty = mma_ty(ctx, &*a_elem);
    let b_ty = mma_ty(ctx, &*b_elem);
    let cd_ty = mma_ty(ctx, &*cd_elem);

    let ab_arg_ty = match a_elem.is::<Float32Type>() {
        true => "float",
        false => "uint32_t",
    };
    let cd_arg_ty = match cd_elem.is::<Float32Type>() {
        true => "float",
        false => "uint32_t",
    };

    let args_a = (0..n_a_registers).map(|i| format!("{ab_arg_ty} const &reg_a_{i}"));
    let args_b = (0..n_b_registers).map(|i| format!("{ab_arg_ty} const &reg_b_{i}"));
    let args_c = (0..n_c_registers).map(|i| format!("{cd_arg_ty} const &reg_c_{i}"));
    let args_d = (0..n_d_registers).map(|i| format!("{cd_arg_ty} &reg_d_{i}"));
    let args = args_a
        .chain(args_b)
        .chain(args_c)
        .chain(args_d)
        .collect::<Vec<_>>()
        .join(", ");

    let kind = if is_fp6_fp4(&*a_elem) || is_fp6_fp4(&*b_elem) {
        ".kind::f8f6f4"
    } else {
        ""
    };

    let mut idx = 0usize;

    let placeholders_d = comma_separated((0..n_d_registers).map(|_| placeholder(&mut idx)));
    let placeholders_d = format!("{{{placeholders_d}}}");

    let placeholders_a = comma_separated((0..n_a_registers).map(|_| placeholder(&mut idx)));
    let placeholders_a = format!("{{{placeholders_a}}}");

    let placeholders_b = comma_separated((0..n_b_registers).map(|_| placeholder(&mut idx)));
    let placeholders_b = format!("{{{placeholders_b}}}");

    let placeholders_c = comma_separated((0..n_c_registers).map(|_| placeholder(&mut idx)));
    let placeholders_c = format!("{{{placeholders_c}}}");

    let params_out =
        comma_separated((0..n_d_registers).map(|i| as_reg(&format!("reg_d_{i}"), &*cd_elem, true)));
    let params_a = (0..n_a_registers).map(|i| as_reg(&format!("reg_a_{i}"), &*a_elem, false));
    let params_b = (0..n_b_registers).map(|i| as_reg(&format!("reg_b_{i}"), &*b_elem, false));
    let params_c = (0..n_c_registers).map(|i| as_reg(&format!("reg_c_{i}"), &*cd_elem, false));
    let params_in = comma_separated(params_a.chain(params_b).chain(params_c));

    format!(
        r#"
inline __device__ void
__mma_m16n8k{k}_{}_{}_{}({args}) {{
  asm volatile("mma.sync.aligned.m16n8k{k}.row.col{kind}.{cd_ty}.{a_ty}.{b_ty}.{cd_ty}"
               " {placeholders_d}, {placeholders_a}, {placeholders_b}, {placeholders_c};"
               : {params_out}
               : {params_in});
    }}
    "#,
        a_elem.to_cpp(ctx),
        b_elem.to_cpp(ctx),
        cd_elem.to_cpp(ctx)
    )
}

fn is_fp6_fp4(elem: &dyn Type) -> bool {
    elem.is::<Float6E2M3Type>() || elem.is::<Float6E3M2Type>() || elem.is::<Float4E2M1Type>()
}

#[allow(clippy::too_many_arguments)]
pub fn mma_scaled_template(
    ctx: &Context,
    a_elem: TypeHandle,
    b_elem: TypeHandle,
    cd_elem: TypeHandle,
    k: usize,
    n_a_registers: usize,
    n_b_registers: usize,
    n_c_registers: usize,
    n_d_registers: usize,
    scales_elem: TypeHandle,
    scales_factor: usize,
) -> String {
    let a_elem = a_elem.deref(ctx);
    let b_elem = b_elem.deref(ctx);
    let cd_elem = cd_elem.deref(ctx);
    let scales_elem = scales_elem.deref(ctx);

    let a_ty = mma_ty(ctx, &*a_elem);
    let b_ty = mma_ty(ctx, &*b_elem);
    let cd_ty = mma_ty(ctx, &*cd_elem);
    // Needs custom mapping because of the ignored sign bit
    let s_ty = if scales_elem.is::<Float8E8M0Type>() {
        "ue8m0"
    } else if scales_elem.is::<Float8E4M3Type>() {
        "ue4m3"
    } else {
        panic!("Unsupported scales type")
    };

    let kind = match scales_factor {
        1 => "mxf8f6f4",
        2 | 4 => "mxf4nvf4",
        _ => panic!("Unsupported scales factor"),
    };

    let ab_arg_ty = match a_elem.is::<Float32Type>() {
        true => "float",
        false => "uint32_t",
    };
    let cd_arg_ty = match cd_elem.is::<Float32Type>() {
        true => "float",
        false => "uint32_t",
    };

    // Note: Scaled MMA actually requires float registers for C/D, unlike normal MMA
    let args_a = (0..n_a_registers).map(|i| format!("{ab_arg_ty} const &reg_a_{i}"));
    let args_b = (0..n_b_registers).map(|i| format!("{ab_arg_ty} const &reg_b_{i}"));
    let args_c = (0..n_c_registers).map(|i| format!("{cd_arg_ty} const &reg_c_{i}"));
    let args_d = (0..n_d_registers).map(|i| format!("{cd_arg_ty} &reg_d_{i}"));
    let args = args_a
        .chain(args_b)
        .chain(args_c)
        .chain(args_d)
        .collect::<Vec<_>>()
        .join(", ");

    let mut idx = 0usize;

    let placeholders_d = comma_separated((0..n_d_registers).map(|_| placeholder(&mut idx)));
    let placeholders_d = format!("{{{placeholders_d}}}");

    let placeholders_a = comma_separated((0..n_a_registers).map(|_| placeholder(&mut idx)));
    let placeholders_a = format!("{{{placeholders_a}}}");

    let placeholders_b = comma_separated((0..n_b_registers).map(|_| placeholder(&mut idx)));
    let placeholders_b = format!("{{{placeholders_b}}}");

    let placeholders_c = comma_separated((0..n_c_registers).map(|_| placeholder(&mut idx)));
    let placeholders_c = format!("{{{placeholders_c}}}");

    let placeholder_scales_a = format!(
        "{{{}}}, {{{}, {}}}",
        placeholder(&mut idx),
        placeholder(&mut idx),
        placeholder(&mut idx)
    );
    let placeholder_scales_b = format!(
        "{{{}}}, {{{}, {}}}",
        placeholder(&mut idx),
        placeholder(&mut idx),
        placeholder(&mut idx)
    );

    let params_out =
        comma_separated((0..n_d_registers).map(|i| as_reg(&format!("reg_d_{i}"), &*cd_elem, true)));
    let params_a = (0..n_a_registers).map(|i| as_reg(&format!("reg_a_{i}"), &*a_elem, false));
    let params_b = (0..n_b_registers).map(|i| as_reg(&format!("reg_b_{i}"), &*b_elem, false));
    let params_c = (0..n_c_registers).map(|i| as_reg(&format!("reg_c_{i}"), &*cd_elem, false));
    let params_in = comma_separated(params_a.chain(params_b).chain(params_c));

    format!(
        r#"
inline __device__ void
__mma_scaled_{scales_factor}x_m16n8k{k}_{}_{}_{}({args}, uint32 const &scales_a, uint32 const &scales_b) {{
    static constexpr uint16 tidA = 0;
    static constexpr uint16 bidA = 0;
    static constexpr uint16 tidB = 0;
    static constexpr uint16 bidB = 0;

    asm volatile("mma.sync.aligned.kind::{kind}.block_scale.scale_vec::{scales_factor}X.m16n8k{k}.row.col.{cd_ty}.{a_ty}.{b_ty}.{cd_ty}.{s_ty} "
               "{placeholders_d}, {placeholders_a}, {placeholders_b}, {placeholders_c}, {placeholder_scales_a}, {placeholder_scales_b};"
               : {params_out}
               : {params_in}, "r"(scales_a), "h"(bidA), "h"(tidA), "r"(scales_b), "h"(bidB), "h"(tidB));
    }}
    "#,
        a_elem.to_cpp(ctx),
        b_elem.to_cpp(ctx),
        cd_elem.to_cpp(ctx)
    )
}

pub(crate) fn comma_separated(it: impl IntoIterator<Item = String>) -> String {
    it.into_iter().collect::<Vec<_>>().join(", ")
}

fn placeholder(idx: &mut usize) -> String {
    let placeholder = format!("%{idx}");
    *idx += 1;
    placeholder
}

fn as_reg(ident: &str, ty: &dyn Type, output: bool) -> String {
    let ty = if ty.is::<Float32Type>() {
        "f"
    } else if ty.is::<Float64Type>() {
        "d"
    } else if let Some(uint) = ty.downcast_ref::<UIntType>()
        && uint.width == 64
    {
        "l"
    } else {
        "r"
    };
    if output {
        format!(r#""={ty}"({ident})"#)
    } else {
        format!(r#""{ty}"({ident})"#)
    }
}

pub fn mma_ty(ctx: &Context, elem: &dyn Type) -> &'static str {
    let elem = type_cast::<dyn ScalarType>(elem).unwrap();
    let elem = elem.storage_type(ctx);
    match elem.elem_type() {
        ElemType::Float(kind) => match kind {
            FloatKind::E2M1 => "e2m1",
            FloatKind::E2M3 => "e2m3",
            FloatKind::E3M2 => "e3m2",
            FloatKind::E4M3 => "e4m3",
            FloatKind::E5M2 => "e5m2",
            FloatKind::UE8M0 => "ue8m0",
            FloatKind::F16 => "f16",
            FloatKind::BF16 => "bf16",
            FloatKind::Flex32 | FloatKind::F32 => "f32",
            FloatKind::TF32 => "tf32",
            FloatKind::F64 => "f64",
        },
        ElemType::Int(kind) => match kind {
            IntKind::I8 => "s8",
            IntKind::I16 => "s16",
            IntKind::I32 => "s32",
            IntKind::I64 => "s64",
        },
        ElemType::UInt(kind) => match kind {
            UIntKind::U8 => "u8",
            UIntKind::U16 => "u16",
            UIntKind::U32 => "u32",
            UIntKind::U64 => "u64",
        },
        ElemType::Bool => "b1",
    }
}

pub fn ldmatrix_call(
    ctx: &Context,
    output: Value,
    ptr: Value,
    factor: usize,
    transpose: bool,
) -> String {
    let elem = frag_elem(ctx, output);
    let width = 16 / elem.size(ctx);
    let output = output.name(ctx);
    let ptr = ptr.name(ctx);
    let elem = elem.to_cpp(ctx);
    let is_transposed = if transpose { "_trans" } else { "" };
    let regs = comma_separated(
        (0..factor).map(|i| format!("reinterpret_cast<uint32&>({output}->data[{i}])")),
    );

    format!("__ldmatrix_m{width}n8_{elem}_{factor}x{is_transposed}({regs}, {ptr});\n")
}

pub fn ldmatrix_template(
    ctx: &Context,
    elem: TypeHandle,
    factor: usize,
    transpose: bool,
) -> String {
    let width = 16 / elem.size(ctx);
    let arg_ty = "uint32_t";

    let args_regs = (0..factor).map(|i| format!("{arg_ty} &reg_{i}"));
    let arg_addr = ["void const *row_addr".to_string()];
    let args = args_regs.chain(arg_addr).collect::<Vec<_>>().join(", ");

    let mut idx = 0usize;

    let placeholders_regs = comma_separated((0..factor).map(|_| placeholder(&mut idx)));
    let placeholders_regs = format!("{{{placeholders_regs}}}");

    let placeholder_addr = format!("[{}]", placeholder(&mut idx));

    let params_regs = comma_separated((0..factor).map(|i| format!(r#""=r"(reg_{i})"#)));
    let param_addr = r#""r"(addr)"#;

    let is_transposed = if transpose { "_trans" } else { "" };
    let transposed_arg = if transpose { ".trans" } else { "" };
    let num = format!("x{factor}");

    let ty = match elem.size(ctx) {
        2 => "b16",
        1 => "b8",
        _ => unreachable!(),
    };
    let elem = elem.to_cpp(ctx);

    format!(
        r#"
inline __device__ void
__ldmatrix_m{width}n8_{elem}_{factor}x{is_transposed}({args}) {{
  uint32 addr = static_cast<uint32>(__cvta_generic_to_shared(row_addr));
  asm volatile("ldmatrix.sync.aligned.m8n{width}.{num}{transposed_arg}.shared::cta.{ty}"
               " {placeholders_regs}, {placeholder_addr};"
               : {params_regs}
               : {param_addr});
    }}
    "#
    )
}

pub fn stmatrix_call(
    ctx: &Context,
    registers: Value,
    ptr: Value,
    factor: usize,
    transpose: bool,
) -> String {
    let elem = frag_elem(ctx, registers);
    let width = 16 / elem.size(ctx);
    let is_transposed = if transpose { "_trans" } else { "" };
    let ptr = ptr.name(ctx);
    let registers = registers.name(ctx);
    let elem = elem.to_cpp(ctx);
    let regs = comma_separated(
        (0..factor).map(|i| format!("reinterpret_cast<const uint32&>({registers}->data[{i}])")),
    );

    format!("__stmatrix_m{width}n8_{elem}_{factor}x{is_transposed}({regs}, {ptr});\n")
}

pub fn stmatrix_template(
    ctx: &Context,
    elem: TypeHandle,
    factor: usize,
    transpose: bool,
) -> String {
    let width = 16 / elem.size(ctx);
    let arg_ty = "uint32_t";

    let args_regs = (0..factor).map(|i| format!("{arg_ty} const &reg_{i}"));
    let arg_addr = ["void *row_addr".to_string()];
    let args = args_regs.chain(arg_addr).collect::<Vec<_>>().join(", ");

    let mut idx = 0usize;

    let placeholder_addr = format!("[{}]", placeholder(&mut idx));

    let placeholders_regs = comma_separated((0..factor).map(|_| placeholder(&mut idx)));
    let placeholders_regs = format!("{{{placeholders_regs}}}");

    let params_regs = comma_separated((0..factor).map(|i| format!(r#""r"(reg_{i})"#)));
    let param_addr = r#""r"(addr)"#;

    let is_transposed = if transpose { "_trans" } else { "" };
    let transposed_arg = if transpose { ".trans" } else { "" };
    let num = format!("x{factor}");

    let ty = match elem.size(ctx) {
        2 => "b16",
        1 => "b8",
        _ => unreachable!(),
    };
    let elem = elem.to_cpp(ctx);

    // Note: smem technically an input
    format!(
        r#"
inline __device__ void
__stmatrix_m{width}n8_{elem}_{factor}x{is_transposed}({args}) {{
  uint32 addr = static_cast<uint32>(__cvta_generic_to_shared(row_addr));
  asm volatile("stmatrix.sync.aligned.m8n{width}.{num}{transposed_arg}.shared::cta.{ty}"
               " {placeholder_addr}, {placeholders_regs};"
               :: {param_addr}, {params_regs});
    }}
    "#
    )
}
