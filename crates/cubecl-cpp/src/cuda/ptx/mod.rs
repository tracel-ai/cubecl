use crate::{
    Dialect,
    shared::{Elem, FP4Kind, FP6Kind, FP8Kind},
};

pub const TMA_LOAD_IM2COL: &str = include_str!("tma_load_im2col.cuh");

pub fn mma_template<D: Dialect>(
    ab_elem: Elem<D>,
    cd_elem: Elem<D>,
    k: u32,
    n_a_registers: usize,
    n_b_registers: usize,
    n_c_registers: usize,
    n_d_registers: usize,
) -> String {
    let ab_ty = mma_ty(ab_elem);
    let cd_ty = mma_ty(cd_elem);

    let args_a = (0..n_a_registers).map(|i| format!("uint32 const &reg_a_{i}"));
    let args_b = (0..n_b_registers).map(|i| format!("uint32 const &reg_b_{i}"));
    let args_c = (0..n_c_registers).map(|i| format!("uint32 const &reg_c_{i}"));
    let args_d = (0..n_d_registers).map(|i| format!("uint32 &reg_d_{i}"));
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

    let params_out =
        comma_separated((0..n_d_registers).map(|i| as_reg(&format!("reg_d_{i}"), cd_elem, true)));
    let params_a = (0..n_a_registers).map(|i| as_reg(&format!("reg_a_{i}"), ab_elem, false));
    let params_b = (0..n_b_registers).map(|i| as_reg(&format!("reg_b_{i}"), ab_elem, false));
    let params_c = (0..n_c_registers).map(|i| as_reg(&format!("reg_c_{i}"), cd_elem, false));
    let params_in = comma_separated(params_a.chain(params_b).chain(params_c));

    format!(
        r#"
inline __device__ void
__mma_m16n8k{k}_{ab_elem}_{cd_elem}({args}) {{
  asm volatile("mma.sync.aligned.m16n8k{k}.row.col.{cd_ty}.{ab_ty}.{ab_ty}.{cd_ty}"
               " {placeholders_d}, {placeholders_a}, {placeholders_b}, {placeholders_c};"
               : {params_out}
               : {params_in});
    }}
    "#
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

fn as_reg<D: Dialect>(ident: &str, ty: Elem<D>, output: bool) -> String {
    let ty = match ty {
        // TODO: Check if this impacts performance
        // Elem::F32 => "f",
        Elem::F64 => "d",
        Elem::U64 => "l",
        _ => "r",
    };
    if output {
        format!(r#""={ty}"({ident})"#)
    } else {
        format!(r#""{ty}"({ident})"#)
    }
}

fn mma_ty<D: Dialect>(elem: Elem<D>) -> &'static str {
    match elem {
        Elem::TF32 => "tf32",
        Elem::F32 => "f32",
        Elem::F64 => "f64",
        Elem::F16 => "f16",
        Elem::BF16 => "bf16",
        Elem::FP4(FP4Kind::E2M1) => "e2m1",
        Elem::FP6(FP6Kind::E2M3) => "e2m3",
        Elem::FP6(FP6Kind::E3M2) => "e3m2",
        Elem::FP8(FP8Kind::E4M3) => "e4m3",
        Elem::FP8(FP8Kind::E5M2) => "e5m2",
        Elem::FP8(FP8Kind::UE8M0) => "ue8m0",
        Elem::I8 => "s8",
        Elem::I16 => "s16",
        Elem::I32 => "s32",
        Elem::I64 => "s64",
        Elem::U8 => "u8",
        Elem::U16 => "u16",
        Elem::U32 => "u32",
        Elem::U64 => "u64",
        Elem::Bool => "b1",
        other => panic!("{other} not supported for MMA"),
    }
}
