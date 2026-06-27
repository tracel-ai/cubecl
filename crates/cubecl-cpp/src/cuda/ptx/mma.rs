use cubecl_core::{
    self as cubecl,
    cmma::{MatrixIdent, MatrixShape},
    ir::{
        AddressType, ContextExt, ElemType, FloatKind, IntKind, UIntKind,
        dialect::matrix::{LdMatrixOp, MmaManualOp, MmaManualScaledOp, StMatrixOp},
        interfaces::{ScalarType, TypedExt},
        types::{VectorType, scalar::UIntType},
    },
    prelude::*,
};
use pliron::{
    context::Context,
    derive::op_interface_impl,
    r#type::{Type, Typed, type_cast},
    value::Value,
};

use crate::{
    cuda::ptx::generic_to_shared,
    shared::{lowering::LowerOp, ty::TypedExtCPP},
    target::Cuda,
};

// Types
define_scalar!(A);
define_scalar!(B);
define_scalar!(CD);
define_scalar!(S);

// Regs per thread
define_size!(NA);
define_size!(NB);
define_size!(NCD);
define_size!(NS);

define_scalar!(RegAB);
define_scalar!(RegCD);

#[cube]
fn mma(
    frag_a: Vector<RegAB, NA>,
    frag_b: Vector<RegAB, NB>,
    frag_c: Vector<RegCD, NCD>,
    #[comptime] k: usize,
    #[comptime] kind: &str,
) -> Vector<RegCD, NCD> {
    let a_ty = ptx_mma_ty::<A>().comptime();
    let b_ty = ptx_mma_ty::<B>().comptime();
    let cd_ty = ptx_mma_ty::<CD>().comptime();

    let out: Vector<RegCD, NCD>;
    gpu_asm!(
        "mma.sync.aligned.m16n8k{k}.row.col{kind}.{cd_ty}.{a_ty}.{b_ty}.{cd_ty} {d}, {a}, {b}, {c};",
        a = in(_) frag_a, b = in(_) frag_b, c = in(_) frag_c, d = out(_) out,
        options(nomem),
    );
    out
}

#[cube]
fn scaled_mma(
    frag_a: Vector<RegAB, NA>,
    frag_b: Vector<RegAB, NB>,
    frag_c: Vector<RegCD, NCD>,
    scale_a: Vector<S, NS>,
    scale_b: Vector<S, NS>,
    #[comptime] k: usize,
    #[comptime] scales_factor: usize,
) -> Vector<RegCD, NCD> {
    let a_ty = ptx_mma_ty::<A>().comptime();
    let b_ty = ptx_mma_ty::<B>().comptime();
    let cd_ty = ptx_mma_ty::<CD>().comptime();
    let scale_ty = ptx_scale_ty::<S>().comptime();

    let kind = comptime![match scales_factor {
        1 => "mxf8f6f4",
        2 | 4 => "mxf4nvf4",
        _ => unreachable!(),
    }];

    let out: Vector<RegCD, NCD>;
    gpu_asm!(
        "mma.sync.aligned.m16n8k{k}.row.col.kind::{kind}.block_scale.scale_vec::{scales_factor}X",
        ".{cd_ty}.{a_ty}.{b_ty}.{cd_ty}.{scale_ty} {d}, {a}, {b}, {c}, ",
        "{scale_a}, {{0, 0}}, {scale_b}, {{0, 0}};",
        a = in(_) frag_a, b = in(_) frag_b, c = in(_) frag_c, d = out(_) out,
        scale_a = in(_) u32::reinterpret(scale_a), scale_b = in(_) u32::reinterpret(scale_b),
        options(nomem),
    );
    out
}

#[cube]
fn ldmatrix(
    row: *const u32,
    #[comptime] num: usize,
    #[comptime] transpose: &str,
) -> Vector<u32, NCD> {
    let row_addr = generic_to_shared::<u32>(row);

    let out: Vector<u32, NCD>;
    gpu_asm!(
        "ldmatrix.sync.aligned.m8n8.x{num}{transpose}.shared::cta.b16 {out}, [{addr}];",
        out = out(_) out, addr = in(_) row_addr, options(readonly),
    );
    out
}

#[cube]
fn stmatrix(
    value: Vector<u32, NCD>,
    row: *const u32,
    #[comptime] num: usize,
    #[comptime] transpose: &str,
) {
    let row_addr = generic_to_shared::<u32>(row);

    gpu_asm!(
        "stmatrix.sync.aligned.m8n8.x{num}{transpose}.shared::cta.b16 [{addr}], {val};",
        addr = in(_) row_addr, val = in(_) value,
    );
}

#[op_interface_impl]
impl LowerOp<Cuda> for MmaManualOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let frag_a = self.registers_a(ctx);
        let frag_b = self.registers_b(ctx);
        let frag_c = self.registers_c(ctx);
        let frag_d = self.registers_d(ctx);
        let shape = self.shape(ctx).0;

        let kind = if frag_a.element_ty(ctx).is_fp8_fp6_fp4(ctx)
            || frag_b.element_ty(ctx).is_fp8_fp6_fp4(ctx)
        {
            ".kind::f8f6f4"
        } else {
            ""
        };

        let (frag_a, frag_b, frag_c) = frags_as_vectors(scope, shape, frag_a, frag_b, frag_c);

        let frag_out = mma::expand(
            scope,
            frag_a.into(),
            frag_b.into(),
            frag_c.into(),
            shape.k,
            kind,
        );
        let frag_out = reinterpret_value(scope, frag_out.read_value(scope), frag_d.unwrap_ptr(ctx));
        assign::expand_element(scope, frag_out.into(), frag_d.into());
        vec![]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for MmaManualScaledOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let frag_a = self.registers_a(ctx);
        let frag_b = self.registers_b(ctx);
        let frag_c = self.registers_c(ctx);
        let frag_d = self.registers_d(ctx);
        let scales_a = self.scales_a(ctx);
        let scales_b = self.scales_b(ctx);
        let scales_factor = self.scales_factor(ctx).0;
        let shape = self.shape(ctx).0;

        scope.register_value_type::<S, NS>(scales_a);

        let (frag_a, frag_b, frag_c) = frags_as_vectors(scope, shape, frag_a, frag_b, frag_c);

        let frag_out = scaled_mma::expand(
            scope,
            frag_a.into(),
            frag_b.into(),
            frag_c.into(),
            scales_a.into(),
            scales_b.into(),
            shape.k,
            scales_factor,
        );
        let frag_out = reinterpret_value(scope, frag_out.read_value(scope), frag_d.unwrap_ptr(ctx));
        assign::expand_element(scope, frag_out.into(), frag_d.into());
        vec![]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for LdMatrixOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let row_ptr = self.ptr(ctx).into();
        let out_arr = self.out_arr(ctx);
        let factor = self.factor(ctx).0;
        let trans = if self.transpose(ctx).0 { ".trans" } else { "" };

        scope.register_size::<NCD>(factor);

        let frag_out = ldmatrix::expand(scope, &row_ptr, factor, trans);
        let frag_out =
            reinterpret_value(scope, frag_out.read_value(scope), out_arr.unwrap_ptr(ctx));
        assign::expand_element(scope, frag_out.into(), out_arr.into());
        vec![]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for StMatrixOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let row_ptr = self.destination(ctx).into();
        let factor = self.factor(ctx).0;
        let trans = if self.transpose(ctx).0 { ".trans" } else { "" };

        let vec_ty = VectorType::get(ctx, UIntType::get(ctx, 32).to_handle(), factor);
        let value = reinterpret_value(scope, self.registers(ctx), vec_ty.to_handle()).into();
        scope.register_size::<NCD>(factor);

        stmatrix::expand(scope, value, &row_ptr, factor, trans);
        vec![]
    }
}

fn num_elems(shape: MatrixShape) -> (usize, usize, usize) {
    let a_elems = shape.num_elems(MatrixIdent::A) / 32;
    let b_elems = shape.num_elems(MatrixIdent::B) / 32;
    let c_elems = shape.num_elems(MatrixIdent::Accumulator) / 32;
    (a_elems, b_elems, c_elems)
}

fn num_regs(
    ctx: &Context,
    shape: MatrixShape,
    frag_a: Value,
    frag_b: Value,
    frag_c: Value,
) -> (usize, usize, usize) {
    let (a_elems, b_elems, c_elems) = num_elems(shape);
    let a_regs = a_elems / (32 / frag_a.unpacked_size_bits(ctx));
    let b_regs = b_elems / (32 / frag_b.unpacked_size_bits(ctx));
    let c_regs = c_elems / (32 / frag_c.unpacked_size_bits(ctx));
    (a_regs, b_regs, c_regs)
}

fn frags_as_vectors(
    scope: &Scope,
    shape: MatrixShape,
    frag_a: Value,
    frag_b: Value,
    frag_c: Value,
) -> (Value, Value, Value) {
    let ctx = scope.ctx();

    scope.register_value_type::<A, ()>(frag_a.element_ty(ctx));
    scope.register_value_type::<B, ()>(frag_b.element_ty(ctx));
    scope.register_value_type::<CD, ()>(frag_c.element_ty(ctx));

    let (a_regs, b_regs, c_regs) = num_regs(ctx, shape, frag_a, frag_b, frag_c);

    scope.register_size::<NA>(a_regs);
    scope.register_size::<NB>(b_regs);
    scope.register_size::<NCD>(c_regs);

    let reg_ty_ab = reg_ty(ctx, frag_a).to_type(ctx);
    let reg_ty_cd = reg_ty(ctx, frag_c).to_type(ctx);

    scope.register_type::<RegAB>(reg_ty(ctx, frag_a));
    scope.register_type::<RegCD>(reg_ty(ctx, frag_c));

    let ty_a = VectorType::get(ctx, reg_ty_ab, a_regs).to_handle();
    let ty_b = VectorType::get(ctx, reg_ty_ab, b_regs).to_handle();
    let ty_c = VectorType::get(ctx, reg_ty_cd, c_regs).to_handle();

    let frag_a = reinterpret_value(scope, frag_a, ty_a);
    let frag_b = reinterpret_value(scope, frag_b, ty_b);
    let frag_c = reinterpret_value(scope, frag_c, ty_c);

    (frag_a, frag_b, frag_c)
}

fn reg_ty(ctx: &Context, frag: impl Typed) -> ElemType {
    if frag.element_ty(ctx).scalar_ty(ctx).is_float32(ctx) {
        FloatKind::F32.into()
    } else {
        UIntKind::U32.into()
    }
}

#[cube]
pub fn ptx_mma_ty<T: Scalar>() -> comptime_type!(&'static str) {
    intrinsic!(|scope| {
        match T::elem_type(scope) {
            ElemType::Index => match scope.ctx().address_type() {
                AddressType::U32 => "u32",
                AddressType::U64 => "u64",
            },
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => "e2m1",
                FloatKind::E2M1x2 => "e2m1",
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
    })
}

#[cube]
pub fn ptx_scale_ty<T: Scalar>() -> comptime_type!(&'static str) {
    intrinsic!(|scope| {
        match T::elem_type(scope) {
            ElemType::Float(FloatKind::UE8M0) => "ue8m0",
            ElemType::Float(FloatKind::E4M3) => "ue4m3",
            _ => panic!("Unsupported scales type"),
        }
    })
}

pub fn mma_ty(ctx: &Context, elem: &dyn Type) -> &'static str {
    let elem = type_cast::<dyn ScalarType>(elem).unwrap();
    match elem.elem_type(ctx) {
        ElemType::Index => match ctx.address_type() {
            AddressType::U32 => "u32",
            AddressType::U64 => "u64",
        },
        ElemType::Float(kind) => match kind {
            FloatKind::E2M1 => "e2m1",
            FloatKind::E2M1x2 => "e2m1",
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
