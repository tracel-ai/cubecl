use core::fmt::Display;

use cubecl_core::{
    self as cubecl,
    cmma::{MatrixIdent, MatrixType},
    ir::{
        ElemType, FloatKind, IntKind, StorageType, UIntKind,
        dialect::matrix::{
            ColIndexOp, LdMatrixOp, MmaManualOp, MmaManualScaledOp, RowIndexOp, StMatrixOp,
        },
        features::{MmaConfig, ScaledMmaConfig},
        interfaces::{IndexableType, TypedExt},
        prelude::*,
    },
    prelude::*,
};
use itertools::Itertools;
use pliron::r#type::TypedHandle;

use crate::{
    cuda::{
        arch::CudaArchitecture,
        cuda_op,
        ptx::{ldmatrix_call, stmatrix_call},
    },
    shared::{
        Architecture, CppValue, SupportedMmaCombinations, SupportedScaledMmaCombinations,
        lowering::LowerOp,
        ty::{PointerType, TypeExtCPP, TypedExtCPP},
    },
    target::Cuda,
};

cuda_op!(LdMatrixOp, |op, ctx| {
    let factor = op.factor(ctx).0;
    let transpose = op.transpose(ctx).0;
    ldmatrix_call(ctx, op.out_arr(ctx), op.ptr(ctx), factor, transpose)
});

cuda_op!(StMatrixOp, |op, ctx| {
    let factor = op.factor(ctx).0;
    let transpose = op.transpose(ctx).0;
    stmatrix_call(
        ctx,
        op.registers(ctx),
        op.destination(ctx),
        factor,
        transpose,
    )
});

#[op_interface_impl]
impl LowerOp<Cuda> for RowIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        let out = row_index::expand(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
        vec![out.value(scope)]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for ColIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        let out = col_index::expand(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
        vec![out.value(scope)]
    }
}

/// Derived from PTX shape documentation
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
#[cube]
fn row_index(
    lane_id: u32,
    i: u32,
    #[comptime] elems_per_reg: usize,
    #[comptime] ident: MatrixIdent,
) -> u32 {
    let elems_per_reg = elems_per_reg as u32;
    match ident {
        MatrixIdent::A => {
            let group_id = lane_id / 4;
            let odd_register = (i / elems_per_reg) & 1;
            group_id + odd_register * 8
        }
        MatrixIdent::B => {
            let thread_id_in_group = lane_id % 4;
            let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
            let reg = i / elems_per_reg;
            offset + elems_per_reg * 4 * reg
        }
        MatrixIdent::Accumulator => {
            let group_id = lane_id / 4;
            let offset = (i << 2) & 8;
            group_id + offset
        }
    }
}

/// Derived from PTX shape documentation
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
#[cube]
fn col_index(
    lane_id: u32,
    i: u32,
    #[comptime] elems_per_reg: usize,
    #[comptime] ident: MatrixIdent,
) -> u32 {
    let elems_per_reg = elems_per_reg as u32;
    match ident {
        MatrixIdent::A => {
            let thread_id_in_group = lane_id % 4;
            let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
            let group_2 = (i / (2 * elems_per_reg)) & 1;
            offset + 4 * elems_per_reg * group_2
        }
        MatrixIdent::B => lane_id >> 2,
        MatrixIdent::Accumulator => {
            let thread_id_in_group = lane_id % 4;
            (thread_id_in_group * 2) + (i % 2)
        }
    }
}

fn as_ty_idx(val: impl Display, idx: impl Display, ty: impl Display) -> String {
    format!("reinterpret_cast<{ty}*>({val})[{idx}]")
}

fn as_const_ty_idx(val: impl Display, idx: impl Display, ty: impl Display) -> String {
    format!("reinterpret_cast<const {ty}*>({val})[{idx}]")
}

cuda_op!(MmaManualOp, compile_manual_mma);

fn compile_manual_mma(op: &MmaManualOp, ctx: &Context) -> String {
    let frag_a = op.registers_a(ctx);
    let frag_b = op.registers_b(ctx);
    let frag_c = op.registers_c(ctx);
    let frag_d = op.registers_d(ctx);

    let shape = op.shape(ctx).0;

    let a_elem = frag_elem(ctx, frag_a);
    let b_elem = frag_elem(ctx, frag_b);
    let cd_elem = frag_elem(ctx, frag_c);

    let ab_ty = match a_elem.is_float32(ctx) {
        true => "float",
        false => "uint32_t",
    };
    let cd_ty = match cd_elem.is_float32(ctx) {
        true => "float",
        false => "uint32_t",
    };

    let a_elems = shape.num_elems(MatrixIdent::A) / 32;
    let b_elems = shape.num_elems(MatrixIdent::B) / 32;
    let cd_elems = shape.num_elems(MatrixIdent::Accumulator) / 32;

    let a_regs = a_elems / (32 / a_elem.unpacked_size_bits(ctx));
    let b_regs = b_elems / (32 / b_elem.unpacked_size_bits(ctx));
    let cd_regs = cd_elems / (32 / cd_elem.unpacked_size_bits(ctx));

    let frag_a = (0..a_regs).map(|i| as_const_ty_idx(frag_a.name(ctx), i, ab_ty));
    let frag_b = (0..b_regs).map(|i| as_const_ty_idx(frag_b.name(ctx), i, ab_ty));
    let frag_c = (0..cd_regs).map(|i| as_const_ty_idx(frag_c.name(ctx), i, cd_ty));
    let frag_d = (0..cd_regs).map(|i| as_ty_idx(frag_d.name(ctx), i, cd_ty));

    let args = frag_a.chain(frag_b).chain(frag_c).chain(frag_d).join(", ");
    format!(
        "__mma_m16n8k{}_{}_{}_{}({args});",
        shape.k,
        a_elem.to_cpp(ctx),
        b_elem.to_cpp(ctx),
        cd_elem.to_cpp(ctx)
    )
}

cuda_op!(MmaManualScaledOp, compile_scaled_mma);

fn compile_scaled_mma(op: &MmaManualScaledOp, ctx: &Context) -> String {
    let frag_a = op.registers_a(ctx);
    let frag_b = op.registers_b(ctx);
    let frag_c = op.registers_c(ctx);
    let frag_d = op.registers_d(ctx);

    let scales_a = op.scales_a(ctx).name(ctx);
    let scales_b = op.scales_b(ctx).name(ctx);

    let scales_factor = op.scales_factor(ctx).0;
    let shape = op.shape(ctx).0;

    let a_elem = frag_elem(ctx, frag_a);
    let b_elem = frag_elem(ctx, frag_b);
    let cd_elem = frag_elem(ctx, frag_c);

    let ab_ty = "uint32_t";
    let cd_ty = "float";

    let a_elems = shape.num_elems(MatrixIdent::A) / 32;
    let b_elems = shape.num_elems(MatrixIdent::B) / 32;
    let cd_elems = shape.num_elems(MatrixIdent::Accumulator) / 32;

    let a_regs = a_elems / (32 / a_elem.unpacked_size_bits(ctx));
    let b_regs = b_elems / (32 / b_elem.unpacked_size_bits(ctx));
    let cd_regs = cd_elems / (32 / cd_elem.unpacked_size_bits(ctx));

    let frag_a = (0..a_regs).map(|i| as_const_ty_idx(frag_a.name(ctx), i, ab_ty));
    let frag_b = (0..b_regs).map(|i| as_const_ty_idx(frag_b.name(ctx), i, ab_ty));
    let frag_c = (0..cd_regs).map(|i| as_const_ty_idx(frag_c.name(ctx), i, cd_ty));
    let frag_d = (0..cd_regs).map(|i| as_ty_idx(frag_d.name(ctx), i, cd_ty));

    let fragments = frag_a.chain(frag_b).chain(frag_c).chain(frag_d).join(", ");
    format!(
        "__mma_scaled_{scales_factor}x_m16n8k{}_{}_{}_{}({fragments}, reinterpret_cast<const uint32&>({scales_a}), reinterpret_cast<const uint32&>({scales_b}));",
        shape.k,
        a_elem.to_cpp(ctx),
        b_elem.to_cpp(ctx),
        cd_elem.to_cpp(ctx)
    )
}

pub(crate) fn frag_elem(ctx: &Context, frag: impl Typed) -> TypeHandle {
    let ty = frag.get_type(ctx).deref(ctx);
    let PointerType { inner, .. } = ty.downcast_ref().expect("Should be ptr");
    let inner = inner.deref(ctx);
    let indexable = type_cast::<dyn IndexableType>(&*inner).expect("Should be array");
    indexable.indexed_type(ctx)
}

pub fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
    let mut result: SupportedMmaCombinations = vec![];
    // Higher than WMMA because we only support the newest shapes. Other shapes would make things
    // very complicated.
    // Also only use f32 accumulators for now
    if arch.get_version() >= 80 {
        result.extend([
            MmaConfig {
                a_type: ElemType::Float(FloatKind::F16).into(),  // a
                b_type: ElemType::Float(FloatKind::F16).into(),  // b
                cd_type: ElemType::Float(FloatKind::F32).into(), // cd
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::BF16).into(),
                b_type: ElemType::Float(FloatKind::BF16).into(),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::TF32).into(),
                b_type: ElemType::Float(FloatKind::TF32).into(),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 16,
                n: 8,
                k: 8,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8).into(),
                b_type: ElemType::Int(IntKind::I8).into(),
                cd_type: ElemType::Int(IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8).into(),
                b_type: ElemType::UInt(UIntKind::U8).into(),
                cd_type: ElemType::Int(IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8).into(),
                b_type: ElemType::UInt(UIntKind::U8).into(),
                cd_type: ElemType::Int(IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8).into(),
                b_type: ElemType::Int(IntKind::I8).into(),
                cd_type: ElemType::Int(IntKind::I32).into(),
                m: 16,
                n: 8,
                k: 32,
            },
            // TODO: u4/i4/b1, there's no types for them yet
        ]);
    }
    if arch.get_version() >= 89 {
        let f8f6f4_types = [
            FloatKind::E4M3,
            FloatKind::E5M2,
            FloatKind::E3M2,
            FloatKind::E2M3,
            FloatKind::E2M1,
        ];
        let combinations = f8f6f4_types.iter().cartesian_product(f8f6f4_types.iter());
        result.extend(combinations.map(|(t1, t2)| MmaConfig {
            a_type: ElemType::Float(*t1).into(),
            b_type: ElemType::Float(*t2).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 16,
            n: 8,
            k: 32,
        }));
    }
    // Warning: this likely does not follow the same layout pattern as those after 80
    if arch.get_version() >= 70 && arch.get_version() < 80 {
        result.push(MmaConfig {
            a_type: ElemType::Float(FloatKind::F16).into(),
            b_type: ElemType::Float(FloatKind::F16).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 16,
            n: 8,
            k: 8,
        });
    }
    result
}

pub fn supported_scaled_mma_combinations(
    arch: &CudaArchitecture,
) -> SupportedScaledMmaCombinations {
    let mut result: SupportedScaledMmaCombinations = vec![];
    // sm_120f
    if arch.get_version() >= 120 && arch.get_version() < 130 {
        let f8f6f4_types = [
            FloatKind::E4M3,
            FloatKind::E5M2,
            FloatKind::E3M2,
            FloatKind::E2M3,
            FloatKind::E2M1,
        ];
        let combinations = f8f6f4_types
            .iter()
            .flat_map(|t1| f8f6f4_types.iter().map(move |t2| (t1, t2)));

        result.extend(combinations.map(|(t1, t2)| ScaledMmaConfig {
            a_type: ElemType::Float(*t1).into(),
            b_type: ElemType::Float(*t2).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            scales_type: ElemType::Float(FloatKind::UE8M0).into(),
            m: 16,
            n: 8,
            k: 32,
            scales_factor: 1,
        }));

        result.extend([
            ScaledMmaConfig {
                a_type: StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                b_type: StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                scales_type: ElemType::Float(FloatKind::UE8M0).into(),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 2,
            },
            // Sign of scales is ignored
            ScaledMmaConfig {
                a_type: StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                b_type: StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                scales_type: ElemType::Float(FloatKind::E4M3).into(),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 4,
            },
        ]);
    }
    result
}

pub fn contiguous_elements_cuda(
    ctx: &Context,
    ident: MatrixIdent,
    matrix: TypedHandle<MatrixType>,
) -> usize {
    let elem = matrix.deref(ctx).elem_ty;
    let size_bits = elem.size(ctx) * 8 / elem.packing_factor(ctx);
    match ident {
        MatrixIdent::A | MatrixIdent::B => 32 / size_bits,
        MatrixIdent::Accumulator => 2,
    }
}
