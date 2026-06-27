use cubecl_core::{
    self as cubecl,
    cmma::{MatrixIdent, MatrixType},
    ir::{
        ElemType, FloatKind, IntKind, UIntKind,
        dialect::matrix::{ColIndexOp, RowIndexOp},
        features::{MmaConfig, ScaledMmaConfig},
        interfaces::{IndexableType, TypedExt},
        prelude::*,
        types::PointerType,
    },
    prelude::*,
};
use itertools::Itertools;
use pliron::r#type::TypedHandle;

use crate::{
    cuda::arch::CudaArchitecture,
    shared::{
        Architecture, SupportedMmaCombinations, SupportedScaledMmaCombinations, lowering::LowerOp,
    },
    target::Cuda,
};

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
                a_type: ElemType::Float(FloatKind::F16),  // a
                b_type: ElemType::Float(FloatKind::F16),  // b
                cd_type: ElemType::Float(FloatKind::F32), // cd
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::BF16),
                b_type: ElemType::Float(FloatKind::BF16),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::TF32),
                b_type: ElemType::Float(FloatKind::TF32),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 8,
                k: 8,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8),
                b_type: ElemType::Int(IntKind::I8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8),
                b_type: ElemType::UInt(UIntKind::U8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8),
                b_type: ElemType::UInt(UIntKind::U8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8),
                b_type: ElemType::Int(IntKind::I8),
                cd_type: ElemType::Int(IntKind::I32),
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
            a_type: ElemType::Float(*t1),
            b_type: ElemType::Float(*t2),
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 32,
        }));
    }
    // Warning: this likely does not follow the same layout pattern as those after 80
    if arch.get_version() >= 70 && arch.get_version() < 80 {
        result.push(MmaConfig {
            a_type: ElemType::Float(FloatKind::F16),
            b_type: ElemType::Float(FloatKind::F16),
            cd_type: ElemType::Float(FloatKind::F32),
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
            a_type: ElemType::Float(*t1),
            b_type: ElemType::Float(*t2),
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(FloatKind::UE8M0),
            m: 16,
            n: 8,
            k: 32,
            scales_factor: 1,
        }));

        result.extend([
            ScaledMmaConfig {
                a_type: ElemType::Float(FloatKind::E2M1x2),
                b_type: ElemType::Float(FloatKind::E2M1x2),
                cd_type: ElemType::Float(FloatKind::F32),
                scales_type: ElemType::Float(FloatKind::UE8M0),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 2,
            },
            // Sign of scales is ignored
            ScaledMmaConfig {
                a_type: ElemType::Float(FloatKind::E2M1x2),
                b_type: ElemType::Float(FloatKind::E2M1x2),
                cd_type: ElemType::Float(FloatKind::F32),
                scales_type: ElemType::Float(FloatKind::E4M3),
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
    match ident {
        MatrixIdent::A | MatrixIdent::B => 32 / elem.size_bits(ctx),
        MatrixIdent::Accumulator => 2,
    }
}
