use cubecl_core::client::ComputeClient;
use cubecl_core::{self as cubecl, Runtime};

use cubecl_core::{AutotuneKey, ir::ElemType};
use serde::{Deserialize, Serialize};

use cubecl_std::tensor::{MatrixBatchLayout, matrix_batch_layout};

use super::components::{MatmulKind, MatmulProblemSize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    pub definition: MatmulProblemDefinition,
    pub analysis: MatmulAutotuneAnalysis,
}

/// Maximum factor relevant for strides. Currently set to 2^5, or 32 since that's the maximum align
/// relevant for CUDA (for interleaved tensors). This can be changed if other platforms or features
/// require more.
const MAX_STRIDE_FACTOR: u32 = 5;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MatmulProblemDefinition {
    #[autotune(anchor)]
    pub m: usize,
    #[autotune(anchor)]
    pub n: usize,
    #[autotune(anchor)]
    pub k: usize,
    pub lhs_pow2_factor: u8,
    /// Power of two that lhs strides are aligned to
    pub lhs_stride_factor: u8,
    pub rhs_pow2_factor: u8,
    /// Power of two that rhs strides are aligned to
    pub rhs_stride_factor: u8,
    pub elem_lhs: MatmulElemType,
    pub elem_rhs: MatmulElemType,
    pub elem_out: MatmulElemType,
    pub matrix_layout_lhs: MatrixBatchLayout,
    pub matrix_layout_rhs: MatrixBatchLayout,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulGlobalScale {
    Large,
    Medium,
    Small,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MatmulAutotuneAnalysis {
    pub scale_global: MatmulGlobalScale,
    pub kind: MatmulKind,
}

impl MatmulGlobalScale {
    pub fn from_size(m: usize, n: usize, k: usize) -> Self {
        if m < 512 && k < 512 && n < 512 {
            MatmulGlobalScale::Small
        } else if m < 2048 && k < 2048 && n < 2048 {
            MatmulGlobalScale::Medium
        } else {
            MatmulGlobalScale::Large
        }
    }
}

/// Whether it's a good idea to try and run double-buffered matmul.
pub fn should_tune_double_buffering(fused: bool, key: &MatmulAutotuneKey) -> bool {
    matches!(key.analysis.kind, MatmulKind::General)
        && match key.analysis.scale_global {
            MatmulGlobalScale::Large => true,
            MatmulGlobalScale::Medium => true,
            MatmulGlobalScale::Small => fused,
        }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MatmulElemType {
    pub elem: ElemType,
    pub quantized: bool,
}

impl MatmulAutotuneKey {
    /// Create the autotune key based on the shape of both lhs and rhs as well as the element type
    /// used for the calculation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<R: Runtime>(
        _client: &ComputeClient<R::Server>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        elem_lhs: MatmulElemType,
        elem_rhs: MatmulElemType,
        elem_out: MatmulElemType,
    ) -> MatmulAutotuneKey {
        let ndims = lhs_shape.len();
        let m = lhs_shape[ndims - 2];
        let k = lhs_shape[ndims - 1];
        let n = rhs_shape[ndims - 1];

        let matrix_layout_lhs = matrix_batch_layout(lhs_strides);
        let matrix_layout_rhs = matrix_batch_layout(rhs_strides);

        let kind = MatmulKind::from(MatmulProblemSize {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        });

        let lhs_pow2_factor = match matrix_layout_lhs {
            MatrixBatchLayout::Contiguous => pow2_factor(k),
            MatrixBatchLayout::MildlyPermuted { transposed, .. } => match transposed {
                true => pow2_factor(m),
                false => pow2_factor(k),
            },
            MatrixBatchLayout::HighlyPermuted => 0,
        };
        let rhs_pow2_factor = match matrix_layout_rhs {
            MatrixBatchLayout::Contiguous => pow2_factor(n),
            MatrixBatchLayout::MildlyPermuted { transposed, .. } => match transposed {
                true => pow2_factor(k),
                false => pow2_factor(n),
            },
            MatrixBatchLayout::HighlyPermuted => 0,
        };

        let lhs_stride_factor = match matrix_layout_lhs {
            MatrixBatchLayout::Contiguous => stride_align(lhs_strides, ndims - 1, elem_lhs.elem),
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => stride_align(lhs_strides, ndims - 2, elem_lhs.elem),
            _ => 0,
        };
        let rhs_stride_factor = match matrix_layout_rhs {
            MatrixBatchLayout::Contiguous => stride_align(rhs_strides, ndims - 1, elem_rhs.elem),
            // TMA can't handle discontiguous batches because they're all combined into one dim
            MatrixBatchLayout::MildlyPermuted {
                transposed: true,
                batch_swap: false,
            } => stride_align(rhs_strides, ndims - 2, elem_rhs.elem),
            _ => 0,
        };

        let definition = MatmulProblemDefinition::new(
            m,
            n,
            k,
            lhs_pow2_factor,
            lhs_stride_factor,
            rhs_pow2_factor,
            rhs_stride_factor,
            elem_lhs,
            elem_rhs,
            elem_out,
            matrix_layout_lhs,
            matrix_layout_rhs,
        );
        let analysis = MatmulAutotuneAnalysis {
            scale_global: MatmulGlobalScale::from_size(m, n, k),
            kind,
        };

        Self::new(definition, analysis)
    }
}

/// Defines the non-contiguous stride alignment in terms of powers of two
fn stride_align(strides: &[usize], exclude_dim: usize, elem: ElemType) -> u8 {
    let max = MAX_STRIDE_FACTOR;
    let factor = strides
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_dim)
        .map(|(_, it)| (*it * elem.size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(max);
    factor.min(max) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}
