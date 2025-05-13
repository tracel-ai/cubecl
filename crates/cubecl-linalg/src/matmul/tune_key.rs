use cubecl_core::client::ComputeClient;
use cubecl_core::{self as cubecl, Runtime};

use cubecl_core::{AutotuneKey, ir::Elem};
use serde::{Deserialize, Serialize};

use crate::tensor::{MatrixBatchLayout, matrix_batch_layout};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    definition: MatmulProblemDefinition,
    pub analysis: MatmulAutotuneAnalysis,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
struct MatmulProblemDefinition {
    #[autotune(anchor)]
    m: usize,
    #[autotune(anchor)]
    n: usize,
    #[autotune(anchor)]
    k: usize,
    elem_lhs: Elem,
    elem_rhs: Elem,
    elem_out: Elem,
    matrix_layout_lhs: MatrixBatchLayout,
    matrix_layout_rhs: MatrixBatchLayout,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulGlobalScale {
    Large,
    Medium,
    Small,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// The potential scale of a stage matmul.
///
/// # Notes
///
/// The values are powers of 2.
///
/// ```rust, ignore
/// let state_size_m = 2.pow(m);
/// ```
pub struct MatmulStageScale {
    pub m: u8,
    pub n: u8,
    pub k: u8,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MatmulAutotuneAnalysis {
    pub scale_global: MatmulGlobalScale,
    pub stage_stage: MatmulStageScale,
    pub may_use_tensor_cores: bool,
    pub kind: MatmulKind,
}

/// Interpretation of matrix multiplication based on input shapes.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulKind {
    /// (M, K) @ (K, N) → (M, N), with M, K, N > 1
    General,

    /// (M, K) @ (K, 1) → (M, 1)
    MatVec,

    /// (1, K) @ (K, N) → (1, N)
    VecMat,

    /// (1, K) @ (K, 1) → (1, 1)
    InnerProduct,

    /// (M, 1) @ (1, N) → (M, N)
    OuterProduct,
}

impl MatmulKind {
    pub fn from_size(m: usize, n: usize, k: usize) -> Self {
        if m > 1 && k > 1 && n > 1 {
            MatmulKind::General
        } else if m > 1 && k > 1 && n == 1 {
            MatmulKind::MatVec
        } else if m == 1 && k > 1 && n > 1 {
            MatmulKind::VecMat
        } else if m == 1 && k > 1 && n == 1 {
            MatmulKind::InnerProduct
        } else if m > 1 && k == 1 && n > 1 {
            MatmulKind::OuterProduct
        } else {
            unreachable!(
                "Invalid dimensions for matrix multiplication: m={}, k={}, n={}",
                m, k, n
            )
        }
    }
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

impl MatmulStageScale {
    pub fn from_size(m_size: usize, n_size: usize, k_size: usize) -> Self {
        let mut m = 0;
        let mut n = 0;
        let mut k = 0;

        let set_scale = |current: u8, size: usize, maybe_updated: u8| {
            if current != 0 && size % 2usize.pow(maybe_updated as u32) == 0 {
                maybe_updated
            } else {
                current
            }
        };

        // Right now we only consider potential stage sizes of 64 (2^6) & 256 (2^8).
        // But we could include more in the analysis based on the autotune level.
        //
        // It is important to start with the biggest stage size first.
        for scale in [8, 6] {
            m = set_scale(m, m_size, scale);
            n = set_scale(n, n_size, scale);
            k = set_scale(k, k_size, scale);
        }

        Self { m, n, k }
    }
}

/// Whether it's a good idea to try and run double-buffered matmul.
pub fn should_tune_double_buffering(fused: bool, key: &MatmulAutotuneKey) -> bool {
    key.analysis.may_use_tensor_cores
        && matches!(key.analysis.kind, MatmulKind::General)
        && match key.analysis.scale_global {
            MatmulGlobalScale::Large => true,
            MatmulGlobalScale::Medium => true,
            MatmulGlobalScale::Small => fused,
        }
}

impl MatmulAutotuneKey {
    /// Create the autotune key based on the shape of both lhs and rhs as well as the element type
    /// used for the calculation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_strides: &[usize],
        elem_lhs: Elem,
        elem_rhs: Elem,
        elem_out: Elem,
    ) -> MatmulAutotuneKey {
        let ndims = lhs_shape.len();
        let m = lhs_shape[ndims - 2];
        let k = lhs_shape[ndims - 1];
        let n = rhs_shape[ndims - 1];

        let matrix_layout_lhs = matrix_batch_layout(lhs_strides);
        let matrix_layout_rhs = matrix_batch_layout(rhs_strides);

        let definition = MatmulProblemDefinition::new(
            m,
            n,
            k,
            elem_lhs,
            elem_rhs,
            elem_out,
            matrix_layout_lhs,
            matrix_layout_rhs,
        );
        let analysis = MatmulAutotuneAnalysis {
            scale_global: MatmulGlobalScale::from_size(m, n, k),
            stage_stage: MatmulStageScale::from_size(m, n, k),
            may_use_tensor_cores: match client
                .properties()
                .hardware
                .min_tensor_cores_dim
                .map(|tc| tc as usize)
            {
                Some(tc) => m > tc && n > tc && k > tc,
                None => false,
            },
            kind: MatmulKind::from_size(m, n, k),
        };

        Self::new(definition, analysis)
    }
}
