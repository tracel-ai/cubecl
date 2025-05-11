use cubecl_core::client::ComputeClient;
use cubecl_core::{self as cubecl, Runtime};

use cubecl_core::{AutotuneKey, ir::Elem};
use serde::{Deserialize, Serialize};

use crate::tensor::{MatrixBatchLayout, matrix_batch_layout};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    #[autotune(anchor)]
    m: usize,
    #[autotune(anchor)]
    n: usize,
    #[autotune(anchor)]
    k: usize,
    round: bool, // If m, k, n are not anchored.
    elem_lhs: Elem,
    elem_rhs: Elem,
    elem_out: Elem,
    matrix_layout_lhs: MatrixBatchLayout,
    matrix_layout_rhs: MatrixBatchLayout,
    pub kind: MatmulAutotuneKind,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum MatmulScale {
    Large,
    Medium,
    Small,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MatmulAutotuneKind {
    pub scale: MatmulScale,
    pub may_use_tensor_cores: bool,
    pub mat2vec: bool,
}

/// Whether it's a good idea to try and run double buffering matmul.
pub fn should_tune_double_buffering(fused: bool, key: &MatmulAutotuneKey) -> bool {
    key.kind.may_use_tensor_cores
        && !key.kind.mat2vec
        && match key.kind.scale {
            MatmulScale::Large => true,
            MatmulScale::Medium => fused,
            MatmulScale::Small => false, // TODO: maybe enable it when autotune level is set to the
                                         // max.
        }
}

impl MatmulAutotuneKey {
    /// Create the autotune key based on the shape of both lhs and rhs as well as the element type
    /// used for the calculation.
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

        // TODO: Use the number of SMs.
        let scale = if m < 512 && k < 512 && n < 512 {
            MatmulScale::Small
        } else if m < 2048 && k < 2048 && n < 2048 {
            MatmulScale::Medium
        } else {
            MatmulScale::Large
        };

        let kind = MatmulAutotuneKind {
            scale,
            may_use_tensor_cores: match client
                .properties()
                .hardware_properties()
                .min_tensor_cores_dim
                .map(|tc| tc as usize)
            {
                Some(tc) => m > tc && n > tc && k > tc,
                None => false,
            },
            mat2vec: n == 1 || m == 1 || k == 1,
        };

        let mut key = Self::new(
            m,
            n,
            k,
            false,
            elem_lhs,
            elem_rhs,
            elem_out,
            matrix_layout_lhs,
            matrix_layout_rhs,
            kind,
        );

        key.round = !(key.m == m && key.n == n && key.k == k);

        key
    }
}
