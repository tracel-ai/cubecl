use cubecl_core::client::ComputeClient;
use cubecl_core::{self as cubecl, Runtime};

use cubecl_core::{AutotuneKey, ir::Elem};
use serde::{Deserialize, Serialize};

use cubecl_std::tensor::{MatrixBatchLayout, matrix_batch_layout};

use super::components::{MatmulKind, MatmulProblemSize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    definition: MatmulProblemDefinition,
    pub analysis: MatmulAutotuneAnalysis,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
struct MatmulProblemDefinition {
    #[autotune(anchor)]
    biggest_axis: usize,
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
pub struct MatmulAutotuneAnalysis {
    pub scale_global: MatmulGlobalScale,
    pub may_use_tensor_cores: bool,
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

        let biggest_axis = usize::max(m, n);
        let biggest_axis = usize::max(biggest_axis, k);
        let kind = MatmulKind::from(MatmulProblemSize {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        });

        let definition = MatmulProblemDefinition::new(
            biggest_axis,
            elem_lhs,
            elem_rhs,
            elem_out,
            matrix_layout_lhs,
            matrix_layout_rhs,
        );
        let analysis = MatmulAutotuneAnalysis {
            scale_global: MatmulGlobalScale::from_size(m, n, k),
            may_use_tensor_cores: match client
                .properties()
                .hardware
                .min_tensor_cores_dim
                .map(|tc| tc as usize)
            {
                Some(tc) => m > tc && n > tc && k > tc,
                None => false,
            },
            kind,
        };

        Self::new(definition, analysis)
    }
}
