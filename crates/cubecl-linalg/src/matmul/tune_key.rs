use std::cmp::max;

use cubecl_core as cubecl;

use cubecl_core::{ir::Elem, AutotuneKey};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct MatmulAutotuneKey {
    round: bool,     // True when all matmul dims are multiples of 64
    broadcast: bool, // True when there are differences in batch size
    #[autotune(anchor)]
    m: usize,
    #[autotune(anchor)]
    k: usize,
    #[autotune(anchor)]
    n: usize,
    #[autotune(anchor(max = 256))]
    batch: usize,
    elem: Elem,
}

impl MatmulAutotuneKey {
    /// Create the autotune key based on the shape of both lhs and rhs as well as the element type
    /// used for the calculation.
    pub fn from_shape(lhs_shape: &[usize], rhs_shape: &[usize], elem: Elem) -> Self {
        let ndims = lhs_shape.len();
        let m = lhs_shape[ndims - 2];
        let k = lhs_shape[ndims - 1];
        let n = rhs_shape[ndims - 1];

        let mut broadcast = false;
        let mut batch_product_lhs = 1;
        let mut batch_product_rhs = 1;

        for b in 0..ndims - 2 {
            batch_product_lhs *= lhs_shape[b];
            batch_product_rhs *= rhs_shape[b];
            if lhs_shape[b] != rhs_shape[b] {
                broadcast = true;
            }
        }
        let batch_product = max(batch_product_lhs, batch_product_rhs);

        let round = m % 64 == 0 && k % 64 == 0 && n % 64 == 0;

        Self::new(round, broadcast, m, k, n, batch_product, elem)
    }
}

#[cfg(test)]
mod tests {
    use cubecl_core::ir::FloatKind;

    use super::*;

    #[test]
    fn matmul_autotune_key_all_same_and_round() {
        let lhs_shape = [4, 512, 512];
        let rhs_shape = [4, 512, 512];
        let key =
            MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, Elem::Float(FloatKind::F32));

        assert!(key.round);
        assert!(!key.broadcast);
        assert_eq!(key.m, 512);
        assert_eq!(key.k, 512);
        assert_eq!(key.n, 512);
    }

    #[test]
    fn matmul_autotune_key_all_different() {
        let lhs_shape = [2, 3, 511, 512];
        let rhs_shape = [3, 2, 512, 513];
        let key =
            MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, Elem::Float(FloatKind::F32));

        assert!(!key.round);
        assert!(key.broadcast);
        assert_eq!(key.m, 512);
        assert_eq!(key.k, 512);
        assert_eq!(key.n, 1024);
        assert_eq!(key.batch, 8);
    }

    #[test]
    fn matmul_autotune_key_large_batch() {
        let lhs_shape = [128, 512, 511, 512];
        let rhs_shape = [200, 400, 512, 513];
        let key =
            MatmulAutotuneKey::from_shape(&lhs_shape, &rhs_shape, Elem::Float(FloatKind::F32));

        assert_eq!(key.batch, 256);
    }
}
