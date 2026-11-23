use cubecl_ir::{ElemType, features::Features};

/// The matrix instruction a [`ComputeCmma`](super::ThroughputMode::ComputeCmma) probe times.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeCmmaConfig {
    /// The data type used to store the running sum.
    pub accumulator_type: AccumulatorType,
    /// The tile one instruction computes.
    pub cmma_dims: CmmaDims,
}

/// The element type of the accumulator.
pub type AccumulatorType = ElemType;

/// The M, N, and K dimensions of a matrix multiplication.
#[derive(Eq, PartialEq, Clone, Hash, Debug, Copy)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct CmmaDims {
    /// Rows in the output matrix.
    pub m: usize,
    /// Columns in the output matrix.
    pub n: usize,
    /// The shared inner dimension of the input matrices.
    pub k: usize,
}

impl CmmaDims {
    /// The multiply-adds one tile of these dims does: `m * n * k`.
    pub fn num_elems(&self) -> usize {
        self.m * self.n * self.k
    }

    /// The largest tile a cooperative or plane matrix instruction of `features` computes on `lhs`
    /// and `rhs` into `acc`, among those that fit within `bound`. `None` where none does.
    pub fn largest_supported(
        features: &Features,
        lhs: ElemType,
        rhs: ElemType,
        acc: ElemType,
        bound: CmmaDims,
    ) -> Option<Self> {
        features
            .matmul
            .cmma
            .iter()
            .chain(features.matmul.mma.iter())
            .filter(|it| it.a_type == lhs && it.b_type == rhs && it.cd_type == acc)
            .map(|it| CmmaDims {
                m: it.m as usize,
                n: it.n as usize,
                k: it.k as usize,
            })
            .filter(|dims| dims.fits_within(&bound))
            .max_by_key(CmmaDims::num_elems)
    }
}

impl CmmaDims {
    fn fits_within(&self, bound: &CmmaDims) -> bool {
        self.m <= bound.m && self.n <= bound.n && self.k <= bound.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_ir::{FloatKind, features::MmaConfig};

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const BOUND: CmmaDims = CmmaDims {
        m: 64,
        n: 64,
        k: 64,
    };

    fn with_tiles(tiles: &[(u32, u32, u32)]) -> Features {
        let mut features = Features::default();
        for &(m, n, k) in tiles {
            features.matmul.cmma.insert(MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: F32,
                m,
                n,
                k,
            });
        }
        features
    }

    #[test]
    fn the_largest_tile_is_chosen() {
        let features = with_tiles(&[(16, 16, 16), (32, 8, 16)]);

        assert_eq!(
            CmmaDims::largest_supported(&features, F16, F16, F32, BOUND),
            Some(CmmaDims { m: 32, n: 8, k: 16 })
        );
    }

    #[test]
    fn a_tile_past_the_bound_is_skipped() {
        let features = with_tiles(&[(128, 16, 16)]);

        assert_eq!(
            CmmaDims::largest_supported(&features, F16, F16, F32, BOUND),
            None
        );
    }

    #[test]
    fn a_tile_of_other_types_is_skipped() {
        let features = with_tiles(&[(16, 16, 16)]);

        assert_eq!(
            CmmaDims::largest_supported(&features, F32, F32, F32, BOUND),
            None
        );
    }

    #[test]
    fn the_volume_is_the_product_of_the_dims() {
        assert_eq!(CmmaDims { m: 2, n: 3, k: 4 }.num_elems(), 24);
    }
}
