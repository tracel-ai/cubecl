use cubecl::zspace::{Shape, Strides};

/// How the base strides of a tensor are laid out before any tile-level rank
/// expansion is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StridedLayout {
    RowMajor,
    Explicit(Vec<usize>),
}

/// Number of elements in the physical buffer required to cover every logical
/// index in `shape` under `strides`, assuming element 0 is at offset 0.
///
/// Exceeds `shape.iter().product()` for jumpy strides (e.g. a slice stepping
/// over padding) and is less than it for broadcast strides (a stride of 0
/// makes every index in that dim share the same physical offset).
pub fn physical_extent(shape: &Shape, strides: &Strides) -> usize {
    let mut max_offset = 0usize;
    for (s, d) in strides.iter().zip(shape.iter()) {
        if *d > 0 && *s > 0 {
            max_offset += (d - 1) * s;
        }
    }
    max_offset + 1
}

impl StridedLayout {
    pub fn compute_strides(&self, shape: &Shape) -> Strides {
        let n = shape.len();
        match self {
            StridedLayout::RowMajor => {
                assert!(n >= 1, "RowMajor requires at least 1 dimension");
                let mut strides = vec![0; n];
                strides[n - 1] = 1;
                for i in (0..n - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                Strides::new(&strides)
            }
            StridedLayout::Explicit(strides) => {
                assert!(
                    strides.len() == n,
                    "Explicit strides must have the same rank as the shape"
                );
                strides.clone().into()
            }
        }
    }
}
