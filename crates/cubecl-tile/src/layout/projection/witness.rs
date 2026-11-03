//! Which buffer dim witnesses an axis's runtime extent.

use crate::{Axis, Projection};

/// The one physical dim carrying `axis` alone at coefficient `1`, whose bound is its extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Witness {
    dim: usize,
}

impl Witness {
    pub(crate) fn new(projection: &Projection, axis: Axis) -> Option<Self> {
        if !projection.addresses(axis) {
            return None;
        }
        match projection.carriers(axis)[..] {
            [pa] if projection.physical_axis(pa).is_identity(axis) => Some(Witness { dim: pa }),
            _ => None,
        }
    }

    /// The physical dim whose bound is the extent.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicalAxisMap, Projection};

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);

    /// A bound is an axis's extent only when one dim carries that axis alone.
    #[test]
    fn a_witness_is_one_dim_carrying_the_axis_alone() {
        let direct = Projection::direct(&[A, B]);
        assert_eq!(Witness::new(&direct, A).map(|w| w.dim()), Some(0));
        assert_eq!(Witness::new(&direct, B).map(|w| w.dim()), Some(1));

        let gathered = Projection::new(&[A, B], &[PhysicalAxisMap::affine(&[(A, 1), (B, 1)])]);
        assert_eq!(Witness::new(&gathered, A), None);
        assert_eq!(Witness::new(&gathered, B), None);

        let tiled = Projection::tiled(&[A, B], &[A, B, B]);
        assert_eq!(Witness::new(&tiled, A).map(|w| w.dim()), Some(0));
        assert_eq!(Witness::new(&tiled, B), None);
    }
}
