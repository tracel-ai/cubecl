//! The manual-mma leaf: `acc += lhs · rhs` via
//! [`MmaDefinition::execute`](cubecl::cmma::MmaDefinition).

use cubecl::cmma::{MatrixIdent, MatrixLayout};
use cubecl::prelude::*;

use crate::*;

#[cube]
impl MmaData {
    pub(crate) fn mma<A: Numeric, L: Numeric, R: Numeric>(&mut self, lhs: &Tile, rhs: &Tile) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let io = comptime!(self.io);
        let operands = comptime!(self.operands);
        let (lhs_layout, rhs_layout) =
            comptime!(window_layouts(&lhs.place.space, &rhs.place.space));
        let load = deep_load(lhs, rhs);
        let a = lhs.mma_operand::<L>(MatrixIdent::A, m, n, k, lhs_layout, io, operands, load);
        let b = rhs.mma_operand::<R>(MatrixIdent::B, m, n, k, rhs_layout, io, operands, load);

        match &mut self.fragment {
            MmaFragment::Acc(acc) => match (&a.fragment, &b.fragment) {
                (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => {
                    mma_execute::<L, R, A>(af.at::<L>(), bf.at::<R>(), acc.at_mut::<A>(), m, n, k)
                }
                // A left factor held as an accumulator, reread as the `A` it contracts as.
                (MmaFragment::Acc(af), MmaFragment::Rhs(bf)) => {
                    let af = accumulator_as_a::<L>(af.at::<L>(), m, n, k);
                    mma_execute::<L, R, A>(&af, bf.at::<R>(), acc.at_mut::<A>(), m, n, k)
                }
                (
                    MmaFragment::LhsBlockScaled(af, a_scales),
                    MmaFragment::RhsBlockScaled(bf, b_scales),
                ) => mma_execute_block_scaled::<A>(
                    af,
                    a_scales,
                    bf,
                    b_scales,
                    acc.at_mut::<A>(),
                    m,
                    n,
                    k,
                ),
                _ => panic!(
                    "MmaData::mma: operands must carry the Lhs/Rhs roles, both block-scaled or \
                     neither"
                ),
            },
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::mma: the accumulator must carry the Acc role")
            }
        }
    }
}

#[cube]
impl Tile {
    /// This factor as the `ident` operand of a manual-mma contraction: the fragment it holds, or
    /// its window loaded in `layout`, a fragment deeper than its instruction as `load` says.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mma_operand<T: Numeric>(
        &self,
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] operands: MmaOperands,
        #[comptime] load: DeepLoad,
    ) -> MmaData {
        match &self.kind {
            TileKind::PlaneTile(tile) => match tile {
                PlaneTile::Mma(data) => data.clone(),
                PlaneTile::Cmma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => {
                    panic!("Tile::mma_operand: a held operand must be an mma fragment")
                }
            },
            TileKind::Memory(_) => {
                let mut data = MmaData::operand::<T>(ident, m, n, k, layout, io, operands);
                data.load_window::<T>(self, load);
                data
            }
            TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::mma_operand: an operand must be a fragment or a memory window")
            }
        }
    }
}

/// How a contraction loads its factors' fragments where they are deeper than its instruction:
/// in lane runs where both are memory windows and either lies in global memory, where whole loads
/// pay, else a step at a time, which `ldmatrix` reads out of shared memory and a factor already
/// in fragments holds.
#[cube]
pub(crate) fn deep_load(lhs: &Tile, rhs: &Tile) -> comptime_type!(DeepLoad) {
    let (lhs_window, rhs_window) = (lhs.is_memory(), rhs.is_memory());
    let (lhs_shared, rhs_shared) = (lhs.is_shared(), rhs.is_shared());
    comptime!(
        match lhs_window && rhs_window && !(lhs_shared && rhs_shared) {
            true => DeepLoad::LaneRuns,
            false => DeepLoad::Steps,
        }
    )
}

/// Each factor's window layout: `A` row-major and `B` col-major when its trailing axis is
/// contracted, the opposite otherwise.
pub(crate) fn window_layouts(lhs: &Space, rhs: &Space) -> (MatrixLayout, MatrixLayout) {
    let trailing = |space: &Space| space.axis_at(space.rank() - 1);
    let lhs_layout = match rhs.contains(trailing(lhs)) {
        true => MatrixLayout::RowMajor,
        false => MatrixLayout::ColMajor,
    };
    let rhs_layout = match lhs.contains(trailing(rhs)) {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };
    (lhs_layout, rhs_layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Axis = Axis(0);
    const M: Axis = Axis(1);
    const N: Axis = Axis(2);
    const K: Axis = Axis(3);
    const TAPS: Axis = Axis(4);

    /// Layout follows whether each factor's trailing axis is contracted.
    #[test]
    fn a_window_reads_in_its_trailing_axis_order() {
        let lhs = Space::new(&[(B, 2), (M, 16), (K, 16)]);
        let row_weight = Space::new(&[(B, 2), (K, 16), (N, 16)]);
        let col_weight = Space::new(&[(B, 2), (N, 16), (K, 16)]);
        let lhs_transposed = Space::new(&[(B, 2), (K, 16), (M, 16)]);
        use MatrixLayout::{ColMajor, RowMajor};
        assert!(matches!(
            window_layouts(&lhs, &row_weight),
            (RowMajor, RowMajor)
        ));
        assert!(matches!(
            window_layouts(&lhs, &col_weight),
            (RowMajor, ColMajor)
        ));
        assert!(matches!(
            window_layouts(&lhs_transposed, &col_weight),
            (ColMajor, ColMajor)
        ));
    }

    /// Two contracted trailing axes read the same as one.
    #[test]
    fn a_window_contracted_over_two_axes_reads_as_over_one() {
        let lhs = Space::new(&[(M, 16), (TAPS, 3), (K, 16)]);
        let weight = Space::new(&[(N, 16), (TAPS, 3), (K, 16)]);
        use MatrixLayout::{ColMajor, RowMajor};
        assert!(matches!(
            window_layouts(&lhs, &weight),
            (RowMajor, ColMajor)
        ));
    }
}
