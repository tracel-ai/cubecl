//! `sum.mma_prefetched(steps, a, b)`: a walk of register-block contractions, each step's loads
//! fetched into the unit's registers before the step ahead of it contracts.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// `self += Σ lhs · rhs` over the steps `steps` walks, each step's unit contracting its leaf
    /// into its register block, as `for step in steps { for unit in step { self.at(&unit)
    /// .mma(&lhs.at(&unit), &rhs.at(&unit)) } }` does, with what step `i + 1` reads fetched into
    /// the unit's registers, undecoded, before step `i` decodes and multiplies what it fetched:
    /// the next step's loads are in flight while this one contracts. Two steps' reads are held at
    /// once.
    ///
    /// A sum of register blocks whose cells sum the rhs's runs along the contraction, against a
    /// packed rhs under at most one level of packed block scales and a plain, unscaled lhs, both
    /// in memory, each step one unit a thread.
    pub fn mma_prefetched(&self, steps: &Walk, lhs: &Tile, rhs: &Tile) {
        let total = steps.total();
        if total > 0 {
            let first = steps.region(0usize).walk().only();
            let (acc, a, b) = (self.at(&first), lhs.at(&first), rhs.at(&first));
            let mut even = acc.fetched_for(&a, &b);
            let mut odd = acc.fetched_for(&a, &b);
            self.fetch_step(steps, 0usize, lhs, rhs, &mut even);
            for pair in 0..total / 2 {
                let step = pair * 2;
                self.fetch_step(steps, step + 1, lhs, rhs, &mut odd);
                self.mma_step(steps, step, lhs, rhs, &even);
                if step + 2 < total {
                    self.fetch_step(steps, step + 2, lhs, rhs, &mut even);
                }
                self.mma_step(steps, step + 1, lhs, rhs, &odd);
            }
            if total % 2 == 1 {
                self.mma_step(steps, total - 1, lhs, rhs, &even);
            }
        }
    }

    /// Registers for what this unit's register block reads of `lhs` and `rhs` in one step.
    fn fetched_for(&self, lhs: &Tile, rhs: &Tile) -> Fetched {
        let out = comptime!(self.place.space.clone());
        let define!(Lhs) = lhs.element;
        let block = self.register_block();
        block.fetched::<Lhs>(lhs, rhs, out)
    }

    /// Load into `fetched` what step `step` of `steps` reads of `lhs` and `rhs` for this thread's
    /// unit.
    fn fetch_step(&self, steps: &Walk, step: usize, lhs: &Tile, rhs: &Tile, fetched: &mut Fetched) {
        let unit = steps.region(step).walk().only();
        let acc = self.at(&unit);
        let out = comptime!(acc.place.space.clone());
        let define!(Lhs) = lhs.element;
        let block = acc.register_block();
        block.fetch::<Lhs>(&lhs.at(&unit), &rhs.at(&unit), out, fetched);
    }

    /// Contract step `step` of `steps` into this thread's unit from what `fetched` holds of it.
    fn mma_step(&self, steps: &Walk, step: usize, lhs: &Tile, rhs: &Tile, fetched: &Fetched) {
        let unit = steps.region(step).walk().only();
        let acc = self.at(&unit);
        let out = comptime!(acc.place.space.clone());
        let (a, b) = (lhs.at(&unit), rhs.at(&unit));
        let define!(Served) = acc.element;
        let define!(Lhs) = a.element;
        let define!(Rhs) = b.read_element();
        // The block's registers are the accumulator's: writes through it land in the sum.
        let mut block = acc.register_block();
        block.mma_fetched::<Served, Lhs, Rhs>(&a, &b, out, fetched);
    }

    /// This unit accumulator's register block.
    fn register_block(&self) -> RegisterData {
        match &self.kind {
            TileKind::PlaneTile(tile) => match tile {
                PlaneTile::Registers(block) => block.clone(),
                PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => {
                    panic!(
                        "Tile::mma_prefetched: a prefetching walk contracts into a register block"
                    )
                }
            },
            TileKind::PlanePartition(partition) => match partition.fragment() {
                PlaneTile::Registers(block) => block,
                PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => {
                    panic!(
                        "Tile::mma_prefetched: a prefetching walk contracts into a register block"
                    )
                }
            },
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::mma_prefetched: a prefetching walk contracts into a unit's register block"
                )
            }
        }
    }
}
