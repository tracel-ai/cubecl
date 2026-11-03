//! Comptime plan shared by every slot of one stages ([`StagePlan`]).

use super::payload::base::StageOperand;
use crate::*;

/// When a slot's buffer is brought to its region across the walk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Refill {
    /// Refilled every region.
    EveryRegion,
    /// Filled once, above the loop.
    Once,
    /// Reads the first slot's buffer; never filled here.
    Shared,
}

impl Refill {
    /// This operand's refill in a later slot, given its refill in the first.
    fn in_later_slot(self) -> Refill {
        match self {
            Refill::Once => Refill::Shared,
            Refill::EveryRegion => Refill::EveryRegion,
            Refill::Shared => unreachable!("Shared is produced only in later slots"),
        }
    }
}

/// Comptime plan for every slot of one stages: rendezvous and per-operand refill.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StagePlan {
    refills: Vec<Refill>,
    sync: Rendezvous,
    collective_full: bool,
    commits: bool,
    fences: bool,
    fillers: usize,
    owner: StageOwner,
}

impl StagePlan {
    /// Plan the operands a walk of `level` over `op_space` stages for `owner`. Where the walk
    /// sets planes aside to fill, they alone fill every operand, its cooperative share too
    /// ([`StageOwner::fill`]).
    pub(crate) fn new(
        operands: &[StageOperand],
        op_space: &Space,
        level: &Level,
        owner: StageOwner,
    ) -> StagePlan {
        let deliveries: Vec<_> = operands.iter().map(|op| op.delivery).collect();
        let fillers = level.fillers();
        let sync = Rendezvous::for_deliveries(&deliveries, fillers, owner);
        let collective_full = Rendezvous::collective_full(&deliveries);
        let commits = Rendezvous::commits(&deliveries);
        let fences = Rendezvous::fences(&deliveries);
        // A barrier slot arrives `full` once per fill, so fixing one operand would break parity.
        let can_fix_invariants = op_space.is_static() && sync != Rendezvous::Barrier;
        let refills = operands
            .iter()
            .map(
                |op| match can_fix_invariants && walk_invariant(level, op_space, op) {
                    true => Refill::Once,
                    false => Refill::EveryRegion,
                },
            )
            .collect();
        StagePlan {
            refills,
            sync,
            collective_full,
            commits,
            fences,
            fillers,
            owner,
        }
    }

    /// When each of slot `slot`'s operands is filled, in payload order.
    pub(crate) fn refills(&self, slot: usize) -> Vec<Refill> {
        match slot {
            FIRST_SLOT => self.refills.clone(),
            _ => self.refills.iter().map(|r| r.in_later_slot()).collect(),
        }
    }

    /// How this walk's slots rendezvous with their readers.
    pub(crate) fn sync(&self) -> Rendezvous {
        self.sync
    }

    /// Whether every unit of the cube arrives at a slot's fill.
    pub(crate) fn collective_full(&self) -> bool {
        self.collective_full
    }

    /// Whether a slot's units issue async copies its barrier must track.
    pub(crate) fn commits(&self) -> bool {
        self.commits
    }

    /// Whether a slot's barriers are fenced into the async proxy for a bulk copy.
    pub(crate) fn fences(&self) -> bool {
        self.fences
    }

    /// Planes that fill this walk's stages and take no tile.
    pub(crate) fn fillers(&self) -> usize {
        self.fillers
    }

    /// Who this walk's stages belong to: the cube, or each plane a copy of its own.
    pub(crate) fn owner(&self) -> StageOwner {
        self.owner
    }
}

/// Whether a walk of `level` over `space` leaves `operand`'s window unchanged: it spans no axis
/// the walk steps, and has no table left to read keyed by one.
fn walk_invariant(level: &Level, space: &Space, operand: &StageOperand) -> bool {
    space.axes().all(|axis| {
        level.tiles(space, axis) == 1
            || !(operand.space.contains(axis) || operand.keys.contains(&axis))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// An `M`/`N`/`K` space with `lhs` over `M`/`K` and `rhs` over `K`/`N`.
    fn spaces() -> (Space, Space, Space) {
        let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
        let lhs = space.subspace(&[M, K]);
        let rhs = space.subspace(&[K, N]);
        (space, lhs, rhs)
    }

    fn operand(delivery: Delivery, space: &Space) -> StageOperand {
        StageOperand {
            delivery,
            space: space.clone(),
            keys: Vec::new(),
        }
    }

    /// A walk stepping `M` leaves `rhs` (over `K`, `N`) where it lies, unless `rhs` has a table
    /// keyed by `M` still to read: then each region places it anew, so its stage is refilled.
    #[test]
    fn an_operand_with_a_table_to_read_is_refilled_along_its_keys() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 4), (N, 8), (K, 8)]);
        let plan = |keys: Vec<Axis>| {
            let rhs = StageOperand {
                keys,
                ..operand(Delivery::SyncPerUnit, &rhs)
            };
            StagePlan::new(
                &[operand(Delivery::SyncPerUnit, &lhs), rhs],
                &space,
                &level,
                StageOwner::CUBE,
            )
        };
        assert_eq!(plan(Vec::new()).refills(0)[1], Refill::Once);
        assert_eq!(plan(vec![M]).refills(0)[1], Refill::EveryRegion);
    }

    #[test]
    fn a_streamed_operand_is_rebuilt_in_every_slot() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 8), (K, 4)]);
        let plan = StagePlan::new(
            &[
                operand(Delivery::SyncPerUnit, &lhs),
                operand(Delivery::SyncPerUnit, &rhs),
            ],
            &space,
            &level,
            StageOwner::CUBE,
        );
        for slot in 0..2 {
            assert_eq!(plan.refills(slot)[0], Refill::EveryRegion);
        }
    }

    #[test]
    fn a_fixed_operand_reuses_the_first_slots_buffer() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 4), (K, 8)]);
        let plan = StagePlan::new(
            &[
                operand(Delivery::SyncPerUnit, &lhs),
                operand(Delivery::SyncPerUnit, &rhs),
            ],
            &space,
            &level,
            StageOwner::CUBE,
        );
        assert_eq!(plan.refills(0)[0], Refill::Once);
        assert_eq!(plan.refills(1)[0], Refill::Shared);
        assert_eq!(plan.refills(1)[1], Refill::EveryRegion);
    }

    #[test]
    fn a_slot_of_a_filled_walk_carries_the_count() {
        let (space, lhs, rhs) = spaces();
        let level = Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[M, N, K])
            .filled_by(2)
            .level();
        let plan = StagePlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Tma, &rhs)],
            &space,
            &level,
            StageOwner::CUBE,
        );
        assert_eq!(plan.fillers(), 2);
    }

    /// Planes set aside fill a slot an operand of which their units copy, beside a bulk copy of
    /// the other: an NVFP4 operand's scale words, copied beside the values the TMA engine lands.
    /// Every one of their units publishes, on a barrier.
    #[test]
    fn planes_set_aside_fill_a_slot_they_also_copy_into() {
        let (space, lhs, rhs) = spaces();
        let level = Levels::leaf(&[(M, 8), (N, 8), (K, 4)])
            .walk_every(&[M, N, K])
            .filled_by(1)
            .level();
        let plan = StagePlan::new(
            &[
                operand(Delivery::Tma, &lhs),
                operand(Delivery::SyncPerUnit, &rhs),
            ],
            &space,
            &level,
            StageOwner::Cube { fillers: 1 },
        );
        assert_eq!(plan.fillers(), 1);
        assert_eq!(plan.sync(), Rendezvous::Barrier);
        assert!(plan.collective_full());
    }

    #[test]
    fn a_tma_operand_is_never_fixed() {
        let (space, lhs, rhs) = spaces();
        let level = Level::every(&[(M, 8), (N, 4), (K, 8)]);
        let plan = StagePlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Tma, &rhs)],
            &space,
            &level,
            StageOwner::CUBE,
        );
        assert_eq!(plan.refills(0)[0], Refill::EveryRegion);
    }
}
