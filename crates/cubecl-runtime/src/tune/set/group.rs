use alloc::{string::String, sync::Arc};
use core::sync::atomic::{AtomicU32, Ordering};

use crate::config::autotune::AutotuneLevel;

/// A priority bucket for tunables, computed from the [autotune key](crate::tune::AutotuneKeyable).
///
/// Higher-priority groups are autotuned first; once any tunable in a group returns a valid
/// result, no later groups are tried.
///
/// # Assumptions
///
/// - Its priority function, and the priority each member joins it with
///   ([`Tunable::group`](crate::tune::Tunable::group)), answer from the key alone: a set plans each key
///   once and keeps the order it planned.
pub struct TuneGroup<K> {
    id: GroupId,
    name: Arc<String>,
    priority: PriorityFunc<K>,
    order: GroupOrder,
}

/// A priority function, of a group or of a member within it.
pub(crate) type PriorityFunc<K> = Arc<dyn Fn(&K) -> i8 + Send + Sync>;

/// Identifies a [`TuneGroup`]: two groups may share a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

/// How a [`TuneGroup`]'s member priorities shape its batch.
#[derive(Debug, Clone, Copy)]
enum GroupOrder {
    /// The highest level is the round, the levels under it are fallbacks.
    Cutoff,
    /// Every member is in one round, best first.
    Ordered(OrderedSettings),
}

/// The settings of an [ordered](TuneGroup::ordered) group.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OrderedSettings {
    /// When to stop measuring the group's members. Without one, every member is measured.
    pub patience: Option<Patience>,
}

impl OrderedSettings {
    /// Stop measuring the group once its members stop beating its best one.
    pub fn with_patience(self, patience: Patience) -> Self {
        Self {
            patience: Some(patience),
        }
    }
}

/// When to stop measuring an [ordered](TuneGroup::ordered) group: once at least
/// [`min_measured_count`](Self::min_measured_count) of its members were measured and the last
/// [`max_miss_count`](Self::max_miss_count) of them in a row missed, none faster than the group's
/// leader by more than 2%. The leader is the first member measured, then each member that was
/// faster than it by more than 2%.
///
/// The group is benchmarked best first, so a run of members that cannot beat its leader says the
/// rest of the order is unlikely to either, and they are never measured: at most one window of
/// them, compiled ahead of measuring, is compiled. The members already measured still compete as
/// usual, the ones skipped are reported as such, and the rest of the batch is measured as if the
/// group had ended there. A member is skipped only when every group that brought it into the batch
/// ran out of patience. Should every member measured fail later on, the skipped ones are measured
/// after all.
///
/// A member is judged on its best first-pass time, which follows a single warmup: a lucky one
/// makes a leader the rest of the group struggles to beat. A member that fails is neither measured
/// nor a miss: it says nothing about how the order is faring.
///
/// Adaptive scheduler only: the fixed-count pass, the only one on wasm, measures every member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Patience {
    /// Members measured before the group may stop. One member is always measured. The batch also
    /// measures this many candidates before the [bounds](crate::tune::Bounds)' time limit may end it.
    pub min_measured_count: usize,
    /// Members in a row that miss before the group stops. Zero stops it at
    /// [`min_measured_count`](Self::min_measured_count), whatever they measured.
    pub max_miss_count: usize,
}

impl Patience {
    /// The patience an [`AutotuneLevel`] races an ordered group with: none at
    /// [`Full`](AutotuneLevel::Full), which measures every member, and twenty misses after ten
    /// measured members at every other level.
    ///
    /// Twenty misses, not fewer, while the estimates that order a group still miss some kernels by
    /// several times: replayed over matmul and attention races on four devices, it settles on the
    /// fastest candidate of each, measuring about a third of them.
    pub const fn for_level(level: &AutotuneLevel) -> Option<Self> {
        match level {
            AutotuneLevel::Full => None,
            AutotuneLevel::Minimal | AutotuneLevel::Balanced | AutotuneLevel::Extensive => {
                Some(LEVEL_PATIENCE)
            }
        }
    }
}

/// The patience of every level that does not measure every member.
const LEVEL_PATIENCE: Patience = Patience {
    min_measured_count: 10,
    max_miss_count: 20,
};

impl<K> TuneGroup<K> {
    /// A group whose members' own priorities ([`Tunable::group`](crate::tune::Tunable::group)) are a
    /// cutoff: the highest level is the round, and the levels under it are fallbacks reached as
    /// each round fails.
    pub fn new(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static) -> Self {
        Self::build(name, f, GroupOrder::Cutoff)
    }

    /// A group whose member priorities order one round rather than cut it off: every member with
    /// a non-negative priority is in the batch, best first.
    ///
    /// The batch is benchmarked in that order and the short circuit ends it at the first candidate
    /// under the [bounds](crate::tune::Bounds)' time limit, once the batch measured its
    /// [`Patience::min_measured_count`], so the priority decides how much of the group is compiled
    /// rather than which of it can win. The batch costs its whole membership without a bounds
    /// generator, on wasm, or with the short circuit disabled.
    ///
    /// The batch sits at the priority of its best member, so a cutoff group at the same group
    /// priority interleaves with it by priority.
    pub fn ordered(
        name: &str,
        f: impl Fn(&K) -> i8 + Send + Sync + 'static,
        settings: OrderedSettings,
    ) -> Self {
        Self::build(name, f, GroupOrder::Ordered(settings))
    }
}

impl<K> TuneGroup<K> {
    fn build(name: &str, f: impl Fn(&K) -> i8 + Send + Sync + 'static, order: GroupOrder) -> Self {
        Self {
            id: GroupId(GROUP_COUNTER.fetch_add(1, Ordering::Relaxed)),
            name: Arc::new(name.into()),
            priority: Arc::new(f),
            order,
        }
    }

    pub(crate) fn id(&self) -> GroupId {
        self.id
    }

    #[cfg(persistence)]
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The group's own priority for `key`.
    pub(crate) fn priority(&self, key: &K) -> i8 {
        (self.priority)(key)
    }

    /// Whether a member joining at `priority` is part of the group's one ordered round.
    pub(crate) fn orders_priority(&self, priority: i8) -> bool {
        self.order.orders_priority(priority)
    }

    #[cfg(persistence)]
    pub(crate) fn is_ordered(&self) -> bool {
        self.order.is_ordered()
    }

    pub(crate) fn patience(&self) -> Option<Patience> {
        self.order.patience()
    }
}

impl GroupOrder {
    fn orders_priority(&self, priority: i8) -> bool {
        self.is_ordered() && priority >= 0
    }

    fn is_ordered(&self) -> bool {
        matches!(self, Self::Ordered(_))
    }

    fn patience(&self) -> Option<Patience> {
        match self {
            Self::Ordered(settings) => settings.patience,
            Self::Cutoff => None,
        }
    }
}

impl<K> core::fmt::Debug for TuneGroup<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TuneGroup")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl<K> Clone for TuneGroup<K> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            priority: self.priority.clone(),
            order: self.order,
        }
    }
}

static GROUP_COUNTER: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_full_level_races_every_member() {
        assert_eq!(Patience::for_level(&AutotuneLevel::Full), None);
    }

    #[test]
    fn every_other_level_races_with_patience() {
        assert_eq!(
            Patience::for_level(&AutotuneLevel::Balanced),
            Some(LEVEL_PATIENCE)
        );
    }

    #[test]
    fn an_ordered_group_orders_only_its_non_negative_members() {
        let group = TuneGroup::<()>::ordered("ordered", |_| 1, OrderedSettings::default());

        assert!(group.orders_priority(0));
        assert!(!group.orders_priority(-1));
    }

    #[test]
    fn a_cutoff_group_orders_no_member_and_has_no_patience() {
        let group = TuneGroup::<()>::new("cutoff", |_| 1);

        assert!(!group.orders_priority(3));
        assert_eq!(group.patience(), None);
    }

    #[test]
    fn two_groups_of_one_name_are_told_apart() {
        let first = TuneGroup::<()>::new("shared", |_| 1);
        let second = TuneGroup::<()>::new("shared", |_| 1);

        assert_ne!(first.id(), second.id());
    }
}
