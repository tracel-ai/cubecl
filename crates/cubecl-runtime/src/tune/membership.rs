use alloc::sync::Arc;

use super::group::{PriorityFunc, TuneGroup};

/// A [tunable](super::Tunable)'s place in one of its [groups](TuneGroup): the group, and the
/// priority the tunable joins it with.
pub struct GroupMembership<K> {
    group: TuneGroup<K>,
    priority: PriorityFunc<K>,
}

impl<K> GroupMembership<K> {
    /// A membership of `group` at `priority`.
    pub fn new(group: &TuneGroup<K>, priority: impl Fn(&K) -> i8 + Send + Sync + 'static) -> Self {
        Self {
            group: group.clone(),
            priority: Arc::new(priority),
        }
    }

    pub fn group(&self) -> &TuneGroup<K> {
        &self.group
    }

    /// The priority the tunable holds within its group for `key`.
    pub fn priority(&self, key: &K) -> i8 {
        (self.priority)(key)
    }
}
