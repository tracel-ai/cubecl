use super::{AutotuneKey, PersistentCacheKey};
use cubecl_environment::environment::{BuildStamp, UsedAnswersName};
use cubecl_environment::persistence::{CacheOption, Store, StoreOptions};

/// Records in the active environment which of a table's answers the work used,
/// loaded or measured, per build: what a saved bundle gives back of them.
///
/// An answer recorded already is read, not written, so a warm start only
/// reads.
#[derive(Debug)]
pub struct UsedAnswerRecorder<K> {
    /// Lazy, as the table's own store is: it memoizes each key it finds
    /// recorded, and resets itself when the environment switches. `None` when
    /// the persistent cache is disabled.
    used: Option<Store<PersistentCacheKey<K>, ()>>,
}

impl<K: AutotuneKey> UsedAnswerRecorder<K> {
    /// Records the uses of the answers of `table`.
    pub fn new(table: &str) -> Self {
        let build = BuildStamp::current();
        let options = StoreOptions::new()
            .storage(UsedAnswersName { build, table }.namespace())
            .cache(CacheOption::Lazy);
        Self {
            used: Some(Store::new(options)),
        }
    }

    /// Records nothing: what a tuner that persists nothing holds.
    pub fn disabled() -> Self {
        Self { used: None }
    }

    /// Record that the work used the answer stored under `key`. A storage that
    /// refuses the write is logged, not reported: a lost record costs a tune,
    /// never a result.
    pub fn record(&mut self, key: PersistentCacheKey<K>) {
        let Some(used) = self.used.as_mut() else {
            return;
        };
        // `get_mut`, not `get`: a lazy store reads through to its storage
        // only there.
        if used.get_mut(&key).is_some() {
            return;
        }
        if let Err(error) = used.insert(key, ()) {
            log::debug!("an answer was not recorded as used: {}", error.reason());
        }
    }
}

#[cfg(test)]
impl UsedAnswerRecorder<alloc::string::String> {
    /// Records the uses of the answers of `table` in memory.
    pub fn in_memory(table: &str) -> Self {
        use alloc::boxed::Box;
        use cubecl_environment::persistence::MemoryStorage;

        let build = BuildStamp::current();
        let namespace = UsedAnswersName { build, table }.namespace();
        let storage = MemoryStorage::new(namespace.as_str());
        let options = StoreOptions::new()
            .storage_with(Box::new(storage), namespace)
            .cache(CacheOption::Lazy);
        Self {
            used: Some(Store::new(options)),
        }
    }

    /// Every use recorded, by key and list.
    pub fn recorded(&mut self) -> alloc::vec::Vec<PersistentCacheKey<alloc::string::String>> {
        let mut recorded = alloc::vec::Vec::new();
        if let Some(used) = &mut self.used {
            used.scan(|key, ()| recorded.push(key));
        }
        recorded.sort_by(|a, b| (&a.key, &a.checksum).cmp(&(&b.key, &b.checksum)));
        recorded
    }
}
