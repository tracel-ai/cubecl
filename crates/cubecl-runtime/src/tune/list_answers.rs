//! The answers a table holds to candidate lists other than the one a tune ran.
//!
//! A tuner stores its answer to a key under the checksum of its candidate
//! list, so a list that changes — a new binary, an edited configuration —
//! leaves the old list's answers beside the new ones. A lookup asks for the
//! answer to the list it runs, key and checksum together, so an old answer is
//! never served in place of the current one; left alone it would only be dead
//! weight, saved with the environment and counted as one of its answers. A
//! tune that stores an answer therefore removes the key's answers to every
//! list no tuner of this process runs on the table. A list another tuner of
//! the same hardware runs keeps its answer: this process still reads it.

use alloc::string::String;
use alloc::vec::Vec;

use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::Store;
use cubecl_environment::sync::{LazyLock, Mutex};

use super::{AutotuneKey, PersistentCacheKey, PersistentCacheValue};

/// A table's answers, as their store keeps them.
pub(crate) type AnswerStore<K> = Store<PersistentCacheKey<K>, PersistentCacheValue>;

/// The candidate lists the tuners of this process run, by the namespace of
/// the table each runs against. Process-wide, since every list that shares a
/// table has a tuner of its own.
static RUNNING: LazyLock<Mutex<HashMap<String, Vec<String>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Records that a tuner of this process runs the list `checksum` against
/// `table`, so no other list's tune removes its answers.
pub(crate) fn register_running_list(table: &str, checksum: &str) {
    let mut running = RUNNING.lock();
    let lists = running.entry(table.into()).or_default();
    if !lists.iter().any(|list| list == checksum) {
        lists.push(checksum.into());
    }
}

/// The lists a table holds an answer under, by key: read from its store once,
/// when the first answer of a process lands in it, then kept in step with what
/// a tune stores and removes.
#[derive(Debug)]
pub(crate) struct ListAnswers<K> {
    lists: HashMap<K, Vec<String>>,
}

impl<K: AutotuneKey> ListAnswers<K> {
    /// The lists `store` holds an answer under, by key.
    pub(crate) fn new(store: &mut AnswerStore<K>) -> Self {
        let mut lists: HashMap<K, Vec<String>> = HashMap::new();
        store.scan(|stored, _| lists.entry(stored.key).or_default().push(stored.checksum));
        Self { lists }
    }

    /// Records that `key` is answered under `checksum` in `store`, the store
    /// of `table`, and removes from it the key's answers to every list no
    /// tuner of this process runs against the table.
    pub(crate) fn supersede(
        &mut self,
        store: &mut AnswerStore<K>,
        table: &str,
        key: &K,
        checksum: &str,
    ) {
        let running = RUNNING.lock().get(table).cloned().unwrap_or_default();
        let lists = self.lists.entry(key.clone()).or_default();
        lists.retain(|list| {
            let current = list == checksum || running.contains(list);
            if !current {
                store.purge_key(&PersistentCacheKey {
                    key: key.clone(),
                    checksum: list.clone(),
                });
            }
            current
        });
        if !lists.iter().any(|list| list == checksum) {
            lists.push(checksum.into());
        }
    }
}
