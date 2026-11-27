//! Removing the tunes a build's work did not use, by what it
//! recorded of those it did.

use super::{RecordedUses, UsedContent};
use crate::collections::{HashMap, HashSet};
use crate::environment::PruneTally;
use crate::persistence::{Database, NamespaceContent, NamespaceRoot};
use alloc::vec::Vec;

/// Removes from a database the tunes one build's work did not use.
///
/// Tables are told apart by their namespace, and tunes by their key as the
/// table's store encoded it: no row is decoded, so the layout of a table is
/// its tuner's alone.
#[derive(Debug)]
pub struct TunePruner<'a> {
    /// What the database records of the build's work, and of other builds'.
    recorded: &'a RecordedUses,
    /// The entries of the table being pruned that go, with the bytes each
    /// takes: filled while the table is scanned and emptied once it is
    /// (Scratch).
    doomed: Vec<(Vec<u8>, u64)>,
}

impl<'a> TunePruner<'a> {
    /// A pruner keeping the tunes `recorded` holds of the build's work.
    pub fn new(recorded: &'a RecordedUses) -> Self {
        Self {
            recorded,
            doomed: Vec::new(),
        }
    }

    /// Remove from `database`, in each autotune table the build used, the
    /// tunes it did not use and other builds' records of the table, and
    /// tally the tunes. A table it never used is left whole, with every
    /// record of it: it may be another hardware's, another version's, or a
    /// tuner's this build never ran.
    pub fn prune_used_tables(&mut self, database: &Database) -> PruneTally {
        let mut tally = PruneTally::default();
        for (content, namespace, kept) in used_tables(&self.recorded.uses) {
            tally += self.retain(database, namespace, kept);
            self.recorded.purge_other_builds(database, content);
        }
        tally
    }

    /// Remove from `database` every tune the build did not use,
    /// whichever table holds it, and count the tunes left.
    #[cfg(native_cache)]
    pub fn keep_only_used(&mut self, database: &Database) -> u64 {
        let tables = database
            .namespaces()
            .into_iter()
            .filter(|namespace| is_tune_table(namespace));
        let mut kept = 0;
        for namespace in tables {
            kept += self.keep_only_used_in(database, &namespace);
        }
        kept
    }
}

impl TunePruner<'_> {
    /// [`keep_only_used`](Self::keep_only_used), of the table `namespace`.
    #[cfg(native_cache)]
    fn keep_only_used_in(&mut self, database: &Database, namespace: &str) -> u64 {
        let content = UsedContent::Entries {
            namespace: namespace.into(),
        };
        let Some(kept) = self.recorded.uses.get(&content) else {
            database.purge(namespace);
            return 0;
        };
        self.retain(database, namespace, kept).kept
    }

    /// Remove from `namespace` every entry whose key `kept` does not hold,
    /// and tally them.
    fn retain(
        &mut self,
        database: &Database,
        namespace: &str,
        kept: &HashSet<Vec<u8>>,
    ) -> PruneTally {
        self.doomed.clear();
        let mut tally = PruneTally::default();
        database.scan(namespace, &mut |key, value| {
            let bytes = (key.len() + value.len()) as u64;
            if kept.contains(key) {
                tally.kept += 1;
            } else {
                self.doomed.push((key.to_vec(), bytes));
            }
        });
        for (key, bytes) in self.doomed.drain(..) {
            database.purge_key(namespace, &key);
            tally.removed += 1;
            tally.removed_bytes += bytes;
        }
        tally
    }
}

/// The autotune tables `uses` records a use of, each with its namespace and
/// the keys used.
fn used_tables(
    uses: &HashMap<UsedContent, HashSet<Vec<u8>>>,
) -> impl Iterator<Item = (&UsedContent, &str, &HashSet<Vec<u8>>)> {
    uses.iter().filter_map(|(content, keys)| match content {
        UsedContent::Entries { namespace } if is_tune_table(namespace) => {
            Some((content, namespace.as_str(), keys))
        }
        UsedContent::Entries { .. } | UsedContent::Kernels => None,
    })
}

/// Whether `namespace` is an autotune table.
fn is_tune_table(namespace: &str) -> bool {
    NamespaceRoot::new(namespace).content() == NamespaceContent::Tunes
}
