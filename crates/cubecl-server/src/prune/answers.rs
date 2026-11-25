//! A prune by the answers the work used, as the tuners record them.

use alloc::string::String;
use alloc::vec::Vec;
use cubecl_environment::environment::{BuildStamp, PruneTally, UsedAnswers, UsedAnswersName};
use cubecl_environment::persistence::{Database, NamespaceContent, NamespaceRoot};

/// Removes from an environment the autotune answers this build's work did not
/// use, by what the environment recorded of those it used.
#[derive(Debug)]
pub struct AnswerPruner {
    /// The answers recorded, none when this build's work used none here.
    used: UsedAnswers,
    /// The namespaces recording what other builds used, removed with the
    /// answers only they used.
    other_builds: Vec<String>,
}

impl AnswerPruner {
    /// What `database` recorded of the work of `build`.
    pub fn read(database: &Database, build: BuildStamp) -> Self {
        let mut other_builds = database.namespaces();
        other_builds.retain(|namespace| {
            UsedAnswersName::parse(namespace).is_some_and(|name| name.build != build)
        });
        Self {
            used: UsedAnswers::read(database, build),
            other_builds,
        }
    }

    /// Remove from `database` the answers this build's work did not
    /// [use](Self::keeps), whichever table holds them, and other builds'
    /// records of theirs.
    pub fn prune(&self, database: &Database) -> PruneTally {
        let mut tally = PruneTally::default();
        for namespace in database.namespaces() {
            let content = NamespaceRoot::new(&namespace).content();
            if content != NamespaceContent::AutotuneAnswers {
                continue;
            }
            tally += self.prune_table(database, &namespace);
        }
        for namespace in &self.other_builds {
            database.purge(namespace);
        }
        tally
    }

    /// [`prune`](Self::prune), of one table.
    fn prune_table(&self, database: &Database, table: &str) -> PruneTally {
        let mut kept = 0;
        let mut doomed = Vec::new();
        database.scan(table, &mut |key, value| {
            let bytes = (key.len() + value.len()) as u64;
            if self.keeps(table, key) {
                kept += 1;
            } else {
                doomed.push((key.to_vec(), bytes));
            }
        });
        for (key, _) in &doomed {
            database.purge_key(table, key);
        }
        PruneTally {
            kept,
            removed: doomed.len() as u64,
            removed_bytes: doomed.iter().map(|(_, bytes)| bytes).sum(),
        }
    }

    /// Whether a prune keeps the answer `table` holds under `key`: one the
    /// work used, and any where the work is recorded using none, since
    /// nothing is then known of what it needs.
    fn keeps(&self, table: &str, key: &[u8]) -> bool {
        self.used.is_empty() || self.used.contains(table, key)
    }
}
