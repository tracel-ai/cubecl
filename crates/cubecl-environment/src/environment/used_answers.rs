//! Which autotune answers the work of a build used in an environment.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{AutotuneAnswer, BuildStamp};
use crate::bundle::Bundle;
use crate::persistence::Namespace;

/// What names the namespace recording which answers of one table the work of
/// one build used, keyed as the table is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedAnswersName<'a> {
    /// The build whose work used them.
    pub build: BuildStamp,
    /// The table, by its whole namespace.
    pub table: &'a str,
}

impl<'a> UsedAnswersName<'a> {
    /// The root of the namespaces the used answers are recorded under.
    pub const ROOT: &'static str = "used-answers";

    /// `used-answers/<version>/<build>/<table>`.
    pub fn namespace(&self) -> Namespace {
        let Self { build, table } = self;
        Namespace::scoped(Self::ROOT, alloc::format!("{build}/{table}"))
    }

    /// The name `namespace` was written from, whichever cubecl version wrote
    /// it, or `None` when it records no used answers.
    pub fn parse(namespace: &'a str) -> Option<Self> {
        let path = namespace.strip_prefix(Self::ROOT)?.strip_prefix('/')?;
        let (_version, path) = path.split_once('/')?;
        let (build, table) = path.split_once('/')?;
        let build = build.parse().ok()?;
        Some(Self { build, table })
    }
}

/// The autotune answers the work of one build used in an environment.
#[derive(Debug, Clone, Default)]
pub struct UsedAnswers {
    /// The keys recorded, by table.
    tables: BTreeMap<String, BTreeSet<Vec<u8>>>,
}

impl UsedAnswers {
    /// What `entries` records of the answers the work of `build` used.
    pub fn read(entries: &dyn Bundle, build: BuildStamp) -> Self {
        let mut tables: BTreeMap<String, BTreeSet<Vec<u8>>> = BTreeMap::new();
        for namespace in entries.namespaces() {
            let Some(name) = UsedAnswersName::parse(&namespace).filter(|name| name.build == build)
            else {
                continue;
            };
            let keys = tables.entry(name.table.to_string()).or_default();
            entries.scan(&namespace, &mut |key, _| {
                keys.insert(key.to_vec());
            });
        }
        Self { tables }
    }

    /// Whether the work is recorded using none, so that nothing is known of the
    /// answers an environment holds.
    pub fn is_empty(&self) -> bool {
        self.tables.values().all(BTreeSet::is_empty)
    }

    /// Whether the work used the answer `table` holds under `key`.
    pub fn contains(&self, table: &str, key: &[u8]) -> bool {
        self.tables
            .get(table)
            .is_some_and(|keys| keys.contains(key))
    }

    /// The answers as `entries` holds them now, trimmed to their picks. One
    /// used and since removed, or whose value does not decode, is left out.
    pub fn answers(&self, entries: &dyn Bundle) -> Vec<AutotuneAnswer> {
        let mut answers = Vec::new();
        for (table, keys) in &self.tables {
            entries.scan(table, &mut |key, value| {
                let used = keys
                    .contains(key)
                    .then(|| AutotuneAnswer::stored(table, key, value));
                answers.extend(used.flatten());
            });
        }
        answers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAME: UsedAnswersName<'static> = UsedAnswersName {
        build: BuildStamp { hash: 0xff },
        table: "autotune/0.12.0/layout-v1/hardware/tuner",
    };

    #[test]
    fn a_name_reads_back_from_its_namespace() {
        let namespace = NAME.namespace();
        assert_eq!(UsedAnswersName::parse(namespace.as_str()), Some(NAME));
    }

    #[test]
    fn a_namespace_of_something_else_names_no_used_answers() {
        for namespace in [
            NAME.table,
            "launched/0.12.0/000000000000000000000000000000ff",
            "used-answers",
            "used-answers/0.12.0/not-a-build/autotune/0.12.0",
            "used-answers-of-mine/0.12.0/000000000000000000000000000000ff/autotune",
        ] {
            assert_eq!(UsedAnswersName::parse(namespace), None, "{namespace}");
        }
    }
}
