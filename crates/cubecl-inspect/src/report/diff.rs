use super::{AutotuneReport, CanonicalKey, TunedKey};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// Two environments' autotune answers, key by key: what one has that the
/// other lacks, and where they answer the same key differently.
///
/// A key is matched by its tuner, its device, the adapter and API the device
/// resolved to, and its value — not by the
/// cubecl version in its namespace, so two environments a version apart
/// still compare. A file that answers one key under several versions answers it
/// with its newest. Winners that differ between two environments filled with the same
/// settings are the run-to-run variance that makes a loaded file compile
/// kernels its own sessions never did.
#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentDiff {
    pub before: PathBuf,
    pub after: PathBuf,
    /// Every key either side answers, in table order.
    pub keys: Vec<KeyDiff>,
}

/// The two environments a diff compares, by name, so they cannot be passed
/// the wrong way around.
pub struct Comparison<'a> {
    pub before: DiffSide<'a>,
    pub after: DiffSide<'a>,
}

/// One environment of a [`Comparison`]: its file and its autotune answers.
pub struct DiffSide<'a> {
    pub path: PathBuf,
    pub autotune: &'a AutotuneReport,
}

/// One key, as each side answers it.
#[derive(Clone, Debug, Serialize)]
pub struct KeyDiff {
    pub tuner: String,
    pub device: String,
    pub key: ciborium::Value,
    pub before: Option<Answer>,
    pub after: Option<Answer>,
}

/// How one side answers a key.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Answer {
    pub winner: String,
    /// The candidate list it was tuned under.
    pub checksum: String,
    pub margin: Option<f64>,
    /// The tune's wall, when it was recorded.
    pub wall: Option<Duration>,
}

/// What happened to a key from one side to the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyChange {
    Added,
    Removed,
    /// Both answer it, differently.
    WinnerChanged,
    /// Both answer it, alike.
    Kept,
}

impl From<&TunedKey> for Answer {
    fn from(key: &TunedKey) -> Self {
        Self {
            winner: key.winner_name(),
            checksum: key.checksum.clone(),
            margin: key.margin(),
            wall: key.wall(),
        }
    }
}

impl KeyDiff {
    pub fn change(&self) -> KeyChange {
        match (&self.before, &self.after) {
            (None, Some(_)) => KeyChange::Added,
            (Some(_), None) => KeyChange::Removed,
            (Some(before), Some(after)) if before.winner != after.winner => {
                KeyChange::WinnerChanged
            }
            (Some(_), Some(_)) | (None, None) => KeyChange::Kept,
        }
    }

    /// Whether the two sides tuned it under different candidate lists: a
    /// changed winner that is a changed list, not variance.
    pub fn candidates_changed(&self) -> bool {
        match (&self.before, &self.after) {
            (Some(before), Some(after)) => before.checksum != after.checksum,
            _ => false,
        }
    }

    /// The tune's wall on each side, before then after, when both were
    /// recorded.
    pub fn walls(&self) -> Option<(Duration, Duration)> {
        Some((self.before.as_ref()?.wall?, self.after.as_ref()?.wall?))
    }
}

/// What makes two stored keys the same key across environments.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Identity {
    tuner: String,
    device: String,
    fingerprint: Option<String>,
    key: CanonicalKey,
}

impl From<&TunedKey> for Identity {
    fn from(key: &TunedKey) -> Self {
        Self {
            tuner: key.table.tuner.clone(),
            device: key.table.device.clone(),
            fingerprint: key.table.fingerprint.clone(),
            key: CanonicalKey::new(&key.key),
        }
    }
}

impl EnvironmentDiff {
    pub fn new(comparison: Comparison<'_>) -> Self {
        let Comparison { before, after } = comparison;
        let mut keys = BTreeMap::<Identity, KeyDiff>::new();
        let blank = |key: &TunedKey| KeyDiff {
            tuner: key.table.tuner.clone(),
            device: key.table.device.clone(),
            key: key.key.clone(),
            before: None,
            after: None,
        };
        for key in newest(before.autotune).into_values() {
            keys.entry(Identity::from(key))
                .or_insert_with(|| blank(key))
                .before = Some(Answer::from(key));
        }
        for key in newest(after.autotune).into_values() {
            keys.entry(Identity::from(key))
                .or_insert_with(|| blank(key))
                .after = Some(Answer::from(key));
        }
        Self {
            before: before.path,
            after: after.path,
            keys: keys.into_values().collect(),
        }
    }

    /// The keys whose change is `change`.
    pub fn with(&self, change: KeyChange) -> impl Iterator<Item = &KeyDiff> {
        self.keys.iter().filter(move |key| key.change() == change)
    }
}

/// Each key of `report` once: when several cubecl versions answer it, the
/// newest's answer, versions compared by number rather than as text, where
/// `0.9` would pass `0.10`.
fn newest(report: &AutotuneReport) -> BTreeMap<Identity, &TunedKey> {
    let mut newest = BTreeMap::<Identity, &TunedKey>::new();
    for key in &report.keys {
        newest
            .entry(Identity::from(key))
            .and_modify(|kept| {
                if version(&key.table.version) > version(&kept.table.version) {
                    *kept = key;
                }
            })
            .or_insert(key);
    }
    newest
}

/// A version, for ordering: by number, a pre-release before its release. One
/// that does not parse orders before every one that does.
fn version(version: &str) -> Option<semver::Version> {
    semver::Version::parse(version).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_order_by_number() {
        assert!(version("0.10.0") > version("0.9.0"));
        assert!(version("0.9.1") > version("0.9.0"));
        assert_eq!(version("0.11.0"), version("0.11.0"));
        assert!(version("0.11.0") > version("0.11.0-pre.4"));
        assert!(version("0.11.0-pre.4") > version("0.10.0"));
        assert!(version("0.9.0") > version("not a version"));
    }
}
