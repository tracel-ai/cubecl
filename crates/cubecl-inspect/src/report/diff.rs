use super::{AutotuneReport, TunedKey};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// Two environments' autotune answers, key by key: what one has that the
/// other lacks, and where they answer the same key differently.
///
/// A key is matched by its tuner, its device and its value — not by the
/// cubecl version in its namespace, so two builds a version apart still
/// compare. Winners that differ between two builds of the same settings are
/// the run-to-run variance that makes a loaded file compile kernels its build
/// never did.
#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentDiff {
    pub before: PathBuf,
    pub after: PathBuf,
    /// Every key either side answers, in table order.
    pub keys: Vec<KeyDiff>,
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
    /// The key's canonical rendering: `ciborium::Value` is not `Ord`.
    key: String,
}

impl From<&TunedKey> for Identity {
    fn from(key: &TunedKey) -> Self {
        Self {
            tuner: key.table.tuner.clone(),
            device: key.table.device.clone(),
            key: format!("{:?}", key.key),
        }
    }
}

impl EnvironmentDiff {
    pub fn new(before: (PathBuf, &AutotuneReport), after: (PathBuf, &AutotuneReport)) -> Self {
        let mut keys = BTreeMap::<Identity, KeyDiff>::new();
        let blank = |key: &TunedKey| KeyDiff {
            tuner: key.table.tuner.clone(),
            device: key.table.device.clone(),
            key: key.key.clone(),
            before: None,
            after: None,
        };
        for key in &before.1.keys {
            keys.entry(Identity::from(key))
                .or_insert_with(|| blank(key))
                .before = Some(Answer::from(key));
        }
        for key in &after.1.keys {
            keys.entry(Identity::from(key))
                .or_insert_with(|| blank(key))
                .after = Some(Answer::from(key));
        }
        Self {
            before: before.0,
            after: after.0,
            keys: keys.into_values().collect(),
        }
    }

    /// The keys whose change is `change`.
    pub fn with(&self, change: KeyChange) -> impl Iterator<Item = &KeyDiff> {
        self.keys.iter().filter(move |key| key.change() == change)
    }
}
