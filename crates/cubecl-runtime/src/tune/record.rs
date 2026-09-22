//! What a tune leaves in the environment beside its answer.
//!
//! The table stores a key's winner and every candidate's result, ranked. What
//! it cannot say is how the tune went: which candidates ran and in what order,
//! what each cost to compile and benchmark, whether the tune stopped early,
//! and how long the key took from its miss to its answer. A [`TuneRecord`] is
//! that account, written to [`cubecl_environment::records`] once per tune.

use crate::tune::{
    AutotuneKey, AutotuneLogContext, AutotuneLogEvent, PersistentCacheKey, TuneCache,
};
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
use cubecl_environment::records::{Record, RecordEffect, Span};
use serde::{Deserialize, Serialize};

/// How one autotune key was decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneRecord<K> {
    /// The namespace of the table the answer is stored in.
    pub table: String,
    /// The table's entry the answer is stored under: the key, and the
    /// checksum of the candidate list it was tuned under.
    pub entry: PersistentCacheKey<K>,
    /// The index of the candidate the key runs.
    pub winner: usize,
    /// The candidates that ran, in the order they ran.
    pub trials: Vec<Trial>,
    /// The candidate that met the time limit and ended the tune before the
    /// rest of the plan ran.
    pub short_circuit: Option<String>,
    /// From the cache miss to the answer committed.
    pub wall: Duration,
    /// Whether the tune ran inside a dry run, where launches compile but do
    /// not execute.
    pub dry_run: bool,
    /// Whether the table took the answer. One that measured nothing, or that
    /// was tuned with the cache disabled, answers this process alone: the
    /// table holds another answer to the key, or none.
    pub stored: bool,
}

/// One candidate's run within a tune.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trial {
    /// The candidate's name.
    pub name: String,
    /// Compiling and benchmarking it: from its first launch to its samples
    /// resolved.
    pub wall: Duration,
}

impl<K> Record for TuneRecord<K> {
    const KIND: &'static str = "autotune";
}

/// A tune being recorded: stamped when it began, written when it ends. Every
/// call is a no-op when the environment records nothing.
#[derive(Debug)]
pub(crate) struct TuneRecording<K> {
    open: Option<OpenRecording<K>>,
}

/// What a [`TuneRecording`] holds while the environment records.
#[derive(Debug)]
struct OpenRecording<K> {
    span: Span,
    table: String,
    entry: PersistentCacheKey<K>,
    dry_run: bool,
}

/// What [`TuneRecording::finish`] needs to know of the answer.
pub(crate) struct Answer {
    pub winner: usize,
    /// Whether the table took it: see [`TuneRecord::stored`].
    pub stored: bool,
}

impl<K: AutotuneKey> TuneRecording<K> {
    /// Begin recording the tune of `key` in `cache`'s table. The key is
    /// cloned only when the environment records.
    pub(crate) fn new(cache: &TuneCache<K>, key: &K, checksum: &str) -> Self {
        let open = Span::new().map(|span| OpenRecording {
            span,
            table: cache.table().into(),
            entry: PersistentCacheKey {
                key: key.clone(),
                checksum: checksum.into(),
            },
            dry_run: crate::dry_run::dry_run(),
        });
        Self { open }
    }

    /// Whether the tune is recorded, and so has to track its steps: the
    /// record's trials are the ones the log context collects.
    pub(crate) fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Write the record of the tune that just answered.
    pub(crate) fn finish(self, answer: Answer, log_context: Option<&AutotuneLogContext>) {
        let Some(open) = self.open else {
            return;
        };
        // A tune the environment switched away from went with its session.
        let Some(wall) = open.span.elapsed() else {
            return;
        };
        let mut trials = Vec::new();
        let mut short_circuit = None;
        for event in log_context.iter().flat_map(|context| &context.events) {
            match event {
                AutotuneLogEvent::TuningStep(name, wall) => trials.push(Trial {
                    name: name.clone(),
                    wall: *wall,
                }),
                AutotuneLogEvent::ShortCircuit(name) => short_circuit = Some(name.clone()),
            }
        }
        let record = TuneRecord {
            table: open.table,
            entry: open.entry,
            winner: answer.winner,
            trials,
            short_circuit,
            wall,
            dry_run: open.dry_run,
            stored: answer.stored,
        };
        // A stored winner is the environment changing; an answer kept in
        // memory is not.
        let effect = if record.stored {
            RecordEffect::Changed
        } else {
            RecordEffect::Observed
        };
        open.span.close(effect, &record);
    }
}
