//! What a tune leaves in the environment beside its answer.
//!
//! The table stores a key's winner and every candidate's result, ranked. What
//! it cannot say is how the tune went: which candidates ran and in what order,
//! what each cost to compile and benchmark, whether the tune stopped early,
//! and how long the key took from its miss to its answer. A [`TuneRecord`] is
//! that account, written to [`cubecl_environment::records`] once per tune.

use crate::tune::{AutotuneLogContext, AutotuneLogEvent};
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
use cubecl_environment::records::{self, Stamp};
use serde::{Deserialize, Serialize};

/// How one autotune key was decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneRecord<K> {
    /// The namespace of the table the answer is stored in.
    pub table: String,
    /// The key, and the checksum of the candidate list it was tuned under:
    /// together, what names the table's entry.
    pub key: K,
    /// See [`key`](Self::key).
    pub checksum: String,
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

impl<K> TuneRecord<K> {
    /// The records namespace kind tunes are written under.
    pub const KIND: &str = "autotune";
}

/// A tune being recorded: stamped when it began, written when it ends.
#[derive(Debug)]
pub(crate) struct Recording {
    stamp: Stamp,
    started: cubecl_common::profile::Instant,
    dry_run: bool,
}

/// What [`Recording::finish`] needs to know of the answer.
pub(crate) struct Answer<'a, K> {
    pub table: &'a str,
    pub key: &'a K,
    pub checksum: &'a str,
    pub winner: usize,
}

impl Recording {
    /// Begin recording a tune, when the environment records. The steps are
    /// collected by the log context, which is created for the purpose when
    /// neither the logger nor the recorder asked for one.
    pub(crate) fn start(log_context: &mut Option<AutotuneLogContext>) -> Option<Self> {
        let stamp = records::stamp()?;
        log_context.get_or_insert_with(AutotuneLogContext::default);
        Some(Self {
            stamp,
            started: cubecl_common::profile::Instant::now(),
            dry_run: crate::dry_run::dry_run(),
        })
    }

    /// Write the record of the tune that just answered.
    pub(crate) fn finish<K: Serialize + Clone>(
        self,
        answer: Answer<'_, K>,
        log_context: Option<&AutotuneLogContext>,
    ) {
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
            table: answer.table.into(),
            key: answer.key.clone(),
            checksum: answer.checksum.into(),
            winner: answer.winner,
            trials,
            short_circuit,
            wall: self.started.elapsed(),
            dry_run: self.dry_run,
        };
        records::write_stamped(TuneRecord::<K>::KIND, self.stamp, &record);
    }
}
