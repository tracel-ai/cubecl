//! What a tune leaves in the environment beside its answer.
//!
//! The table stores a key's decided candidate and every candidate's result, ranked. What
//! it cannot say is how the tune went: which candidates ran and in what order,
//! what each cost to compile and benchmark, whether the tune stopped early,
//! and how long the key took from its miss to its answer. A [`TuneRecord`] is
//! that account, written to [`cubecl_environment::records`] once per tune.

use crate::tune::cache::table::TuneTable;
use crate::tune::{AutotuneKey, AutotuneLogContext, AutotuneLogEvent, PersistentCacheKey};
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
    /// Read under its earlier name too, which environments already saved hold.
    #[serde(alias = "winner")]
    pub decided_index: usize,
    /// Each candidate's cost, in the order the tune reports them (Arrival Usage): run order on the
    /// fixed-count pass, batch order on the adaptive one, where a candidate its group's patience
    /// skipped or that a short circuit left unreached costs zero. A skipped candidate the batch
    /// measures after all, once every measured one failed, has a second trial for that run.
    pub trials: Vec<Trial>,
    /// The candidate that met the time limit and ended the tune before the
    /// rest of the plan ran.
    pub short_circuit: Option<String>,
    /// From the cache miss to the answer committed.
    pub wall: Duration,
    /// Whether the tune ran under a process mode that discards launches, where
    /// they compile but do not execute. Stored under its earlier name, which
    /// environments already saved hold.
    #[serde(rename = "dry_run")]
    pub launches_discarded: bool,
    /// Whether the table took the answer. One that measured nothing, or that
    /// was tuned with the cache disabled, answers this process alone: the
    /// table holds another answer to the key, or none.
    pub stored: bool,
    /// Every candidate in the order the plan releases them, whether or not
    /// it ran: the priorities that decided the order, read when the tune
    /// began, in that order (Arrival Usage). Empty in a record written before plans were kept.
    #[serde(default)]
    pub plan: Vec<PlannedCandidate>,
}

/// One candidate's place in the plan of a tune.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedCandidate {
    /// The candidate's name.
    pub name: String,
    /// Its index in the tunable set, as the record's decided index names it.
    pub index: usize,
    /// The batch the plan releases it in, counting from 0. A tune stops
    /// after the first batch with a valid result, so later batches only run
    /// when every earlier one failed. `None` when each of its groups gave it
    /// a negative priority for this key.
    pub batch: Option<usize>,
    /// Its place in each group it belongs to, in the order it joined them
    /// (Arrival Usage); empty for a candidate in no group, which trails the
    /// first batch.
    pub groups: Vec<Placement>,
}

/// A tunable's place in one of its [groups](crate::tune::TuneGroup), as one key priced it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Placement {
    /// The group's name.
    pub group: String,
    /// Whether the group is [ordered](crate::tune::TuneGroup::ordered): its members' priorities order one round
    /// rather than cutting it into rounds.
    pub ordered: bool,
    /// The group's own priority: higher groups run first.
    pub group_priority: i8,
    /// The tunable's priority within the group; negative skips it.
    pub priority: i8,
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

/// Records a tune: stamped when it began, written when it ends. Every call is a no-op when the
/// environment records nothing.
#[derive(Debug)]
pub struct TuneRecorder<K> {
    open: Option<OpenRecording<K>>,
}

/// What a [`TuneRecorder`] holds while the environment records.
#[derive(Debug)]
pub struct OpenRecording<K> {
    pub span: Span,
    pub table: String,
    pub entry: PersistentCacheKey<K>,
    pub launches_discarded: bool,
    /// Every candidate, in the order the plan releases them (Arrival Usage).
    pub plan: Vec<PlannedCandidate>,
}

/// What [`TuneRecorder::record_answer`] needs to know of the answer.
pub struct Answer {
    pub decided_index: usize,
    /// Whether the table took it: see [`TuneRecord::stored`].
    pub stored: bool,
}

impl<K: AutotuneKey> TuneRecorder<K> {
    /// Begin recording the tune of `key` in `table`. The key is cloned only when the environment
    /// records.
    pub fn new(table: &TuneTable, key: &K, checksum: &str) -> Self {
        let open = Span::new().map(|span| OpenRecording {
            span,
            table: table.namespace.clone(),
            entry: PersistentCacheKey {
                key: key.clone(),
                checksum: checksum.into(),
            },
            launches_discarded: crate::execution::ProcessMode::current().discards_launches(),
            plan: Vec::new(),
        });
        Self { open }
    }

    /// Whether the tune is recorded, and so has to track its steps: the record's trials are the
    /// ones the log context collects.
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Keep the plan the tune runs, as `account` lists it: called only when the environment
    /// records.
    pub fn record_plan(&mut self, account: impl FnOnce() -> Vec<PlannedCandidate>) {
        let Some(open) = &mut self.open else {
            return;
        };
        open.plan = account();
    }

    /// Write the record of the tune that just answered.
    pub fn record_answer(self, answer: Answer, log_context: Option<&AutotuneLogContext>) {
        let Some(open) = self.open else {
            return;
        };
        // A tune the environment switched away from went with its session.
        let Some(wall) = open.span.elapsed() else {
            return;
        };
        let (trials, short_circuit) = trials(log_context);
        let record = TuneRecord {
            table: open.table,
            entry: open.entry,
            decided_index: answer.decided_index,
            trials,
            short_circuit,
            wall,
            launches_discarded: open.launches_discarded,
            stored: answer.stored,
            plan: open.plan,
        };
        open.span.close(effect_when_stored(record.stored), &record);
    }
}

/// The trials `log_context` collected, in the order they ran, and the candidate that short
/// circuited the tune, if one did.
fn trials(log_context: Option<&AutotuneLogContext>) -> (Vec<Trial>, Option<String>) {
    let events = log_context.iter().flat_map(|context| &context.events);
    let trials = events
        .clone()
        .filter_map(|event| match event {
            AutotuneLogEvent::TuningStep(name, wall) => Some(Trial {
                name: name.clone(),
                wall: *wall,
            }),
            AutotuneLogEvent::ShortCircuit(_) => None,
        })
        .collect();
    let short_circuit = events
        .filter_map(|event| match event {
            AutotuneLogEvent::ShortCircuit(name) => Some(name.clone()),
            AutotuneLogEvent::TuningStep(..) => None,
        })
        .next_back();
    (trials, short_circuit)
}

/// A stored decision is the environment changing; an answer kept in memory is not.
fn effect_when_stored(stored: bool) -> RecordEffect {
    if stored {
        RecordEffect::Changed
    } else {
        RecordEffect::Observed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    /// A record saved before the decided candidate had its name still reads.
    #[test]
    fn a_record_saved_under_the_earlier_name_reads() {
        let record = TuneRecord {
            table: "table".to_string(),
            entry: PersistentCacheKey {
                key: "key".to_string(),
                checksum: "checksum".to_string(),
            },
            decided_index: 2,
            trials: Vec::new(),
            short_circuit: None,
            wall: Duration::ZERO,
            launches_discarded: false,
            stored: true,
            plan: Vec::new(),
        };
        let mut value = serde_json::to_value(&record).unwrap();
        let fields = value
            .as_object_mut()
            .expect("a record should serialize as a map");
        let decided = fields
            .remove("decided_index")
            .expect("a record should name its decided candidate");
        fields.insert("winner".to_string(), decided);

        let read: TuneRecord<String> = serde_json::from_value(value).unwrap();

        assert_eq!(read.decided_index, 2);
    }

    #[test]
    fn a_closed_recorder_never_lists_the_plan() {
        let mut recorder = TuneRecorder::<String> { open: None };

        recorder.record_plan(|| panic!("a closed recorder should not list the plan"));
    }

    fn short_circuited() -> AutotuneLogContext {
        let step = |name: &str, micros| {
            AutotuneLogEvent::TuningStep(name.to_string(), Duration::from_micros(micros))
        };
        AutotuneLogContext {
            events: vec![
                step("first", 1),
                step("second", 2),
                AutotuneLogEvent::ShortCircuit("second".to_string()),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn the_trials_are_the_steps_in_order() {
        let (ran, _) = trials(Some(&short_circuited()));

        let names: Vec<_> = ran.iter().map(|trial| trial.name.as_str()).collect();
        assert_eq!(names, ["first", "second"]);
    }

    #[test]
    fn the_short_circuit_names_the_candidate_that_stopped_the_tune() {
        let (_, short_circuit) = trials(Some(&short_circuited()));

        assert_eq!(short_circuit.as_deref(), Some("second"));
    }

    #[test]
    fn a_tune_that_logged_nothing_has_no_trial() {
        assert_eq!(trials(None), (Vec::new(), None));
    }

    #[test]
    fn a_stored_answer_changes_the_environment() {
        assert_eq!(effect_when_stored(true), RecordEffect::Changed);
    }

    #[test]
    fn an_answer_kept_in_memory_only_observes_the_environment() {
        assert_eq!(effect_when_stored(false), RecordEffect::Observed);
    }
}
