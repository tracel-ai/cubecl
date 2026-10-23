use super::{CompilationSpans, EnvironmentRecords};
use crate::report::{Activity, SlowTune, UntypedTuneRecord};
use cubecl_environment::records::{MarkRecord, SessionId, Stamped};
use cubecl_server::compiler::CompilationRecord;
use std::ops::Range;
use std::time::Duration;

/// The records one session id stamped, each list oldest first, and the
/// stretches of its clock its compilations took: filtered and laid out once
/// for every fold over them, whether or not the file lists the session.
#[derive(Clone, Debug)]
pub struct StampedRecords<'a> {
    pub tunes: Vec<&'a Stamped<UntypedTuneRecord>>,
    pub compilations: Vec<&'a Stamped<CompilationRecord>>,
    pub marks: Vec<&'a Stamped<MarkRecord>>,
    pub compiling: CompilationSpans,
}

impl<'a> StampedRecords<'a> {
    /// The records of `records` that `session` stamped.
    pub fn new(records: &'a EnvironmentRecords, session: SessionId) -> Self {
        let compilations = stamped_by(&records.compilations, session);
        Self {
            compiling: CompilationSpans::new(
                stamped_by(&records.batches, session),
                compilations.iter().copied(),
            ),
            tunes: stamped_by(&records.tunes, session),
            compilations,
            marks: stamped_by(&records.marks, session),
        }
    }

    /// The tunes and compilations that started inside `window` of the
    /// session's clock, and what they took.
    pub fn activity(&self, window: Range<Duration>) -> Activity {
        let tunes: Vec<&Stamped<UntypedTuneRecord>> = self
            .tunes
            .iter()
            .copied()
            .filter(|tune| window.contains(&tune.stamp.offset))
            .collect();
        // Each tune's stretch, cut where the window ends: a compilation inside
        // any of them is the tunes', once however many of them overlap.
        let in_tunes: Vec<Range<Duration>> = tunes
            .iter()
            .map(|tune| tune.stamp.offset..(tune.stamp.offset + tune.record.wall).min(window.end))
            .collect();
        // The first of the longest, as the tunes ran.
        let slowest = tunes
            .iter()
            .min_by_key(|tune| std::cmp::Reverse(tune.record.wall))
            .map(|tune| SlowTune {
                table: tune.record.table.clone(),
                key: tune.record.entry.key.clone(),
                wall: tune.record.wall,
            });
        Activity {
            tunes: tunes.len() as u64,
            tuning: tunes.iter().map(|tune| tune.record.wall).sum(),
            compilations: self
                .compilations
                .iter()
                .filter(|trip| window.contains(&trip.stamp.offset))
                .count() as u64,
            compiling: self.compiling.within(&window),
            compiling_in_tunes: self.compiling.within_any(&in_tunes),
            slowest,
        }
    }
}

/// The records of `records` that `session` stamped.
fn stamped_by<R>(records: &[Stamped<R>], session: SessionId) -> Vec<&Stamped<R>> {
    records
        .iter()
        .filter(|record| record.stamp.session == session)
        .collect()
}
