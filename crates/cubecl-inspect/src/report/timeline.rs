use super::TuneTrace;
use cubecl_environment::records::{MarkRecord, Session, Stamp, Stamped};
use cubecl_server::compiler::CompilationRecord;
use cubecl_server::memory_management::MemoryRecord;
use serde::Serialize;
use std::time::Duration;

/// The marks each session placed, in order, with the tuning and compiling
/// that happened inside each: where a build's time went, phase by phase.
#[derive(Clone, Debug, Serialize)]
pub struct Timeline {
    pub sessions: Vec<SessionTimeline>,
}

/// One session's marks.
#[derive(Clone, Debug, Serialize)]
pub struct SessionTimeline {
    pub session: Session,
    /// In the order they opened; a span inside another follows it, one
    /// [`depth`](Span::depth) deeper.
    pub spans: Vec<Span>,
}

/// One mark, and what was recorded inside it.
#[derive(Clone, Debug, Serialize)]
pub struct Span {
    pub stamp: Stamp,
    pub label: String,
    pub wall: Duration,
    /// How many marks enclose it.
    pub depth: usize,
    /// Tunes that started inside it, the ones inside nested spans included.
    pub tunes: u64,
    pub tuning: Duration,
    /// Compilations that started inside it, likewise.
    pub compilations: u64,
    pub compiling: Duration,
    /// The longest tune inside it.
    pub slowest: Option<SlowTune>,
}

/// The tune a span spent longest on.
#[derive(Clone, Debug, Serialize)]
pub struct SlowTune {
    /// The table's namespace.
    pub table: String,
    pub key: ciborium::Value,
    pub wall: Duration,
}

/// Every memory snapshot the file's sessions recorded, in the order they were
/// taken.
#[derive(Clone, Debug, Serialize)]
pub struct MemorySnapshots {
    pub snapshots: Vec<Stamped<MemoryRecord>>,
}

impl Span {
    /// Whether `stamp` was taken inside this span, on its session's clock.
    fn contains(&self, stamp: &Stamp) -> bool {
        stamp.session == self.stamp.session
            && (self.stamp.offset..self.stamp.offset + self.wall).contains(&stamp.offset)
    }
}

impl SessionTimeline {
    /// Lay `marks` out for `session`, folding in the `tunes` and
    /// `compilations` stamped inside each.
    pub fn new(
        session: Session,
        marks: &[Stamped<MarkRecord>],
        tunes: &[TuneTrace],
        compilations: &[Stamped<CompilationRecord>],
    ) -> Self {
        let mut spans: Vec<Span> = marks
            .iter()
            .filter(|mark| mark.stamp.session == session.id)
            .map(|mark| Span {
                stamp: mark.stamp,
                label: mark.record.label.clone(),
                wall: mark.record.wall,
                depth: 0,
                tunes: 0,
                tuning: Duration::ZERO,
                compilations: 0,
                compiling: Duration::ZERO,
                slowest: None,
            })
            .collect();
        // Opened first comes first; of two opened at once, the longer
        // encloses the other.
        spans.sort_by_key(|span| (span.stamp.offset, std::cmp::Reverse(span.wall)));
        for index in 0..spans.len() {
            let stamp = spans[index].stamp;
            spans[index].depth = spans[..index]
                .iter()
                .filter(|outer| outer.contains(&stamp))
                .count();
        }
        for span in &mut spans {
            let inside: Vec<&TuneTrace> = tunes
                .iter()
                .filter(|tune| span.contains(&tune.stamp))
                .collect();
            for tune in inside {
                span.tunes += 1;
                span.tuning += tune.record.wall;
                if span
                    .slowest
                    .as_ref()
                    .is_none_or(|slowest| tune.record.wall > slowest.wall)
                {
                    span.slowest = Some(SlowTune {
                        table: tune.record.table.clone(),
                        key: tune.record.key.clone(),
                        wall: tune.record.wall,
                    });
                }
            }
            let compiled: Vec<Duration> = compilations
                .iter()
                .filter(|trip| span.contains(&trip.stamp))
                .map(|trip| trip.record.outcome.duration())
                .collect();
            span.compilations = compiled.len() as u64;
            span.compiling = compiled.iter().sum();
        }
        Self { session, spans }
    }
}
