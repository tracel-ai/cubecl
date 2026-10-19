use super::SessionRecords;
use cubecl_environment::records::{Session, Stamp};
use serde::Serialize;
use std::ops::Range;
use std::time::Duration;

/// The marks each session placed, in order, with the tuning and compiling
/// that happened inside each: where a session's time went, phase by phase.
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
    /// The time they took: a batch's wall, however many threads compiled it.
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

impl Span {
    /// The stretch of its session's clock it covers.
    fn window(&self) -> Range<Duration> {
        self.stamp.offset..self.stamp.offset + self.wall
    }
}

impl SessionTimeline {
    /// Lay out the session's marks, folding in the tunes and compilations
    /// stamped inside each, and the time its compiling took inside each.
    pub fn new(records: SessionRecords<'_>) -> Self {
        let mut spans: Vec<Span> = records
            .marks()
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
            let start = spans[index].stamp.offset;
            spans[index].depth = spans[..index]
                .iter()
                .filter(|outer| outer.window().contains(&start))
                .count();
        }
        let compiling = records.compiling();
        for span in &mut spans {
            let window = span.window();
            for tune in records
                .tunes()
                .filter(|tune| window.contains(&tune.stamp.offset))
            {
                span.tunes += 1;
                span.tuning += tune.record.wall;
                if span
                    .slowest
                    .as_ref()
                    .is_none_or(|slowest| tune.record.wall > slowest.wall)
                {
                    span.slowest = Some(SlowTune {
                        table: tune.record.table.clone(),
                        key: tune.record.entry.key.clone(),
                        wall: tune.record.wall,
                    });
                }
            }
            span.compilations = records
                .compilations()
                .filter(|trip| window.contains(&trip.stamp.offset))
                .count() as u64;
            span.compiling = compiling.within(window);
        }
        Self {
            session: records.session().clone(),
            spans,
        }
    }
}
