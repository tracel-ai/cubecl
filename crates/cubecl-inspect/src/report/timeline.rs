use super::Activity;
use crate::records::SessionRecords;
use cubecl_environment::records::{Session, Stamp};
use serde::Serialize;
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
    /// What started inside it, inside nested spans included.
    ///
    /// Flattened, so a JSON reader finds `tunes` and `compiling` on the span.
    #[serde(flatten)]
    pub activity: Activity,
}

impl SessionTimeline {
    /// Lay out the session's marks, folding in what started inside each.
    pub(crate) fn new(records: SessionRecords<'_>) -> Self {
        let mut spans: Vec<Span> = records
            .marks()
            .map(|mark| {
                let start = mark.stamp.offset;
                Span {
                    stamp: mark.stamp,
                    label: mark.record.label.clone(),
                    wall: mark.record.wall,
                    depth: 0,
                    activity: records.activity(start..start + mark.record.wall),
                }
            })
            .collect();
        // Opened first comes first; of two opened at once, the longer
        // encloses the other.
        spans.sort_by_key(|span| (span.stamp.offset, std::cmp::Reverse(span.wall)));
        for index in 0..spans.len() {
            let start = spans[index].stamp.offset;
            spans[index].depth = spans[..index]
                .iter()
                .filter(|outer| {
                    (outer.stamp.offset..outer.stamp.offset + outer.wall).contains(&start)
                })
                .count();
        }
        Self {
            session: records.session().clone(),
            spans,
        }
    }
}
