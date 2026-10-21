use super::{EnvironmentRecords, StampedRecords};
use crate::report::{Activity, SessionRow, SessionTimeline, Span, Timeline};
use cubecl_environment::records::Session;
use std::time::Duration;

/// One session the file lists, and the records it stamped: what its row
/// in the summary and its timeline are folded from.
#[derive(Clone, Debug)]
pub struct SessionRecords<'a> {
    session: &'a Session,
    stamped: StampedRecords<'a>,
}

impl<'a> SessionRecords<'a> {
    /// The records of `records` that `session` stamped.
    pub fn new(session: &'a Session, records: &'a EnvironmentRecords) -> Self {
        Self {
            session,
            stamped: StampedRecords::new(records, session.id),
        }
    }
}

/// What the session's records add up to.
impl From<&SessionRecords<'_>> for SessionRow {
    fn from(records: &SessionRecords<'_>) -> Self {
        let stamped = &records.stamped;
        let tunes = stamped
            .tunes
            .iter()
            .map(|trace| trace.stamp.offset + trace.record.wall);
        // A phase after the last compile — a walk served from the store —
        // still ran.
        let marks = stamped
            .marks
            .iter()
            .map(|mark| mark.stamp.offset + mark.record.wall);
        let span = tunes
            .chain(marks)
            .chain([stamped.compiling.end()])
            .max()
            .unwrap_or_default();
        Self {
            session: records.session.clone(),
            activity: stamped.activity(Duration::ZERO..Duration::MAX),
            span,
        }
    }
}

/// The session's marks laid out, each with what started inside it.
impl From<&SessionRecords<'_>> for SessionTimeline {
    fn from(records: &SessionRecords<'_>) -> Self {
        let mut spans: Vec<Span> = records
            .stamped
            .marks
            .iter()
            .map(|mark| Span {
                stamp: mark.stamp,
                label: mark.record.label.clone(),
                wall: mark.record.wall,
                depth: 0,
                activity: Activity::default(),
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
            spans[index].activity = records.stamped.activity(spans[index].window());
        }
        Self {
            session: records.session.clone(),
            spans,
        }
    }
}

/// Each session's marks.
impl From<&[SessionRecords<'_>]> for Timeline {
    fn from(sessions: &[SessionRecords<'_>]) -> Self {
        Self {
            sessions: sessions.iter().map(SessionTimeline::from).collect(),
        }
    }
}
