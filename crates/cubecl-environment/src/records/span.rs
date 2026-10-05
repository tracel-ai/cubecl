#[cfg(not(native_cache))]
use super::disabled as session;
#[cfg(native_cache)]
use super::session;
use super::{Record, RecordEffect, Stamp, base::write_stamped};
use alloc::string::String;
use core::time::Duration;
use serde::{Deserialize, Serialize};

/// A stretch of the session in progress that a record describes: stamped when
/// it opens, timed on the session's clock, and recorded when it closes, so
/// everything stamped meanwhile lies inside it.
#[must_use = "a span records nothing until it is closed"]
#[derive(Debug)]
pub struct Span {
    start: Stamp,
}

impl Span {
    /// Opens a span now, starting the session if there is none. `None` when
    /// recording is off.
    pub fn new() -> Option<Self> {
        session::stamp().map(|start| Self { start })
    }

    /// The time since the span opened, on the session's clock. `None` when
    /// the environment switched sessions since: the span went with its own.
    pub fn elapsed(&self) -> Option<Duration> {
        session::offset(self.start.session).map(|now| now.saturating_sub(self.start.offset))
    }

    /// Records `record`, stamped when the span opened: written if the session
    /// is kept, as [`write()`](super::write) is. `false` when the session the
    /// span opened in is gone, or the write failed.
    pub fn close<R: Record + Serialize>(self, effect: RecordEffect, record: &R) -> bool {
        write_stamped(self.start, effect, record)
    }
}

/// A named span of the session in progress — a phase of the caller's work,
/// `load weights`, `plan turn 7` — recorded as a [`MarkRecord`] when it
/// drops. Everything else stamped meanwhile lies inside it on the session's
/// clock, which is what lets a reader say what a phase cost.
#[must_use = "a mark records the span until it is dropped"]
#[derive(Debug)]
pub struct Mark {
    label: String,
    span: Option<Span>,
}

impl Mark {
    /// Opens the span now. The label is converted only when the environment
    /// records, so a `&str` on a path that records nothing is never copied.
    pub fn new<S: Into<String>>(label: S) -> Self {
        let span = Span::new();
        let label = match span {
            Some(_) => label.into(),
            None => String::new(),
        };
        Self { label, span }
    }
}

impl Drop for Mark {
    fn drop(&mut self) {
        let Some(span) = self.span.take() else {
            return;
        };
        let Some(wall) = span.elapsed() else {
            return;
        };
        let record = MarkRecord {
            label: core::mem::take(&mut self.label),
            wall,
        };
        span.close(RecordEffect::Observed, &record);
    }
}

/// A named span of a session, as a [`Mark`] records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkRecord {
    /// What the span was.
    pub label: String,
    /// How long it lasted.
    pub wall: Duration,
}

impl Record for MarkRecord {
    const KIND: &'static str = "marks";
}
