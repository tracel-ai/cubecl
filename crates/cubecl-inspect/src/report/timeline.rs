use super::Activity;
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
    /// What started inside its [window](Self::window), inside nested spans
    /// included.
    ///
    /// Flattened, so a JSON reader finds `tunes` and `compiling` on the span.
    #[serde(flatten)]
    pub activity: Activity,
}

impl Span {
    /// The stretch of its session's clock it covers.
    pub fn window(&self) -> Range<Duration> {
        self.stamp.offset..self.stamp.offset + self.wall
    }
}
