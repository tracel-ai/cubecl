use alloc::vec::Vec;
use cubecl_environment::stream::StreamId;

use crate::server::ServerError;

/// The errors queued on one pooled backend stream, each remembering the logical
/// stream it belongs to.
///
/// Several logical streams share a pooled stream: [`stream_index`] folds the id
/// space onto `max_streams` slots, so thread A and thread B can land on the same
/// backend stream. Errors are lazy — a failed launch is queued and surfaces at
/// the next flush, sync or profile end — so a single untagged queue per slot
/// lets B drain A's error: B panics on a kernel it never launched while A reads
/// back a zeroed buffer as if its launch had succeeded.
///
/// Tagging fixes the attribution: [`push`](Self::push) records the stream that
/// caused the error and only that stream ever takes it. Errors a slot cannot
/// attribute — a failed submission, a mapping failure, a validation message
/// from the driver — go in with [`push_shared`](Self::push_shared) and surface
/// on whichever stream flushes next, which is what every error did before.
///
/// [`stream_index`]: super::stream_index
#[derive(Debug, Default)]
pub struct StreamErrors {
    entries: Vec<(Option<StreamId>, ServerError)>,
}

impl StreamErrors {
    /// Queue an error caused by `owner`, for `owner` alone to surface.
    pub fn push(&mut self, owner: StreamId, error: ServerError) {
        self.entries.push((Some(owner), error));
    }

    /// Queue an error the pooled stream cannot attribute to one of the logical
    /// streams sharing it, so the next flush of any of them surfaces it.
    pub fn push_shared(&mut self, error: ServerError) {
        self.entries.push((None, error));
    }

    /// [`push_shared`](Self::push_shared) for a batch of errors.
    pub fn extend_shared(&mut self, errors: impl IntoIterator<Item = ServerError>) {
        self.entries
            .extend(errors.into_iter().map(|error| (None, error)));
    }

    /// Whether the queue holds nothing at all, for any stream.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `owner` has anything to surface: its own errors plus the shared
    /// ones. `None` sees the shared errors only.
    pub fn any(&self, owner: Option<StreamId>) -> bool {
        self.entries
            .iter()
            .any(|(entry, _)| owned_by(*entry, owner))
    }

    /// The errors `owner` would surface, left in the queue.
    pub fn peek(&self, owner: Option<StreamId>) -> Vec<ServerError> {
        self.entries
            .iter()
            .filter(|(entry, _)| owned_by(*entry, owner))
            .map(|(_, error)| error.clone())
            .collect()
    }

    /// Take the errors `owner` surfaces, leaving the other streams' behind.
    pub fn take(&mut self, owner: Option<StreamId>) -> Vec<ServerError> {
        let (taken, kept) = core::mem::take(&mut self.entries)
            .into_iter()
            .partition(|(entry, _)| owned_by(*entry, owner));

        self.entries = kept;
        taken.into_iter().map(|(_, error)| error).collect()
    }

    /// Take everything, whoever it belongs to. For the paths that speak for the
    /// whole device rather than for one stream.
    pub fn take_all(&mut self) -> Vec<ServerError> {
        core::mem::take(&mut self.entries)
            .into_iter()
            .map(|(_, error)| error)
            .collect()
    }
}

/// Whether `owner` surfaces an entry queued for `entry`: its own, plus the
/// shared entries no stream owns.
fn owned_by(entry: Option<StreamId>, owner: Option<StreamId>) -> bool {
    match entry {
        Some(entry) => Some(entry) == owner,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn error(reason: &str) -> ServerError {
        ServerError::Generic {
            reason: reason.to_string(),
            backtrace: Default::default(),
        }
    }

    fn reasons(errors: Vec<ServerError>) -> Vec<alloc::string::String> {
        errors
            .into_iter()
            .map(|error| match error {
                ServerError::Generic { reason, .. } => reason,
                other => panic!("unexpected error: {other}"),
            })
            .collect()
    }

    #[test]
    fn a_stream_takes_its_own_errors_only() {
        let first = StreamId { value: 1 };
        let second = StreamId { value: 2 };

        let mut errors = StreamErrors::default();
        errors.push(first, error("first"));
        errors.push(second, error("second"));

        assert_eq!(reasons(errors.take(Some(first))), ["first"]);
        assert!(errors.any(Some(second)), "the other stream keeps its error");
        assert_eq!(reasons(errors.take(Some(second))), ["second"]);
        assert!(errors.is_empty());
    }

    #[test]
    fn shared_errors_surface_on_the_next_flush() {
        let stream = StreamId { value: 1 };

        let mut errors = StreamErrors::default();
        errors.push_shared(error("submission"));

        assert!(errors.any(None));
        assert_eq!(reasons(errors.peek(Some(stream))), ["submission"]);
        assert_eq!(reasons(errors.take(Some(stream))), ["submission"]);
        assert!(errors.is_empty(), "a shared error is taken once");
    }

    #[test]
    fn an_unowned_flush_leaves_the_owned_errors() {
        let stream = StreamId { value: 1 };

        let mut errors = StreamErrors::default();
        errors.push(stream, error("launch"));
        errors.push_shared(error("submission"));

        assert_eq!(reasons(errors.take(None)), ["submission"]);
        assert_eq!(reasons(errors.take_all()), ["launch"]);
    }
}
