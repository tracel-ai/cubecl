use super::{ServiceStream, StreamMode};
use alloc::vec::Vec;
use cubecl_environment::sync::{AtomicUsize, Mutex, Ordering};

/// One live [`StreamModeOverride`](super::StreamModeOverride).
#[derive(Debug)]
struct StreamModeEntry {
    /// The override's id, which it unsets its entry by.
    pub id: usize,
    /// The stream it sets the mode of.
    pub stream: ServiceStream,
    /// The mode it sets.
    pub mode: StreamMode,
}

/// The live [`StreamModeOverride`](super::StreamModeOverride)s: what sets each stream's mode, and how
/// many set each mode, so a launch looks them up only when one sets the mode
/// its process mode does not — a measurement under no override, which sets
/// the mode every stream already has, costs other launches nothing.
pub struct StreamModeSetter {
    /// The live overrides, in the order they were set: searched newest first
    /// for a stream's, and one is removed by its id as it drops.
    entries: Mutex<Vec<StreamModeEntry>>,
    /// How many live overrides set each mode, by [`slot`].
    live_counts: [AtomicUsize; 2],
    /// The id the next override takes.
    next_id: AtomicUsize,
}

/// The process's stream modes.
pub static STREAM_MODES: StreamModeSetter = StreamModeSetter {
    entries: Mutex::new(Vec::new()),
    live_counts: [AtomicUsize::new(0), AtomicUsize::new(0)],
    next_id: AtomicUsize::new(0),
};

impl StreamModeSetter {
    /// Puts `stream` in `mode` until it is [unset](Self::unset) by the id
    /// it hands back.
    pub fn set(&self, mode: StreamMode, stream: ServiceStream) -> usize {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock();
        entries.push(StreamModeEntry { id, stream, mode });
        self.live_counts[slot(mode)].fetch_add(1, Ordering::Release);
        id
    }

    /// `stream`'s mode now: what the newest live override sets for it, or
    /// `default` when none does.
    pub fn mode(&self, stream: ServiceStream, default: StreamMode) -> StreamMode {
        // An override setting the default changes nothing, unless it is
        // newer than one setting the other mode on the same stream, and then
        // there is one of those to look for.
        if self.live_counts[slot(opposite(default))].load(Ordering::Acquire) == 0 {
            return default;
        }
        self.entries
            .lock()
            .iter()
            .rev()
            .find(|entry| entry.stream == stream)
            .map_or(default, |entry| entry.mode)
    }

    /// Removes the override `id` [set](Self::set) `mode`.
    pub fn unset(&self, id: usize, mode: StreamMode) {
        let mut entries = self.entries.lock();
        entries.retain(|entry| entry.id != id);
        self.live_counts[slot(mode)].fetch_sub(1, Ordering::Release);
    }
}

/// `mode`'s slot in a [`StreamModeSetter`]'s live counts.
fn slot(mode: StreamMode) -> usize {
    match mode {
        StreamMode::Execute => 0,
        StreamMode::Discard => 1,
    }
}

/// The mode other than `mode`.
fn opposite(mode: StreamMode) -> StreamMode {
    match mode {
        StreamMode::Execute => StreamMode::Discard,
        StreamMode::Discard => StreamMode::Execute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::{DeviceId, ServiceId};
    use cubecl_environment::stream::StreamId;

    /// A setter no other test shares.
    fn setter() -> StreamModeSetter {
        StreamModeSetter {
            entries: Mutex::new(Vec::new()),
            live_counts: [AtomicUsize::new(0), AtomicUsize::new(0)],
            next_id: AtomicUsize::new(0),
        }
    }

    /// Stream `value` of the service of device `index`.
    fn stream(index: u16, value: u64) -> ServiceStream {
        ServiceStream {
            service: ServiceId::of::<()>(DeviceId {
                type_id: 0,
                index_id: index,
            }),
            stream: StreamId { value },
        }
    }

    #[test]
    fn a_stream_no_override_sets_has_the_default() {
        assert_eq!(
            setter().mode(stream(0, 1), StreamMode::Discard),
            StreamMode::Discard
        );
    }

    #[test]
    fn a_stream_has_the_mode_its_override_sets() {
        let setter = setter();
        setter.set(StreamMode::Execute, stream(0, 1));
        assert_eq!(
            setter.mode(stream(0, 1), StreamMode::Discard),
            StreamMode::Execute
        );
    }

    #[test]
    fn a_stream_has_the_default_once_its_override_is_unset() {
        let setter = setter();
        let id = setter.set(StreamMode::Execute, stream(0, 1));
        setter.unset(id, StreamMode::Execute);
        assert_eq!(
            setter.mode(stream(0, 1), StreamMode::Discard),
            StreamMode::Discard
        );
    }

    #[test]
    fn each_stream_mode_is_the_opposite_of_the_other() {
        assert_eq!(opposite(StreamMode::Execute), StreamMode::Discard);
    }

    #[test]
    fn each_stream_mode_has_a_slot_of_its_own() {
        assert_ne!(slot(StreamMode::Execute), slot(StreamMode::Discard));
    }
}
