use crate::memory_management::ManagedMemoryDescriptor;
use crate::server::{
    BufferBinding, CopyDescriptor, Handle, HeldRanges, LazyReadCount, LazyReadHold, LazyReadPlan,
    ServerError,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::sync::{Mutex, Once};

/// A read of device memory whose copy waits for the first access to its bytes, shared by those
/// bytes, their views, and the server that keeps it.
///
/// It holds no copy in flight: the server that starts it keeps the copy until the first access
/// asks for it, and then only the bytes the copy returned.
///
/// # Guarantees
///
/// - While a server keeps it, work that server queues to write the memory starts the read
///   first, so the bytes are the memory as it was when the server took it.
/// - It counts on its server's [`LazyReadCount`] from when its server keeps it until its bytes
///   are resolved or it is dropped, so work writing the memory meanwhile finds it.
pub struct LazyRead {
    /// A handle rather than a binding, so the memory stays free to move until a copy reads it.
    source: CopyDescriptor<Handle>,
    state: Mutex<LazyReadState>,
    held: Mutex<HeldRanges>,
    /// What the started read returned, resolved once for every holder.
    started: Once<Result<StartedPieces, ServerError>>,
}

enum LazyReadState {
    /// No server keeps it, so the read on first access copies the memory as it is then.
    Unkept,
    /// Its server keeps it, counted on `LazyReadCount`, and no work has written the memory since.
    Kept(LazyReadCount),
    /// Its server refused to keep it, for this error.
    Refused(ServerError),
    /// Work was about to write the memory, so its server started the read of the pieces at
    /// `offsets` first, and keeps the copy until the first access asks for it.
    Started {
        offsets: Vec<usize>,
        count: LazyReadCount,
    },
    /// The started read resolved into [`LazyRead::started`]: no copy of it is in flight.
    Resolved,
}

/// What a started read returned, each piece with its offset in the read.
struct StartedPieces {
    pieces: Vec<(usize, Bytes)>,
}

impl LazyRead {
    /// A read of `source` no server keeps yet.
    pub fn new(source: CopyDescriptor<Handle>) -> Self {
        Self {
            source,
            state: Mutex::new(LazyReadState::Unkept),
            held: Mutex::new(HeldRanges::default()),
            started: Once::new(),
        }
    }

    /// What it reads, bound for the server to read it.
    pub fn to_source(&self) -> CopyDescriptor {
        self.source.clone().binding()
    }

    /// Hold `range` of its bytes until the hold drops, for bytes or a view of them to read.
    ///
    /// # Assumptions
    ///
    /// - `range` lies within the bytes the whole read returns.
    pub fn hold(self: &Arc<Self>, range: Range<usize>) -> LazyReadHold {
        self.held.lock().hold(range.clone());
        LazyReadHold::new(self.clone(), range)
    }

    /// Whether `written` shares a byte with what it still holds of its memory.
    pub fn is_overwritten_by(&self, written: &BufferBinding) -> bool {
        written.memory.id() == self.descriptor().id() && self.plan().is_overwritten_by(written)
    }

    /// The copies of the bytes it still holds, one per piece of memory they lie in, with the
    /// offset of each in the read; none once nothing holds any.
    pub fn to_held_sources(&self) -> (Vec<CopyDescriptor>, Vec<usize>) {
        self.plan().to_sources()
    }

    /// Its server keeps it from now on, counted on `count`: work writing the memory starts it
    /// first.
    ///
    /// # Assumptions
    ///
    /// - It runs on the device thread of the memory's server, as the work that writes it does.
    pub fn keep(&self, count: &LazyReadCount) {
        let mut state = self.state.lock();
        if let LazyReadState::Unkept = *state {
            count.add(self.descriptor());
            *state = LazyReadState::Kept(count.clone());
        }
    }

    /// Whether work writing the memory still has to start it first.
    pub fn is_kept(&self) -> bool {
        matches!(*self.state.lock(), LazyReadState::Kept(_))
    }

    /// Whether a write started it and no access has its bytes yet, so its copy may still be
    /// in flight and work writing the memory from another stream has to wait for it.
    pub fn is_in_flight(&self) -> bool {
        matches!(*self.state.lock(), LazyReadState::Started { .. })
    }

    /// Its server queued the copies [`to_held_sources`](Self::to_held_sources) named, whose
    /// pieces lie at `offsets`, ahead of any work that writes the memory after it.
    ///
    /// # Assumptions
    ///
    /// - Its server keeps it, and keeps the copy for the first access to ask for.
    pub fn start(&self, offsets: Vec<usize>) {
        let mut state = self.state.lock();
        if let LazyReadState::Kept(count) = &*state {
            let count = count.clone();
            *state = LazyReadState::Started { offsets, count };
        }
    }

    /// The descriptor of the memory it reads.
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        self.source.handle.memory.descriptor()
    }
}

impl LazyRead {
    /// Its server refused to keep it, which the first access answers with.
    pub(crate) fn refuse(&self, error: ServerError) {
        *self.state.lock() = LazyReadState::Refused(error);
    }

    /// Whether a write started the read, which the first access then takes its bytes from.
    ///
    /// # Errors
    ///
    /// The error its server refused to keep it with.
    pub(crate) fn is_started(&self) -> Result<bool, ServerError> {
        match &*self.state.lock() {
            LazyReadState::Refused(error) => Err(error.clone()),
            LazyReadState::Started { .. } | LazyReadState::Resolved => Ok(true),
            LazyReadState::Unkept | LazyReadState::Kept(_) => Ok(false),
        }
    }

    /// The bytes in `range` of what the started read returned, which `resolve` waits for on the
    /// first call: the result itself, shared, for a whole piece, and an aligned copy for less.
    ///
    /// # Assumptions
    ///
    /// - [`is_started`](Self::is_started) answered `true`.
    /// - `range` is held, so the started read, which copied every held byte, covers it.
    /// - `resolve` returns the started copy's pieces, one per offset, in order.
    ///
    /// # Errors
    ///
    /// The started read's.
    pub(crate) fn started_bytes(
        &self,
        range: Range<usize>,
        resolve: impl FnOnce() -> Result<Vec<Bytes>, ServerError>,
    ) -> Result<Bytes, ServerError> {
        self.started
            .call_once(|| self.resolve(resolve))
            .as_ref()
            .map_err(Clone::clone)
            .map(|started| started.window(range))
    }

    pub(crate) fn let_go(&self, range: &Range<usize>) {
        self.held.lock().let_go(range);
    }

    fn resolve(
        &self,
        resolve: impl FnOnce() -> Result<Vec<Bytes>, ServerError>,
    ) -> Result<StartedPieces, ServerError> {
        let offsets = self.offsets();
        let pieces = resolve();
        self.let_go_of_count(LazyReadState::Resolved);
        pieces.map(|pieces| StartedPieces::new(offsets, pieces))
    }

    fn offsets(&self) -> Vec<usize> {
        match &*self.state.lock() {
            LazyReadState::Started { offsets, .. } => offsets.clone(),
            _ => Vec::new(),
        }
    }

    /// Move to `next`, letting go of its count if it held one.
    fn let_go_of_count(&self, next: LazyReadState) {
        let previous = core::mem::replace(&mut *self.state.lock(), next);
        if let LazyReadState::Kept(count) | LazyReadState::Started { count, .. } = previous {
            count.remove(self.descriptor());
        }
    }

    fn plan(&self) -> LazyReadPlan<'_> {
        LazyReadPlan {
            source: &self.source,
            held: self.held.lock().merged(),
        }
    }
}

impl StartedPieces {
    fn new(offsets: Vec<usize>, pieces: Vec<Bytes>) -> Self {
        assert_eq!(
            offsets.len(),
            pieces.len(),
            "a started read should return one piece per held range"
        );
        Self {
            pieces: offsets
                .into_iter()
                .zip(pieces.into_iter().map(Bytes::shared))
                .collect(),
        }
    }

    /// The bytes in `range` of the read: a whole piece shared, and an aligned copy of less,
    /// since a window that starts inside a piece loses the alignment typed reads rely on.
    fn window(&self, range: Range<usize>) -> Bytes {
        let (offset, bytes) = self
            .pieces
            .iter()
            .find(|(offset, bytes)| *offset <= range.start && range.end <= offset + bytes.len())
            .expect("a held range should lie in one started piece");
        let (start, end) = (range.start - offset, range.end - offset);
        let whole = start == 0 && end == bytes.len();
        match whole.then(|| bytes.view(start, end).ok()).flatten() {
            Some(window) => window,
            None => Bytes::from_bytes_vec(bytes[start..end].to_vec()),
        }
    }
}

impl Drop for LazyRead {
    fn drop(&mut self) {
        self.let_go_of_count(LazyReadState::Resolved);
    }
}

impl core::fmt::Debug for LazyRead {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyRead")
            .field("source", &self.source.handle)
            .finish()
    }
}
