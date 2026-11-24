use crate::server::{BufferBinding, CopyDescriptor, Handle, ServerError};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::future::DynFut;
use cubecl_environment::sync::{Mutex, Once};

/// A read of device memory whose copy waits for the first access to its bytes, shared by those
/// bytes, their views, and the server that keeps it.
///
/// # Guarantees
///
/// - While a server keeps it, work that server queues to write the memory starts the read
///   first, so the bytes are the memory as it was when the server took it.
/// - It counts on its memory's descriptor from when its server keeps it until its bytes are
///   resolved or it is dropped, so work writing the memory meanwhile finds it.
pub struct LazyRead {
    /// A handle rather than a binding, so the memory stays free to move until a copy reads it.
    source: CopyDescriptor<Handle>,
    state: Mutex<LazyReadState>,
    /// Arrival usage: the byte ranges of the read its live bytes and views hold, so a write
    /// beside all of them leaves it be and a start copies only their span.
    held: Mutex<Vec<Range<usize>>>,
    /// What the started read returned and its offset in the read, resolved once for every
    /// holder.
    started: Once<Result<(usize, Bytes), ServerError>>,
}

enum LazyReadState {
    /// No server keeps it, so the read on first access copies the memory as it is then.
    Unkept,
    /// Its server keeps it, and no work has written the memory since.
    Kept,
    /// Its server refused to keep it, for this error.
    Refused(ServerError),
    /// Work was about to write the memory, so its server started the read of the bytes from
    /// `offset` on first. `read` is `None` once taken to be resolved into
    /// [`LazyRead::started`].
    Started {
        offset: usize,
        read: Option<DynFut<Result<Vec<Bytes>, ServerError>>>,
    },
    /// The started read resolved into [`LazyRead::started`]: no copy of it is in flight.
    Resolved,
}

impl LazyRead {
    /// A read of `source` no server keeps yet.
    pub fn new(source: CopyDescriptor<Handle>) -> Self {
        Self {
            source,
            state: Mutex::new(LazyReadState::Unkept),
            held: Mutex::new(Vec::new()),
            started: Once::new(),
        }
    }

    /// What it reads, bound for the server to read it.
    pub fn to_source(&self) -> CopyDescriptor {
        self.source.clone().binding()
    }

    /// Hold `range` of its bytes until the hold drops, for bytes or a view of them to read.
    pub fn hold(self: &Arc<Self>, range: Range<usize>) -> LazyReadHold {
        self.held.lock().push(range.clone());
        LazyReadHold {
            read: self.clone(),
            range,
        }
    }

    /// Whether `written` overlaps the bytes it still holds.
    pub fn is_overwritten_by(&self, written: &BufferBinding) -> bool {
        let Some(span) = self.held_span() else {
            return false;
        };
        let read = self.source.handle.to_binding().range();
        let read = match span == (0..self.len()) {
            true => read,
            false => read.start + span.start as u64..read.start + span.end as u64,
        };
        let written = written.range();
        read.start < written.end && written.start < read.end
    }

    /// The bytes it still holds, bound for the server to read them, and their offset in it.
    ///
    /// # Assumptions
    ///
    /// - A span narrower than the whole read is held only through a view, which only a
    ///   contiguous source gives.
    pub fn to_held_source(&self) -> (CopyDescriptor, usize) {
        let source = self.to_source();
        let span = self.held_span().unwrap_or(0..self.len());
        if span == (0..self.len()) {
            return (source, 0);
        }
        let mut handle = source.handle;
        let base = handle.offset_start.unwrap_or(0);
        handle.offset_end = Some(handle.size - (base + span.end as u64));
        handle.offset_start = Some(base + span.start as u64);
        let narrowed = CopyDescriptor::new(handle, [span.len()].into(), [1].into(), 1);
        (narrowed, span.start)
    }

    /// Its server keeps it from now on: work writing the memory starts it first.
    ///
    /// # Assumptions
    ///
    /// - It runs on the device thread of the memory's server, as the work that writes it does.
    pub fn keep(&self) {
        let mut state = self.state.lock();
        if let LazyReadState::Unkept = *state {
            *state = LazyReadState::Kept;
            self.source.handle.memory.descriptor().lazy_read_kept();
        }
    }

    /// Whether work writing the memory still has to start it first.
    pub fn is_kept(&self) -> bool {
        matches!(*self.state.lock(), LazyReadState::Kept)
    }

    /// Whether a write started it and its copy may still be in flight, so work writing the
    /// memory from another stream has to wait for it.
    pub fn is_in_flight(&self) -> bool {
        matches!(*self.state.lock(), LazyReadState::Started { .. })
    }

    /// `read` is the server's read of the bytes from `offset` on that
    /// [`to_held_source`](Self::to_held_source) named, queued ahead of any work that writes the
    /// memory after it, which the first access takes its bytes from.
    ///
    /// # Assumptions
    ///
    /// - Its server keeps it.
    pub fn start(&self, offset: usize, read: DynFut<Result<Vec<Bytes>, ServerError>>) {
        *self.state.lock() = LazyReadState::Started {
            offset,
            read: Some(read),
        };
    }

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
            LazyReadState::Unkept | LazyReadState::Kept => Ok(false),
        }
    }

    /// The bytes in `range` of what the started read returned, copied out so every holder takes
    /// its own.
    ///
    /// # Assumptions
    ///
    /// - [`is_started`](Self::is_started) answered `true`.
    /// - `range` is held, so the started read, which copied every held byte, covers it.
    ///
    /// # Errors
    ///
    /// The started read's.
    pub(crate) fn started_bytes(&self, range: Range<usize>) -> Result<Bytes, ServerError> {
        let started = self.started.call_once(|| {
            let (offset, read) = match &mut *self.state.lock() {
                LazyReadState::Started { offset, read } => (*offset, read.take()),
                _ => (0, None),
            };
            let read = read.expect("a started read should be resolved once");
            let bytes = cubecl_environment::future::reader::read_sync(read)
                .map(|mut bytes| (offset, bytes.remove(0)));
            *self.state.lock() = LazyReadState::Resolved;
            self.source.handle.memory.descriptor().lazy_read_let_go();
            bytes
        });
        match started {
            Ok((offset, bytes)) => Ok(Bytes::from_bytes_vec(
                bytes[range.start - offset..range.end - offset].to_vec(),
            )),
            Err(error) => Err(error.clone()),
        }
    }

    /// How many bytes the whole read returns.
    fn len(&self) -> usize {
        self.source.shape.iter().product::<usize>() * self.source.elem_size
    }

    /// The span of the bytes still held, `None` once nothing holds any.
    fn held_span(&self) -> Option<Range<usize>> {
        let held = self.held.lock();
        let start = held.iter().map(|range| range.start).min()?;
        let end = held.iter().map(|range| range.end).max()?;
        Some(start..end)
    }

    fn let_go(&self, range: &Range<usize>) {
        let mut held = self.held.lock();
        if let Some(index) = held.iter().position(|held| held == range) {
            held.swap_remove(index);
        }
    }
}

/// A range of a [`LazyRead`]'s bytes, held for bytes or a view of them to read until dropped.
pub struct LazyReadHold {
    read: Arc<LazyRead>,
    range: Range<usize>,
}

impl LazyReadHold {
    /// The read it holds bytes of.
    pub fn read(&self) -> &Arc<LazyRead> {
        &self.read
    }

    /// The bytes of the read it holds.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// A hold of `range` within these bytes, for a view of them.
    pub fn narrowed(&self, range: Range<usize>) -> LazyReadHold {
        let start = self.range.start + range.start;
        self.read.hold(start..self.range.start + range.end)
    }
}

impl Drop for LazyReadHold {
    fn drop(&mut self) {
        self.read.let_go(&self.range);
    }
}

impl core::fmt::Debug for LazyReadHold {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyReadHold")
            .field("range", &self.range)
            .finish()
    }
}

impl Drop for LazyRead {
    fn drop(&mut self) {
        if let LazyReadState::Kept | LazyReadState::Started { .. } = *self.state.lock() {
            self.source.handle.memory.descriptor().lazy_read_let_go();
        }
    }
}

impl core::fmt::Debug for LazyRead {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyRead")
            .field("source", &self.source.handle)
            .finish()
    }
}
