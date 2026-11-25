use crate::memory_management::ManagedMemoryDescriptor;
use crate::server::{BufferBinding, CopyDescriptor, Handle, ServerError};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_common::bytes::Bytes;
use cubecl_environment::future::DynFut;
use cubecl_environment::sync::{Mutex, Once};
use cubecl_zspace::striding::try_check_contiguous_row_major_strides;

/// What takes a started read's copy when nothing will resolve it.
pub type RetireCopy = Box<dyn FnOnce(DynFut<Result<Vec<Bytes>, ServerError>>) + Send>;

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
    /// beside every one of them leaves it be and a start copies only them.
    held: Mutex<Vec<Range<usize>>>,
    /// What the started read returned, each piece with its offset in the read, resolved once
    /// for every holder.
    started: Once<Result<Vec<(usize, Bytes)>, ServerError>>,
}

enum LazyReadState {
    /// No server keeps it, so the read on first access copies the memory as it is then.
    Unkept,
    /// Its server keeps it, and no work has written the memory since.
    Kept,
    /// Its server refused to keep it, for this error.
    Refused(ServerError),
    /// Work was about to write the memory, so its server started the read of the pieces at
    /// `offsets` first. `read` is `None` once taken to be resolved into [`LazyRead::started`].
    Started {
        offsets: Vec<usize>,
        read: Option<DynFut<Result<Vec<Bytes>, ServerError>>>,
        retire: Option<RetireCopy>,
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

    /// Whether `written` overlaps any of the bytes it still holds.
    pub fn is_overwritten_by(&self, written: &BufferBinding) -> bool {
        let written = written.range();
        self.held_pieces()
            .into_iter()
            .any(|held| held.start < written.end && written.start < held.end)
    }

    /// The bytes it still holds, as the pieces of memory the server reads them from and their
    /// offsets in it, none once nothing holds any.
    ///
    /// # Assumptions
    ///
    /// - Bytes narrower than the whole read are held only through a view, which only a
    ///   contiguous source gives.
    pub fn to_held_sources(&self) -> (Vec<CopyDescriptor>, Vec<usize>) {
        let held = self.held_ranges();
        if held.is_empty() {
            return (Vec::new(), Vec::new());
        }
        if self.is_whole(&held) {
            return (vec![self.to_source()], vec![0]);
        }
        let whole = self.to_source().handle;
        let sources = held
            .iter()
            .map(|range| {
                let mut handle = whole.clone();
                let base = handle.offset_start.unwrap_or(0);
                handle.offset_end = Some(handle.size - (base + range.end as u64));
                handle.offset_start = Some(base + range.start as u64);
                CopyDescriptor::new(handle, [range.len()].into(), [1].into(), 1)
            })
            .collect();
        (sources, held.into_iter().map(|range| range.start).collect())
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

    /// `read` is the server's read of the pieces at `offsets` that
    /// [`to_held_sources`](Self::to_held_sources) named, queued ahead of any work that writes
    /// the memory after it, which the first access takes its bytes from.
    ///
    /// # Assumptions
    ///
    /// - Its server keeps it.
    ///
    /// `retire` takes `read` if the read drops before an access resolves it, so its copy is
    /// let go of where waiting for it blocks nothing.
    pub fn start(
        &self,
        offsets: Vec<usize>,
        read: DynFut<Result<Vec<Bytes>, ServerError>>,
        retire: RetireCopy,
    ) {
        *self.state.lock() = LazyReadState::Started {
            offsets,
            read: Some(read),
            retire: Some(retire),
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

    /// The bytes in `range` of what the started read returned: the result itself, shared, for
    /// the whole of it, and an aligned copy for less.
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
            let (offsets, read) = match &mut *self.state.lock() {
                LazyReadState::Started { offsets, read, .. } => {
                    (core::mem::take(offsets), read.take())
                }
                _ => (Vec::new(), None),
            };
            let read = read.expect("a started read should be resolved once");
            let pieces = cubecl_environment::future::reader::read_sync(read).map(|pieces| {
                offsets
                    .into_iter()
                    .zip(pieces.into_iter().map(Bytes::shared))
                    .collect()
            });
            *self.state.lock() = LazyReadState::Resolved;
            self.source.handle.memory.descriptor().lazy_read_let_go();
            pieces
        });
        let pieces = started.as_ref().map_err(Clone::clone)?;
        let (offset, bytes) = pieces
            .iter()
            .find(|(offset, bytes)| *offset <= range.start && range.end <= offset + bytes.len())
            .expect("a held range should lie in one started piece");
        let (start, end) = (range.start - offset, range.end - offset);
        // A window that starts inside a piece loses the alignment typed reads rely on.
        let whole = start == 0 && end == bytes.len();
        Ok(match whole.then(|| bytes.view(start, end).ok()).flatten() {
            Some(window) => window,
            None => Bytes::from_bytes_vec(bytes[start..end].to_vec()),
        })
    }

    /// The descriptor of the memory it reads.
    pub fn descriptor(&self) -> &ManagedMemoryDescriptor {
        self.source.handle.memory.descriptor()
    }

    /// How many bytes the whole read returns.
    fn len(&self) -> usize {
        self.source.shape.iter().product::<usize>() * self.source.elem_size
    }

    /// The ranges of the bytes still held, sorted, with overlapping and touching ones merged.
    fn held_ranges(&self) -> Vec<Range<usize>> {
        let mut held = self.held.lock().clone();
        held.sort_by_key(|range| range.start);
        let mut merged: Vec<Range<usize>> = Vec::with_capacity(held.len());
        for range in held {
            match merged.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        merged
    }

    fn is_whole(&self, held: &[Range<usize>]) -> bool {
        matches!(held, [range] if *range == (0..self.len()))
    }

    /// The memory the bytes still held lie in, the whole binding's for a strided read held
    /// whole.
    fn held_pieces(&self) -> Vec<Range<u64>> {
        let read = self.source.handle.to_binding().range();
        let held = self.held_ranges();
        let contiguous =
            try_check_contiguous_row_major_strides(&self.source.shape, &self.source.strides)
                .is_ok();
        if self.is_whole(&held) && !contiguous {
            return vec![read];
        }
        held.into_iter()
            .map(|range| read.start + range.start as u64..read.start + range.end as u64)
            .collect()
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
        let mut state = self.state.lock();
        if let LazyReadState::Kept | LazyReadState::Started { .. } = *state {
            self.source.handle.memory.descriptor().lazy_read_let_go();
        }
        let unread = match &mut *state {
            LazyReadState::Started { read, retire, .. } => read.take().zip(retire.take()),
            _ => None,
        };
        core::mem::drop(state);
        if let Some((read, retire)) = unread {
            retire(read);
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
