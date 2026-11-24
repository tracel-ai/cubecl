use crate::server::{CopyDescriptor, ServerError};
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
pub struct LazyRead {
    source: CopyDescriptor,
    state: Mutex<LazyReadState>,
    /// What the started read returned, resolved once for every holder.
    started: Once<Result<Bytes, ServerError>>,
}

enum LazyReadState {
    /// No server keeps it, so the read on first access copies the memory as it is then.
    Unkept,
    /// Its server keeps it, and no work has written the memory since.
    Kept,
    /// Its server refused to keep it, for this error.
    Refused(ServerError),
    /// Work was about to write the memory, so its server started the read first. `None` once
    /// taken to be resolved into [`LazyRead::started`].
    Started(Option<DynFut<Result<Vec<Bytes>, ServerError>>>),
}

impl LazyRead {
    /// A read of `source` no server keeps yet.
    pub fn new(source: CopyDescriptor) -> Self {
        Self {
            source,
            state: Mutex::new(LazyReadState::Unkept),
            started: Once::new(),
        }
    }

    /// What it reads.
    pub fn source(&self) -> &CopyDescriptor {
        &self.source
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

    /// `read` is the server's read of [`source`](Self::source), queued ahead of any work that
    /// writes the memory after it, which the first access takes its bytes from.
    pub fn start(&self, read: DynFut<Result<Vec<Bytes>, ServerError>>) {
        let mut state = self.state.lock();
        if let LazyReadState::Kept = *state {
            self.source.handle.memory.descriptor().lazy_read_let_go();
        }
        *state = LazyReadState::Started(Some(read));
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
            LazyReadState::Started(_) => Ok(true),
            LazyReadState::Unkept | LazyReadState::Kept => Ok(false),
        }
    }

    /// The bytes in `range` of what the started read returned, copied out so every holder takes
    /// its own.
    ///
    /// # Assumptions
    ///
    /// - [`is_started`](Self::is_started) answered `true`.
    ///
    /// # Errors
    ///
    /// The started read's.
    pub(crate) fn started_bytes(&self, range: Range<usize>) -> Result<Bytes, ServerError> {
        let started = self.started.call_once(|| {
            let read = match &mut *self.state.lock() {
                LazyReadState::Started(read) => read.take(),
                _ => None,
            }
            .expect("a started read is resolved once");
            cubecl_environment::future::reader::read_sync(read).map(|mut bytes| bytes.remove(0))
        });
        match started {
            Ok(bytes) => Ok(Bytes::from_bytes_vec(bytes[range].to_vec())),
            Err(error) => Err(error.clone()),
        }
    }
}

impl Drop for LazyRead {
    fn drop(&mut self) {
        if let LazyReadState::Kept = *self.state.lock() {
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
