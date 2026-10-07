//! What the environment records of a kernel's compilation.

use crate::id::KernelId;
use cubecl_environment::records::{Record, RecordEffect, RecordLevel, Span};

use super::{KernelCacheKey, build_id_hash};

/// One kernel's trip through a backend's compilation path, as the environment
/// records it: compiled fresh, or loaded from the compilation store.
///
/// A hit in a server's in-memory cache is not a trip and is not recorded:
/// nothing here runs per launch. Neither is a trip that fails: the launch
/// error carries that account, and the environment holds nothing of it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompilationRecord {
    /// The kernel's type.
    pub kernel: alloc::string::String,
    /// The store entry naming the artifact: what tells two instances of one
    /// kernel type apart.
    pub key: KernelCacheKey,
    /// The kernel as cubecl defined it, before the backend's compiler — the
    /// IR's textual form, for a reader to render — at [`RecordLevel::Full`].
    /// Only a trip that misses the store defines the kernel, so a
    /// [`Loaded`](CompilationOutcome::Loaded) one carries none.
    pub ir: Option<alloc::string::String>,
    /// How the artifact was obtained.
    pub outcome: CompilationOutcome,
    /// The work obtaining it took: its own lowering, finalizing and loading,
    /// which is what it would have cost alone. A kernel compiled beside others
    /// counts none of the time it waited for them, so the work of a batch adds
    /// up to more than the time it took when it ran on several threads: that
    /// time is its [`CompilationBatchRecord::wall`].
    ///
    /// Recorded as `duration` before kernels compiled in batches, when every
    /// kernel compiled alone and its work was the time it took: those records
    /// still read, with the meaning they had.
    #[serde(alias = "duration")]
    pub work: core::time::Duration,
    /// The source the backend compiled, at [`RecordLevel::Full`].
    pub source: Option<alloc::string::String>,
}

/// How a backend obtained a kernel's artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompilationOutcome {
    /// Compiled from its definition: expanded, compiled by the backend's
    /// compiler, loaded on the device.
    Compiled,
    /// Read from the compilation store and loaded on the device.
    Loaded,
    /// Expanded to a source the store already held an artifact for, under
    /// another key: the artifact was kept under this one too — moved, when
    /// the other key is of an earlier build — and loaded, and the backend's
    /// compiler never ran.
    Rekeyed,
}

impl Record for CompilationRecord {
    const KIND: &'static str = "compilation";
}

/// One batch of kernels a server obtained together — compiled, or read from
/// the compilation store, and loaded — as the environment records it. A
/// kernel a launch compiles alone is a batch of one.
///
/// Each kernel the batch obtained has its [`CompilationRecord`], stamped when
/// the batch started, as this one is. What the batch cost the process is its
/// [`wall`](Self::wall): adding up its kernels' [`work`](CompilationRecord::work)
/// counts every thread's time as if they ran one after another.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompilationBatchRecord {
    /// How many kernels the batch was asked for, failures included.
    pub kernels: u32,
    /// How many threads compiled at once: one when at most one kernel had to
    /// be compiled, the rest read from the store.
    pub threads: u32,
    /// The time from the batch's start to its last kernel loaded.
    pub wall: core::time::Duration,
}

impl Record for CompilationBatchRecord {
    const KIND: &'static str = "compilation_batch";
}

/// A batch being recorded: opened where a server starts obtaining kernels,
/// closed when it has them. A no-op when the environment records nothing.
#[derive(Debug)]
pub(crate) struct CompilationBatchRecording {
    span: Option<Span>,
}

impl CompilationBatchRecording {
    pub(crate) fn new() -> Self {
        Self { span: Span::new() }
    }

    /// The batch obtained what it was asked for. `changed` is whether the
    /// store took any of its kernels, as their records say.
    pub(crate) fn close(self, kernels: usize, threads: usize, changed: bool) {
        let Some(span) = self.span else {
            return;
        };
        // A batch for a session that is gone is dropped, as its kernels are.
        let Some(wall) = span.elapsed() else {
            return;
        };
        let record = CompilationBatchRecord {
            kernels: kernels.try_into().unwrap_or(u32::MAX),
            threads: threads.try_into().unwrap_or(u32::MAX),
            wall,
        };
        span.close(effect(changed), &record);
    }
}

/// A compilation being recorded: a backend opens one where its compilation
/// path starts — past its in-memory cache — tells it what the trip goes
/// through, and closes it with how the artifact was obtained. Every call is a
/// no-op when the environment records nothing, and one dropped unclosed, by a
/// trip that failed, records nothing.
#[derive(Debug)]
pub struct CompilationRecording {
    open: Option<OpenRecording>,
}

/// What a [`CompilationRecording`] holds while the environment records.
#[derive(Debug)]
struct OpenRecording {
    span: Span,
    /// The work done so far, as [`CompilationRecording::worked`] reports it.
    work: core::time::Duration,
    kernel: &'static str,
    key: KernelCacheKey,
    ir: Option<alloc::string::String>,
    source: Option<alloc::string::String>,
}

impl CompilationRecording {
    /// Start recording `kernel_id`'s trip.
    pub fn new(kernel_id: &KernelId) -> Self {
        let open = Span::new().map(|span| OpenRecording {
            span,
            work: core::time::Duration::ZERO,
            kernel: kernel_id.type_name(),
            key: KernelCacheKey::new(kernel_id, build_id_hash()),
            ir: None,
            source: None,
        });
        Self { open }
    }

    /// The kernel was defined: keep its IR, at [`RecordLevel::Full`] only.
    /// The textual IR runs to hundreds of KB per kernel, where the compiled
    /// artifact is tens.
    pub fn defined(&mut self, definition: &crate::kernel::KernelDefinition) {
        if let Some(open) = self.open.as_mut().filter(|_| keeps_code()) {
            open.ir = Some(alloc::format!("{}", definition.body));
        }
    }

    /// The backend's compiler produced `source`: keep it, at
    /// [`RecordLevel::Full`] only.
    pub fn source(&mut self, source: &str) {
        if let Some(open) = self.open.as_mut().filter(|_| keeps_code()) {
            open.source = Some(source.into());
        }
    }

    /// The trip did `duration` more of its work: what the record's
    /// [`work`](CompilationRecord::work) adds up.
    pub fn worked(&mut self, duration: core::time::Duration) {
        if let Some(open) = self.open.as_mut() {
            open.work += duration;
        }
    }

    /// The artifact came from the compilation store: the environment did not
    /// change.
    pub fn loaded(self) {
        self.close(CompilationOutcome::Loaded, RecordEffect::Observed);
    }

    /// The artifact was compiled. `stored` is whether the store took it, as
    /// [`store_compiled`](super::store_compiled) answers: a compile the store did not take, or with
    /// no store to take it, changed nothing.
    pub fn compiled(self, stored: bool) {
        self.close(CompilationOutcome::Compiled, effect(stored));
    }

    /// The artifact was already stored under another key, and moved under
    /// this one. `stored` is whether the store took it there.
    pub fn rekeyed(self, stored: bool) {
        self.close(CompilationOutcome::Rekeyed, effect(stored));
    }

    fn close(self, outcome: CompilationOutcome, effect: RecordEffect) {
        let Some(open) = self.open else {
            return;
        };
        // The span still says whether the session the trip started in is
        // the one recording: a record for a session that is gone is dropped.
        if open.span.elapsed().is_none() {
            return;
        }
        let record = CompilationRecord {
            kernel: open.kernel.into(),
            key: open.key,
            ir: open.ir,
            outcome,
            work: open.work,
            source: open.source,
        };
        open.span.close(effect, &record);
    }
}

/// Whether a record keeps the kernel's code, its IR and its source: code is
/// the heaviest thing a record can carry.
fn keeps_code() -> bool {
    cubecl_environment::records::level() == RecordLevel::Full
}

/// An artifact the store took is the environment changing.
fn effect(stored: bool) -> RecordEffect {
    if stored {
        RecordEffect::Changed
    } else {
        RecordEffect::Observed
    }
}
