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
    /// What obtaining it took, from where the trip started to the artifact
    /// loaded on the device.
    pub duration: core::time::Duration,
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
    /// another key: the artifact was moved under this one and loaded, and the
    /// backend's compiler never ran.
    Rekeyed,
}

impl Record for CompilationRecord {
    const KIND: &'static str = "compilation";
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
        let Some(duration) = open.span.elapsed() else {
            return;
        };
        let record = CompilationRecord {
            kernel: open.kernel.into(),
            key: open.key,
            ir: open.ir,
            outcome,
            duration,
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
