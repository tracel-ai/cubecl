//! Who asked for an allocation, kept with its memory handle for as long as the
//! allocation lives.

use super::AllocationOriginReport;
pub use crate::config::memory::MemoryTracing;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use cubecl_common::allocation_label::{self, AllocationLabel};

/// The process's [`MemoryTracing`], as [`configure_tracing`] last set it.
static TRACING: AtomicU8 = AtomicU8::new(MemoryTracing::Disabled as u8);
/// How many origins were captured: each origin's place in that order.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Set what allocations remember from now on, process-wide: the runtime calls
/// it with its configuration's [`MemoryTracing`] once the configuration loads.
/// Allocations made before keep what they captured.
pub fn configure_tracing(tracing: MemoryTracing) {
    TRACING.store(tracing as u8, Ordering::Relaxed);
    allocation_label::set_recording(tracing != MemoryTracing::Disabled);
}

/// What allocations remember now.
pub fn memory_tracing() -> MemoryTracing {
    match TRACING.load(Ordering::Relaxed) {
        x if x == MemoryTracing::Labels as u8 => MemoryTracing::Labels,
        x if x == MemoryTracing::Backtraces as u8 => MemoryTracing::Backtraces,
        _ => MemoryTracing::Disabled,
    }
}

/// Who asked for an allocation: the labels the allocating thread had entered,
/// the thread, the allocation's place in the order origins were captured, and
/// under [`MemoryTracing::Backtraces`] the call stack.
///
/// Captured when a memory handle is minted — on the thread that asked for the
/// memory, before anything is queued to the device — and held by the handle's
/// descriptor, so it lives and dies with the allocation.
pub struct AllocationOrigin {
    labels: Vec<Arc<str>>,
    thread: Option<String>,
    sequence: u64,
    #[cfg(feature = "std")]
    backtrace: Option<std::backtrace::Backtrace>,
}

impl core::fmt::Debug for AllocationOrigin {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AllocationOrigin")
            .field("labels", &self.labels)
            .field("thread", &self.thread)
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

impl AllocationOrigin {
    /// The origin of an allocation asked for now, on this thread; `None` while
    /// [tracing](memory_tracing) is disabled.
    pub(crate) fn capture() -> Option<Arc<Self>> {
        let tracing = memory_tracing();
        if tracing == MemoryTracing::Disabled {
            return None;
        }
        Some(Arc::new(Self {
            labels: AllocationLabel::current(),
            #[cfg(feature = "std")]
            thread: std::thread::current().name().map(String::from),
            #[cfg(not(feature = "std"))]
            thread: None,
            sequence: SEQUENCE.fetch_add(1, Ordering::Relaxed),
            #[cfg(feature = "std")]
            backtrace: (tracing == MemoryTracing::Backtraces)
                .then(std::backtrace::Backtrace::force_capture),
        }))
    }

    /// The origin as a report states it, its call stack resolved.
    pub fn report(&self) -> AllocationOriginReport {
        AllocationOriginReport {
            labels: self
                .labels
                .iter()
                .map(|label| String::from(&**label))
                .collect(),
            thread: self.thread.clone(),
            sequence: self.sequence,
            #[cfg(feature = "std")]
            frames: self
                .backtrace
                .as_ref()
                .map(|backtrace| frames::caller_frames(&alloc::format!("{backtrace}")))
                .unwrap_or_default(),
            #[cfg(not(feature = "std"))]
            frames: Vec::new(),
        }
    }
}

/// Reading a resolved backtrace down to the frames of the code that asked for
/// the memory.
#[cfg(feature = "std")]
mod frames {
    use alloc::string::String;
    use alloc::vec::Vec;

    /// The most frames a report keeps of one call stack.
    const MAX_FRAMES: usize = 48;

    /// Frames the capture itself adds above the caller: the standard library
    /// taking the backtrace, and the runtime minting the handle.
    const CAPTURE_FRAMES: [&str; 4] = [
        "std::",
        "core::",
        "cubecl_runtime::memory_management::",
        "<cubecl_runtime::memory_management::",
    ];

    /// Frames below the program's own: the thread or process entry points.
    const ENTRY_FRAMES: [&str; 3] = [
        "std::rt::lang_start",
        "std::sys::backtrace::__rust_begin_short_backtrace",
        "std::thread::",
    ];

    /// The frames of `backtrace` — the standard library's rendering, a
    /// numbered `function` line followed by its `at file:line:column` — from
    /// the caller of the capture down to the program's entry point, each as
    /// `function [file:line:column]` with the symbol hash and the path up to
    /// the crate dropped.
    pub(super) fn caller_frames(backtrace: &str) -> Vec<String> {
        let mut frames = Vec::new();
        let mut lines = backtrace.lines().peekable();
        let mut above_caller = true;
        while let Some(line) = lines.next() {
            let Some(function) = numbered_frame(line) else {
                continue;
            };
            let location = lines
                .next_if(|next| next.trim_start().starts_with("at "))
                .map(|next| short_path(next.trim_start().trim_start_matches("at ")));
            if above_caller
                && CAPTURE_FRAMES
                    .iter()
                    .any(|prefix| function.starts_with(prefix))
            {
                continue;
            }
            above_caller = false;
            if ENTRY_FRAMES
                .iter()
                .any(|prefix| function.starts_with(prefix))
            {
                break;
            }
            let function = without_hash(function);
            frames.push(match location {
                Some(location) => alloc::format!("{function} [{location}]"),
                None => String::from(function),
            });
            if frames.len() == MAX_FRAMES {
                break;
            }
        }
        frames
    }

    /// The function a `   12: function` line names.
    fn numbered_frame(line: &str) -> Option<&str> {
        let (number, function) = line.trim_start().split_once(": ")?;
        number
            .chars()
            .all(|c| c.is_ascii_digit())
            .then_some(function)
    }

    /// `function` without the `::h0123456789abcdef` hash a symbol ends with.
    fn without_hash(function: &str) -> &str {
        match function.rsplit_once("::h") {
            Some((name, hash))
                if hash.len() == 16 && hash.chars().all(|c| c.is_ascii_hexdigit()) =>
            {
                name
            }
            _ => function,
        }
    }

    /// `path` from the directory of the crate it is in: `…/burn-core/src/x.rs:1:2`
    /// reads `burn-core/src/x.rs:1:2`.
    fn short_path(path: &str) -> String {
        match path.rfind("/src/") {
            Some(src) => {
                let crate_start = path[..src].rfind('/').map_or(0, |slash| slash + 1);
                String::from(&path[crate_start..])
            }
            None => String::from(path),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_frames_start_at_the_caller_and_stop_at_the_entry_point() {
            let backtrace = "\
   0: std::backtrace::Backtrace::force_capture
             at /rustc/abc/library/std/src/backtrace.rs:312:9
   1: cubecl_runtime::memory_management::origin::AllocationOrigin::capture::h0123456789abcdef
             at /home/me/cubecl/crates/cubecl-runtime/src/memory_management/origin.rs:70:13
   2: burn_core::store::load::h0123456789abcdef
             at /home/me/.cargo/git/checkouts/burn-1/abc/crates/burn-core/src/store/mod.rs:238:29
   3: my_app::main
   4: std::rt::lang_start::{{closure}}
             at /rustc/abc/library/std/src/rt.rs:206:18
   5: main";
            assert_eq!(
                caller_frames(backtrace),
                [
                    "burn_core::store::load [burn-core/src/store/mod.rs:238:29]",
                    "my_app::main",
                ]
            );
        }
    }
}
