//! Pointers to the code that asked for an allocation, kept with its memory handle
//! for as long as the allocation lives.

use super::AllocationOriginReport;
pub use crate::config::memory::MemoryTracing;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use cubecl_common::allocation_label::{self, AllocationLabel};

/// The process's [`MemoryTracing`], as [`configure_tracing`] last set it.
static TRACING: AtomicU8 = AtomicU8::new(MemoryTracing::Disabled as u8);
/// How many allocations were captured: each origin's index in that order.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Set what allocations carry from now on, process-wide: the runtime calls
/// it with its configuration's [`MemoryTracing`] once the configuration loads.
/// Allocations made before keep what they captured.
pub fn configure_tracing(tracing: MemoryTracing) {
    TRACING.store(tracing as u8, Ordering::Relaxed);
    allocation_label::set_recording(tracing != MemoryTracing::Disabled);
}

/// What allocations carry now.
pub fn memory_tracing() -> MemoryTracing {
    match TRACING.load(Ordering::Relaxed) {
        x if x == MemoryTracing::Labels as u8 => MemoryTracing::Labels,
        x if x == MemoryTracing::Backtraces as u8 => MemoryTracing::Backtraces,
        _ => MemoryTracing::Disabled,
    }
}

/// Pointers to the code that asked for an allocation: the labels the allocating
/// thread had entered, the thread, the allocation's index in the order origins
/// were captured, and under [`MemoryTracing::Backtraces`] the call stack.
///
/// Captured when a memory handle is created — on the thread that asked for the
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

    /// The crates whose frames sit above the caller: the standard library
    /// taking the backtrace, and the runtime minting the handle for whoever
    /// asked for memory.
    const RUNTIME_CRATES: [&str; 5] = ["core", "std", "alloc", "cubecl-runtime", "cubecl-server"];

    /// Where the program's own frames end: the process and thread entry
    /// points, as their file or, without one, their function.
    const ENTRY_POINTS: [&str; 5] = [
        "std/src/rt.rs",
        "std/src/sys/backtrace.rs",
        "std/src/thread/",
        "std::rt::lang_start",
        "std::sys::backtrace::__rust_begin_short_backtrace",
    ];

    /// Where a frame of the stack stands relative to the code that asked for
    /// the memory.
    #[derive(Debug, PartialEq, Eq)]
    enum FramePlace {
        /// The standard library or the runtime, above the caller.
        Runtime,
        /// The program, from the caller down.
        Program,
        /// A process or thread entry point: nothing below it is the program's.
        Entry,
    }

    impl FramePlace {
        /// Where the frame running `function`, at `location` when the
        /// backtrace knows it, stands. A frame is placed by its file when it
        /// has one: a function inlined into another is often named without
        /// its module.
        fn of(function: &str, location: Option<&str>) -> Self {
            let place = location.unwrap_or(function);
            if ENTRY_POINTS.iter().any(|entry| place.starts_with(entry)) {
                return Self::Entry;
            }
            let in_runtime = match location {
                Some(location) => {
                    let krate = location.split("/src/").next().unwrap_or_default();
                    RUNTIME_CRATES.contains(&krate)
                }
                None => RUNTIME_CRATES.iter().any(|krate| {
                    let module = krate.replace('-', "_");
                    function.starts_with(&alloc::format!("{module}::"))
                        || function.starts_with(&alloc::format!("<{module}::"))
                }),
            };
            match in_runtime {
                true => Self::Runtime,
                false => Self::Program,
            }
        }
    }

    /// The frames of `backtrace` — the standard library's rendering, a
    /// numbered `function` line followed by its `at file:line:column` — from
    /// the first frame outside the standard library and the runtime down to
    /// the program's entry point, each as `function [file:line:column]` with
    /// the symbol hash and the path up to the crate dropped.
    pub(super) fn caller_frames(backtrace: &str) -> Vec<String> {
        let mut frames = Vec::new();
        let mut lines = backtrace.lines().peekable();
        while let Some(line) = lines.next() {
            let Some(function) = numbered_frame(line) else {
                continue;
            };
            let location = lines
                .next_if(|next| next.trim_start().starts_with("at "))
                .map(|next| short_path(next.trim_start().trim_start_matches("at ")));
            match FramePlace::of(function, location.as_deref()) {
                FramePlace::Runtime if frames.is_empty() => continue,
                FramePlace::Entry => break,
                FramePlace::Runtime | FramePlace::Program => {}
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
            // As a release build with line tables renders it: inlined frames
            // are named without their module.
            let backtrace = "\
   0: std::backtrace::Backtrace::force_capture
             at /rustc/abc/library/std/src/backtrace.rs:312:9
   1: call_once<fn() -> std::backtrace::Backtrace, ()>
             at /rustc/abc/library/core/src/ops/function.rs:250:5
   2: capture
             at /home/me/cubecl/crates/cubecl-runtime/src/memory_management/origin.rs:70:13
   3: apply
             at /home/me/cubecl/crates/cubecl-server/src/allocator.rs:65:27
   4: empty_qtensor<Shape>
             at /home/me/.cargo/git/checkouts/burn-1/abc/crates/burn-cubecl/src/ops/qtensor.rs:67:5
   5: call_once<F, ()>
             at /rustc/abc/library/core/src/ops/function.rs:250:5
   6: my_app::main::h0123456789abcdef
   7: std::rt::lang_start::{{closure}}
             at /rustc/abc/library/std/src/rt.rs:206:18
   8: main";
            assert_eq!(
                caller_frames(backtrace),
                [
                    "empty_qtensor<Shape> [burn-cubecl/src/ops/qtensor.rs:67:5]",
                    "call_once<F, ()> [core/src/ops/function.rs:250:5]",
                    "my_app::main",
                ]
            );
        }
    }
}
