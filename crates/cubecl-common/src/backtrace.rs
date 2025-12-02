#[cfg(feature = "std")]
type BacktraceState = backtrace_std::BacktraceState;
#[cfg(not(feature = "std"))]
type BacktraceState = alloc::string::String;

/// Contains the backtrace information if available.
///
/// # Notes
///
/// We chose BackTrace for the name since Backtrace is often confused with the nighly-only backtrace
/// feature by `thiserror`.
#[derive(Clone, Default)]
pub struct BackTrace {
    state: Option<BacktraceState>,
}

impl core::fmt::Debug for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl core::fmt::Display for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.state {
            Some(state) => f.write_fmt(format_args!("{state}")),
            None => f.write_str("No backtrace available"),
        }
    }
}

impl BackTrace {
    /// Creates a new backtrace from the current thread.
    ///
    /// # Notes
    ///
    /// It is quite cheap to create a backtrace, but quite expensive to display.
    pub fn capture() -> Self {
        Self {
            #[cfg(feature = "std")]
            state: {
                // We only resolve the backtrace when displaying the result.
                //
                // Making it cheaper to create.
                //
                // We capture the backtrace here to reduce the number of frames to ignore.
                let backtrace = backtrace::Backtrace::new_unresolved();
                Some(BacktraceState::new(backtrace))
            },
            #[cfg(not(feature = "std"))]
            state: None,
        }
    }
}

#[cfg(feature = "std")]
mod backtrace_std {
    use backtrace::BytesOrWideString;
    use core::fmt::Display;
    use std::sync::{Arc, Mutex};

    /// A modified version of [backtrace::BacktraceFmt] to skip the first capture.
    #[derive(Clone)]
    pub struct BacktraceState {
        backtrace: Arc<Mutex<backtrace::Backtrace>>,
    }

    impl BacktraceState {
        pub fn new(backtrace: backtrace::Backtrace) -> Self {
            Self {
                backtrace: Arc::new(Mutex::new(backtrace)),
            }
        }
    }

    impl Display for BacktraceState {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let mut backtrace = self.backtrace.lock().unwrap();
            backtrace.resolve();

            let cwd = std::env::current_dir();

            let mut print_path =
                move |fmt: &mut core::fmt::Formatter<'_>, path: BytesOrWideString<'_>| {
                    let path = path.into_path_buf();
                    if let Ok(cwd) = &cwd
                        && let Ok(suffix) = path.strip_prefix(cwd)
                    {
                        return core::fmt::Display::fmt(&suffix.display(), fmt);
                    }

                    core::fmt::Display::fmt(&path.display(), fmt)
                };

            let mut fmt =
                backtrace::BacktraceFmt::new(f, backtrace::PrintFmt::Short, &mut print_path);
            fmt.add_context()?;
            for frame in backtrace.frames().iter().skip(1) {
                fmt.frame().backtrace_frame(frame)?;
            }
            fmt.finish()?;
            Ok(())
        }
    }
}
