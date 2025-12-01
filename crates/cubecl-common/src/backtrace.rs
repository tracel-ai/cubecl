use alloc::format;
use alloc::string::String;

/// Contains the backtrace information if available.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackTrace {
    inner: String,
}

impl BackTrace {
    /// Creates a new backtrace from the current thread.
    pub fn capture() -> Self {
        Self {
            #[cfg(feature = "std")]
            inner: {
                let backtrace = backtrace::Backtrace::new();
                let fmt = backtrace_std::BacktraceFmt { backtrace };
                format!("{fmt}")
            },
            #[cfg(not(feature = "std"))]
            inner: format!("No backtrace available"),
        }
    }
}

#[cfg(feature = "std")]
mod backtrace_std {
    use backtrace::BytesOrWideString;
    use core::fmt::Display;

    /// A modified version of [backtrace::BacktraceFmt] to skip the first capture.
    pub struct BacktraceFmt {
        pub backtrace: backtrace::Backtrace,
    }

    impl Display for BacktraceFmt {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let cwd = std::env::current_dir();

            let mut print_path =
                move |fmt: &mut core::fmt::Formatter<'_>, path: BytesOrWideString<'_>| {
                    let path = path.into_path_buf();
                    if let Ok(cwd) = &cwd {
                        if let Ok(suffix) = path.strip_prefix(cwd) {
                            return core::fmt::Display::fmt(&suffix.display(), fmt);
                        }
                    }

                    core::fmt::Display::fmt(&path.display(), fmt)
                };

            let mut fmt =
                backtrace::BacktraceFmt::new(f, backtrace::PrintFmt::Short, &mut print_path);
            fmt.add_context()?;
            for frame in self.backtrace.frames().iter().skip(1) {
                fmt.frame().backtrace_frame(frame)?;
            }
            fmt.finish()?;
            Ok(())
        }
    }
}

impl core::fmt::Debug for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.inner))
    }
}

impl core::fmt::Display for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.inner))
    }
}
