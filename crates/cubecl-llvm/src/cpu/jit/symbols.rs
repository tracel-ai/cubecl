//! Symbol files for the profilers, in the open formats that `perf` and `samply` read.
//!
//! The files stay after the process stops, so a run-time switch must ask for them:
//! `CUBECL_JIT_SYMBOLS`, or `DOTNET_PerfMapEnabled`, the .NET variable for the same files.

use std::{
    fs::File,
    io::Write,
    sync::{Mutex, OnceLock},
};

/// The symbol files that the environment asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JitSymbols {
    /// `/tmp/perf-<pid>.map`: one line for each kernel.
    pub perf_map: bool,
    /// `jit-<pid>.dump`: lines and inlined frames, for `perf inject --jit`.
    pub jitdump: bool,
}

impl JitSymbols {
    /// The files that `CUBECL_JIT_SYMBOLS` asks for. Without it, the files that
    /// `DOTNET_PerfMapEnabled` asks for.
    pub(crate) fn from_env() -> Self {
        static FROM_ENV: OnceLock<JitSymbols> = OnceLock::new();
        *FROM_ENV.get_or_init(|| {
            Self::parse(
                std::env::var("CUBECL_JIT_SYMBOLS").ok().as_deref(),
                std::env::var("DOTNET_PerfMapEnabled").ok().as_deref(),
            )
        })
    }

    /// `cubecl` is `perf` (both files), `perfmap` or `jitdump`. Any other value asks for no
    /// file. `dotnet` has the .NET meanings: `1` both files, `2` the jitdump, `3` the perf map.
    fn parse(cubecl: Option<&str>, dotnet: Option<&str>) -> Self {
        let value = match (cubecl, dotnet) {
            (Some(value), _) => value,
            (None, Some("1")) => "perf",
            (None, Some("2")) => "jitdump",
            (None, Some("3")) => "perfmap",
            (None, _) => "",
        };
        Self {
            perf_map: matches!(value, "perf" | "perfmap"),
            jitdump: matches!(value, "perf" | "jitdump"),
        }
    }
}

/// The symbol sizes in the object code of one JIT.
#[derive(Default)]
pub(crate) struct SymbolSizes(Mutex<Vec<(String, u64)>>);

impl SymbolSizes {
    pub(crate) fn insert(&self, name: String, size: u64) {
        if let Ok(mut sizes) = self.0.lock() {
            sizes.push((name, size));
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<u64> {
        let sizes = self.0.lock().ok()?;
        sizes.iter().find(|(n, _)| n == name).map(|(_, size)| *size)
    }
}

/// Appends `<addr> <size> <name>` to `/tmp/perf-<pid>.map`, the perf map format.
pub(crate) fn write_perf_map(addr: u64, size: u64, name: &str) {
    static PERF_MAP: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    let file = PERF_MAP.get_or_init(|| {
        let path = format!("/tmp/perf-{}.map", std::process::id());
        match File::options().create(true).append(true).open(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(err) => {
                log::warn!("Can't open the perf map {path}: {err}");
                None
            }
        }
    });
    let Some(Ok(mut file)) = file.as_ref().map(Mutex::lock) else {
        return;
    };
    // One write for each line, so a reader never sees half a line.
    let line = format!("{addr:x} {size:x} {name}\n");
    if let Err(err) = file.write_all(line.as_bytes()) {
        log::warn!("Can't write the perf map: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CUBECL_JIT_SYMBOLS` decides when it is set. Else `DOTNET_PerfMapEnabled` has its .NET
    /// meanings: `samply record --coreclr` sets `2` on Linux.
    #[test]
    fn the_variables_select_the_files() {
        let cases = [
            (Some("perf"), Some("0"), true, true),
            (Some("perfmap"), None, true, false),
            (Some("jitdump"), None, false, true),
            (Some("none"), Some("1"), false, false),
            (None, Some("1"), true, true),
            (None, Some("2"), false, true),
            (None, Some("3"), true, false),
            (None, Some("0"), false, false),
            (None, None, false, false),
        ];
        for (cubecl, dotnet, perf_map, jitdump) in cases {
            assert_eq!(
                JitSymbols::parse(cubecl, dotnet),
                JitSymbols { perf_map, jitdump },
                "{cubecl:?} {dotnet:?}"
            );
        }
    }
}
