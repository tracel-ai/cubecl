// Helpers shared between `build.rs` and its unit tests.
//
// Included verbatim by `build.rs`, and by `src/lib.rs` under `cfg(test)` so
// the pure parts can be tested with `cargo test`.

use std::ffi::OsString;
use std::path::Path;

/// Builds the compiler arguments for the AMDGPU C++ shims out of the raw
/// `llvm-config --cxxflags` string and the LLVM prefix it was queried with.
///
/// The raw string cannot simply be split on whitespace: it starts with an `-I`
/// pointing inside the bundled LLVM prefix, and on macOS that prefix always
/// contains a space (`~/Library/Application Support/tracel/...`), so splitting
/// would hand clang++ two broken arguments. The include directory is known from
/// the prefix, so it is passed through as a single argument and its `-I` entry
/// is removed from the string before the remaining, space-free flags are split.
fn shim_flags(cxxflags: &str, prefix: &Path) -> Vec<OsString> {
    let include_dir = prefix.join("include");
    let mut args = vec![
        OsString::from("-isystem"),
        include_dir.clone().into_os_string(),
    ];

    let rest = match include_dir.to_str() {
        Some(dir) => cxxflags.replace(&format!("-I{dir}"), " "),
        None => cxxflags.to_string(),
    };

    for flag in rest.split_whitespace() {
        match flag.strip_prefix("-I") {
            // Some other include directory; llvm-config doesn't currently emit any.
            Some(dir) => {
                args.push(OsString::from("-isystem"));
                args.push(OsString::from(dir));
            }
            None => args.push(OsString::from(flag)),
        }
    }

    args
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// What `llvm-config --cxxflags` returns for the bundled LLVM.
    fn cxxflags(prefix: &Path) -> String {
        format!(
            "-I{}/include -std=c++17  -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
             -D__STDC_LIMIT_MACROS -fno-exceptions",
            prefix.display()
        )
    }

    #[test]
    fn include_dir_with_a_space_stays_one_argument() {
        // The macOS prefix, which is always under `Library/Application Support`.
        let prefix = PathBuf::from("/Users/someone/Library/Application Support/tracel/llvm-22");
        let args = shim_flags(&cxxflags(&prefix), &prefix);

        let isystem = args.iter().position(|a| a == "-isystem").unwrap();
        assert_eq!(args[isystem + 1], prefix.join("include").into_os_string());
        assert!(
            !args.iter().any(|a| a == "Support/tracel/llvm-22/include"),
            "the include path was split: {args:?}"
        );
    }

    #[test]
    fn other_flags_are_kept_and_include_flag_is_dropped() {
        let prefix = PathBuf::from("/Users/someone/Library/Application Support/tracel/llvm-22");
        let args = shim_flags(&cxxflags(&prefix), &prefix);

        assert!(!args.iter().any(|a| a.to_string_lossy().starts_with("-I")));
        assert_eq!(
            args[2..],
            [
                "-std=c++17",
                "-D__STDC_CONSTANT_MACROS",
                "-D__STDC_FORMAT_MACROS",
                "-D__STDC_LIMIT_MACROS",
                "-fno-exceptions",
            ]
            .map(OsString::from)
        );
    }

    #[test]
    fn space_free_prefix_works_the_same() {
        let prefix = PathBuf::from("/home/someone/.local/share/tracel/llvm-22");
        let args = shim_flags(&cxxflags(&prefix), &prefix);

        assert_eq!(args[0], OsString::from("-isystem"));
        assert_eq!(args[1], prefix.join("include").into_os_string());
        assert!(args.iter().any(|a| a == "-fno-exceptions"));
    }
}
