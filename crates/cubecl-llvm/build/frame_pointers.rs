// Included by `build.rs`, and by the tests in `src/lib.rs`.

/// Whether the rustc flags, separated by `0x1f`, turn on `force-frame-pointers`.
fn forces_frame_pointers(flags: &str) -> bool {
    let mut flags = flags.split('\x1f').peekable();
    let mut forced = false;
    while let Some(flag) = flags.next() {
        let codegen = match flag {
            "-C" => flags.next().unwrap_or_default(),
            flag => flag.strip_prefix("-C").unwrap_or(flag),
        };
        if let Some(value) = codegen.strip_prefix("force-frame-pointers") {
            // The last value wins, as in rustc.
            forced = matches!(value, "" | "=yes" | "=y" | "=on" | "=true" | "=always");
        }
    }
    forced
}
