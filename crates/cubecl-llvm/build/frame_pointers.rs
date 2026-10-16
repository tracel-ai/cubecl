// Included by `build.rs`, and by the tests in `src/lib.rs`.

/// Whether the rustc flags, separated by `0x1f`, turn on `force-frame-pointers`.
fn forces_frame_pointers(flags: &str) -> bool {
    // A codegen option is `-C<option>`, or `-C` and then `<option>` as the next flag. The bare `-C`
    // matches no option, so each flag is read alone. The last value wins, as in rustc.
    flags
        .split('\x1f')
        .rev()
        .find_map(|flag| {
            flag.strip_prefix("-C")
                .unwrap_or(flag)
                .strip_prefix("force-frame-pointers")
        })
        .is_some_and(|value| matches!(value, "" | "=yes" | "=y" | "=on" | "=true" | "=always"))
}
