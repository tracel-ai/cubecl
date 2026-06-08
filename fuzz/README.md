# cubecl fuzz targets

## `cube_macro`

Fuzzes the `#[cube]` macro by feeding it random source text and checking it never panics (a proc-macro panic aborts the user's build instead of emitting a diagnostic). The logic lives in `cubecl-macros-core`, and this target calls `cube_impl` on the parsed tokens.

Run (nightly), using the token dictionary:

```bash
cargo +nightly fuzz run cube_macro -- -dict=fuzz/cube_macro.dict
```

Reproduce a crash. The input is `&str`, so an artifact is exactly the offending source:

```bash
cat fuzz/artifacts/cube_macro/crash-<hash>            # read the source that broke the macro
cargo +nightly fuzz run cube_macro fuzz/artifacts/cube_macro/crash-<hash>   # replay it
cargo +nightly fuzz tmin cube_macro fuzz/artifacts/cube_macro/crash-<hash>  # minimize it
```
