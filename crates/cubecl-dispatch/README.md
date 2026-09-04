# CubeCL Dispatch

This crate is where a binary picks the runtimes it links.

A kernel library depends on `cubecl` and never on a runtime.
A binary, a benchmark or a test suite depends on this crate with the runtime features it wants, and gets the runtime crates, the test runtime, and the one thing `cubecl` cannot do on its own: turn a `Device` into a `Client`.
