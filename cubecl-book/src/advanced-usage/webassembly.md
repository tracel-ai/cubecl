# WebAssembly startup

CubeCL's compiler uses Pliron, which initializes its registries through `inventory` constructors.
With wasm-bindgen, the Wasm linker can run these constructors on every call from JavaScript,
including async callbacks. This repeated work caused the WebGPU slowdown reported in
[Burn #5992](https://github.com/tracel-ai/burn/issues/5992).

For `wasm32-unknown-unknown`, call `__wasm_call_ctors()` once at the beginning of your application's
`#[wasm_bindgen(start)]` function. This tells the linker that the application handles
initialization, so it stops adding constructor calls to every exported function. If you already have
a startup function, add the call there:

```rust,ignore
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn start() {
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }

    // SAFETY: called once during module startup, before constructor-dependent work.
    unsafe {
        __wasm_call_ctors();
    }

    // Set up the application or spawn async work here.
}
```

This runs all module constructors, including those from other libraries, so do not call it again.
Keep setup that uses CubeCL after this call in the same startup function, since
[separate startup hooks have no guaranteed order](https://wasm-bindgen.github.io/wasm-bindgen/reference/attributes/on-rust-exports/start.html).

See
[inventory's Wasm guidance](https://docs.rs/inventory/latest/inventory/#webassembly-and-constructors)
and the [upstream wasm-bindgen discussion](https://github.com/wasm-bindgen/wasm-bindgen/issues/1216)
for more background.
