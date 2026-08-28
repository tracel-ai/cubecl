fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Feature unification causes `pliron-llvm` to pull in the `llvm-sys` feature even though
    // it's not enabled here. So this is needed for CI.
    #[cfg(test)]
    tracel_llvm_bundler::llvm_sys::link()?;
    Ok(())
}
