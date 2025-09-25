fn main() -> Result<(), Box<dyn std::error::Error>> {
    // required on macos
    tracel_llvm_bundler::config::set_homebrew_library_path()?;
    Ok(())
}
