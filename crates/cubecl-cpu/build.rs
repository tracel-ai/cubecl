fn main() -> Result<(), Box<dyn std::error::Error>> {
    // required on macos
    tracel_llvm_bundler_rs::config::set_homebrew_library_path()?;
    Ok(())
}
