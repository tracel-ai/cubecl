use tracel_xtask::prelude::*;

#[macros::extend_command_args(CheckCmdArgs, Target, CheckSubCommand)]
pub struct CubeCLCheckCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(mut args: CubeCLCheckCmdArgs) -> anyhow::Result<()> {
    if args.ci {
        // Exclude crates that are not supported on CI
        args.exclude
            .extend(vec!["cubecl-cuda".to_string(), "cubecl-hip".to_string()]);
    }
    base_commands::check::handle_command(args.try_into().unwrap())?;
    // Specific additional commands to test specific features
    // cubecl-wgpu with SPIR-V
    // cubecl-wgpu with exclusive-memory-only
    // cubecl-runtime without default features
    // Disabled on MacOS see:
    #[cfg(not(target_os = "macos"))]
    helpers::custom_crates_check(
        vec!["cubecl-wgpu"],
        vec!["--features", "spirv"],
        None,
        None,
        "std with SPIR-V compiler",
    )?;
    helpers::custom_crates_check(
        vec!["cubecl-wgpu"],
        vec!["--features", "exclusive-memory-only"],
        None,
        None,
        "std with exclusive_memory_only",
    )?;
    helpers::custom_crates_check(
        vec!["cubecl-runtime"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}
