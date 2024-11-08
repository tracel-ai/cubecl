use tracel_xtask::prelude::*;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeCLTestCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(mut args: CubeCLTestCmdArgs) -> anyhow::Result<()> {
    if args.ci {
        // Exclude crates that are not supported on CI
        args.exclude
            .extend(vec!["cubecl-cuda".to_string(), "cubecl-hip".to_string()]);
    }
    base_commands::test::handle_command(args.try_into().unwrap())?;
    // Specific additional commands to test specific features
    // cubecl-wgpu with exclusive-memory-only
    helpers::custom_crates_tests(
        vec!["cubecl-wgpu"],
        vec!["--features", "exclusive-memory-only"],
        None,
        None,
        "std with exclusive_memory_only",
    )?;
    Ok(())
}
