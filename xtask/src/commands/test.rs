use tracel_xtask::prelude::*;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeCLTestCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLTestCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    if args.ci {
        args.exclude.extend(vec![
            "cubecl-cuda".to_string(),
            "cubecl-hip".to_string(),
            "cubecl-metal".to_string(),
        ]);
    }
    base_commands::test::handle_command(args.try_into().unwrap(), env, context)?;
    // Additional feature-specific tests
    helpers::custom_crates_tests(
        vec!["cubecl-wgpu"],
        vec!["--features", "exclusive-memory-only", "--lib"],
        None,
        None,
        "std with exclusive_memory_only",
    )?;
    Ok(())
}
