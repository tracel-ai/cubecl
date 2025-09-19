use tracel_xtask::prelude::*;

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct CubeCLBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLBuildCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    if args.ci {
        // Exclude crates that are not supported on CI
        args.exclude
            .extend(vec!["cubecl-cuda".to_string(), "cubecl-hip".to_string()]);
    }
    base_commands::build::handle_command(args.try_into().unwrap(), env, context)?;
    // Specific additional commands to test specific features
    // burn-wgpu with SPIR-V
    helpers::custom_crates_build(
        vec!["cubecl-wgpu"],
        vec!["--features", "spirv"],
        None,
        None,
        "std with SPIR-V compiler",
    )?;
    helpers::custom_crates_build(
        vec!["cubecl-runtime"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}
