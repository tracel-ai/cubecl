use tracel_xtask::prelude::*;

#[macros::extend_command_args(DocCmdArgs, Target, DocSubCommand)]
pub struct CubeCLDocCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLDocCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    // `cubecl-metal` depends on `objc2`, which only compiles on Apple platforms, so
    // exclude it from doc builds on non-Apple CI.
    if args.ci {
        args.exclude.push("cubecl-metal".to_string());
    }
    base_commands::doc::handle_command(args.try_into().unwrap(), env, context)
}
