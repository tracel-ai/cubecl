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
    if args.ci {
        args.exclude.extend(vec![
            "cubecl-cuda".to_string(),
            "cubecl-hip".to_string(),
            "cubecl-metal".to_string(),
        ]);
    }
    base_commands::doc::handle_command(args.try_into().unwrap(), env, context)
}
