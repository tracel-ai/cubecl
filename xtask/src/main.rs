mod commands;

use tracel_xtask::prelude::*;

#[derive(clap::Subcommand, strum::Display)]
pub enum Command {
    Bump(BumpCmdArgs),
    Compile(CompileCmdArgs),
    Coverage(CoverageCmdArgs),
    Dependencies(DependenciesCmdArgs),
    Fix(FixCmdArgs),
    Publish(PublishCmdArgs),
    Validate(ValidateCmdArgs),
    Vulnerabilities(VulnerabilitiesCmdArgs),
    /// Build cubecl in different modes.
    Build(commands::build::CubeCLBuildCmdArgs),
    /// Build cubecl in different modes.
    Check(commands::check::CubeCLCheckCmdArgs),
    /// Build documentation.
    Doc(commands::doc::CubeCLDocCmdArgs),
    /// Test cubecl.
    Test(commands::test::CubeCLTestCmdArgs),
    /// Run commands to manage the book.
    Book(commands::book::BookArgs),
    /// Profile kernels.
    Profile(commands::profile::ProfileArgs),
}

fn dispatch_base_commands(args: XtaskArgs<Command>, env: Environment) -> anyhow::Result<()> {
    match args.command {
        Command::Bump(cmd) => base_commands::bump::handle_command(cmd, env, args.context),
        Command::Compile(cmd) => base_commands::compile::handle_command(cmd, env, args.context),
        Command::Coverage(cmd) => base_commands::coverage::handle_command(cmd, env, args.context),
        Command::Dependencies(cmd) => {
            base_commands::dependencies::handle_command(cmd, env, args.context)
        }
        Command::Fix(cmd) => base_commands::fix::handle_command(cmd, env, args.context, None),
        Command::Publish(cmd) => base_commands::publish::handle_command(cmd, env, args.context),
        Command::Vulnerabilities(cmd) => {
            base_commands::vulnerabilities::handle_command(cmd, env, args.context)
        }
        _ => Err(anyhow::anyhow!("Unknown command")),
    }
}

fn main() -> anyhow::Result<()> {
    let (args, environment) = init_xtask::<Command>(parse_args::<Command>()?)?;
    match args.command {
        Command::Build(cmd_args) => {
            commands::build::handle_command(cmd_args, environment, args.context)
        }
        Command::Check(cmd_args) => {
            commands::check::handle_command(cmd_args, environment, args.context)
        }
        Command::Doc(cmd_args) => {
            commands::doc::handle_command(cmd_args, environment, args.context)
        }
        Command::Test(cmd_args) => {
            commands::test::handle_command(cmd_args, environment, args.context)
        }
        Command::Book(cmd_args) => cmd_args.parse(),
        Command::Profile(cmd_args) => cmd_args.run(),
        Command::Validate(cmd_args) => {
            commands::validate::handle_command(&cmd_args, environment, args.context)
        }
        _ => dispatch_base_commands(args, environment),
    }?;
    Ok(())
}
