mod commands;

#[macro_use]
extern crate log;

use tracel_xtask::prelude::*;

#[macros::base_commands(
    Bump,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Build cubecl in different modes.
    Build(commands::build::CubeCLBuildCmdArgs),
    /// Build cubecl in different modes.
    Check(commands::check::CubeCLCheckCmdArgs),
    /// Test cubecl.
    Test(commands::test::CubeCLTestCmdArgs),
    /// Run commands to manage the book.
    Book(commands::book::BookArgs),
    /// Profile kernels.
    Profile(commands::profile::ProfileArgs),
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
