mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
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
    let start = Instant::now();
    let args = init_xtask::<Command>()?;
    match args.command {
        Command::Build(cmd_args) => commands::build::handle_command(cmd_args),
        Command::Check(cmd_args) => commands::check::handle_command(cmd_args),
        Command::Test(cmd_args) => commands::test::handle_command(cmd_args),
        Command::Book(cmd_args) => cmd_args.parse(),
        Command::Profile(cmd_args) => cmd_args.run(),
        Command::Validate(cmd_args) => commands::validate::handle_command(&cmd_args),
        _ => dispatch_base_commands(args),
    }?;
    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );
    Ok(())
}
