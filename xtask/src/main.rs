mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

#[macros::base_commands(
    Bump,
    Check,
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
    /// Build Burn in different modes.
    Build(commands::build::CubeCLBuildCmdArgs),
    /// Test Burn.
    Test(commands::test::CubeCLTestCmdArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>()?;
    match args.command {
        Command::Build(cmd_args) => commands::build::handle_command(cmd_args),
        Command::Test(cmd_args) => commands::test::handle_command(cmd_args),
        _ => dispatch_base_commands(args),
    }?;
    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );
    Ok(())
}
