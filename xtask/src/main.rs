mod commands;
// mod dependencies;
mod logging;
// mod runchecks;
mod utils;
// mod vulnerabilities;

use crate::{logging::init_logger, utils::time::format_duration};
use clap::{Parser, Subcommand};
use std::time::Instant;

#[macro_use]
extern crate log;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct XtaskArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Runs checks and fix issues (used for development purposes)
    Check(commands::check::CheckCmdArgs),
    /// Runs checks for Continous Integration
    CI(commands::ci::CICmdArgs),
    /// Runs tests.
    Test(commands::test::TestCmdArgs),
    /// Runs all tests and checks that should pass before opening a Pull Request.
    PullRequestChecks,
}

fn main() -> anyhow::Result<()> {
    init_logger().init();
    let args = XtaskArgs::parse();

    let start = Instant::now();
    match args.command {
        Command::Check(args) => commands::check::handle_command(args, None),
        Command::CI(args) => commands::ci::handle_command(args),
        Command::Test(args) => commands::test::handle_command(args),
        Command::PullRequestChecks => commands::pull_request_checks::handle_command(),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
