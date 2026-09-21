//! `cubecl-inspect`: read a cubecl environment file back.

use clap::{Parser, Subcommand};
use cubecl_inspect::command::{Inspection, Output};
use cubecl_inspect::{InspectError, Inspector};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "cubecl-inspect", version, about)]
struct Cli {
    /// The environment file to read.
    #[arg(long, global = true, value_name = "PATH")]
    env: Option<PathBuf>,

    #[command(flatten)]
    output: Output,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Every environment file in a directory, newest build first.
    List {
        /// The directory holding the files.
        directory: PathBuf,
    },
    #[command(flatten)]
    Inspect(Inspection),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::List { directory } => cli
            .output
            .print(&Inspector::list(&directory).map_err(|err| err.to_string())?)
            .map_err(|err| err.to_string()),
        Command::Inspect(inspection) => {
            let path = cli
                .env
                .ok_or("name the environment file with --env <PATH>")?;
            Inspector::open(path)
                .and_then(|inspector| inspection.run(&inspector, cli.output))
                .map_err(|err: InspectError| err.to_string())
        }
    }
}
