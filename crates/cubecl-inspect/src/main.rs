//! `cubecl-inspect`: read a cubecl environment file back.

use clap::{Parser, Subcommand};
use cubecl_inspect::command::{self, Inspection, Output};
use cubecl_inspect::tui::{self, Browsing};
use cubecl_inspect::{EnvironmentDirectory, EnvironmentName, InspectError, Inspector};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "cubecl-inspect",
    version,
    about,
    arg_required_else_help = true,
    styles = command::styles(),
    before_help = command::wordmark()
)]
struct Cli {
    /// Which environment: `active` (the one the runtime runs in, the
    /// default), a file's path, or a name, read as `<root>/<name>.db` under
    /// the root `[cubecl.environment]` sets in the `cubecl.toml` or
    /// `burn.toml` above the working directory. `tui` without it starts on
    /// the list of environments.
    #[arg(long, global = true, value_name = "ENV")]
    env: Option<EnvironmentName>,

    #[command(flatten)]
    output: Output,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Every environment file in a directory, newest first.
    #[command(alias = "ls")]
    List {
        /// The directory holding the files: the runtime's root by default.
        directory: Option<PathBuf>,
        /// The extension that marks an environment file, without its dot:
        /// the runtime's by default.
        #[arg(long)]
        extension: Option<String>,
    },
    /// Browse the environments and their reports in the terminal, read
    /// again whenever the file changes: open one beside a running session to
    /// watch it fill the file.
    Tui,
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
        Command::List {
            directory,
            extension,
        } => {
            let runtime = EnvironmentDirectory::runtime();
            let directory = EnvironmentDirectory {
                path: directory.unwrap_or(runtime.path),
                extension: extension.unwrap_or(runtime.extension),
            };
            cli.output
                .print(&cubecl_inspect::list(&[directory]))
                .map_err(|err| err.to_string())
        }
        Command::Tui => tui::run(Browsing {
            directories: vec![EnvironmentDirectory::runtime()],
            open: cli.env.map(|name| name.resolve()),
        })
        .map_err(|err| err.to_string()),
        Command::Inspect(inspection) => {
            let name = cli.env.unwrap_or(EnvironmentName::Active);
            Inspector::open(name.resolve())
                .and_then(|inspector| inspection.run(inspector, cli.output))
                .map_err(|err: InspectError| err.to_string())
        }
    }
}
