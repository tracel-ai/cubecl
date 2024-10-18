use std::path::Path;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct BookArgs {
    #[command(subcommand)]
    command: BookSubCommand,
}

#[derive(clap::Subcommand, strum::Display)]
pub(crate) enum BookSubCommand {
    /// Build the book
    Build,
    /// Open the book on the specified port or random port and rebuild it automatically upon changes
    Open(OpenArgs),
}

#[derive(clap::Args)]
pub(crate) struct OpenArgs {
    /// Specify the port to open the book on (defaults to a random port if not specified)
    #[clap(long, default_value_t = random_port())]
    port: u16,
}

/// Book information
pub(crate) struct Book {
    name: &'static str,
    path: &'static Path,
}

impl BookArgs {
    pub(crate) fn parse(&self) -> anyhow::Result<()> {
        Book::run(&self.command)
    }
}

impl Book {
    const BOOK_NAME: &'static str = "CubeCL Book";
    const BOOK_PATH: &'static str = "./cubecl-book";

    pub(crate) fn run(args: &BookSubCommand) -> anyhow::Result<()> {
        let book = Self {
            name: Self::BOOK_NAME,
            path: Path::new(Self::BOOK_PATH),
        };
        book.execute(args)
    }

    fn execute(&self, command: &BookSubCommand) -> anyhow::Result<()> {
        ensure_cargo_crate_is_installed("mdbook", None, None, false)?;
        group!("{}: {}", self.name, command);
        match command {
            BookSubCommand::Build => self.build(),
            BookSubCommand::Open(args) => self.open(args),
        }?;
        endgroup!();
        Ok(())
    }

    fn build(&self) -> anyhow::Result<()> {
        run_process(
            "mdbook",
            &["build"],
            None,
            Some(self.path),
            "mdbook should build the book successfully",
        )
    }

    fn open(&self, args: &OpenArgs) -> anyhow::Result<()> {
        run_process(
            "mdbook",
            &["serve", "--open", "--port", &args.port.to_string()],
            None,
            Some(self.path),
            "mdbook should open the book successfully",
        )
    }
}
