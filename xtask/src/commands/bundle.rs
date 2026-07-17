use std::path::PathBuf;

use tracel_xtask::prelude::*;

use cubecl_environment::bundle::ExportOptions;
use cubecl_environment::persistence::CacheConfig;

#[derive(clap::Args)]
pub struct BundleArgs {
    #[command(subcommand)]
    command: BundleSubCommand,
}

#[derive(clap::Subcommand, strum::Display)]
pub(crate) enum BundleSubCommand {
    /// Export warm cache roots into an environment bundle directory.
    ///
    /// Typical workflow: run your application once so autotune and the
    /// compilation caches are warm, then export.
    Export(ExportArgs),
}

#[derive(clap::Args)]
pub(crate) struct ExportArgs {
    /// Human-chosen bundle name, e.g. "H100 Linux".
    #[arg(long)]
    pub name: String,
    /// Output directory for the bundle.
    #[arg(long)]
    pub out: PathBuf,
    /// Cache roots to snapshot. Defaults to the standard cache root
    /// (the project target directory).
    #[arg(long)]
    pub root: Vec<PathBuf>,
}

impl BundleArgs {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        match &self.command {
            BundleSubCommand::Export(args) => {
                let roots = if args.root.is_empty() {
                    vec![CacheConfig::default().root()]
                } else {
                    args.root.clone()
                };

                let options = ExportOptions {
                    name: args.name.clone(),
                    ..Default::default()
                };
                let manifest = cubecl_environment::bundle::export(&roots, &args.out, &options)?;

                info!(
                    "Exported bundle '{}' (cubecl {}) to {:?}",
                    manifest.name, manifest.cubecl_version, args.out
                );
                Ok(())
            }
        }
    }
}
