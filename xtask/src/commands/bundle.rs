use std::path::PathBuf;

use tracel_xtask::prelude::*;

use cubecl_environment::bundle::{Bundle, ExportOptions};
use cubecl_environment::persistence::CacheConfig;

#[derive(clap::Args)]
pub struct BundleArgs {
    #[command(subcommand)]
    command: BundleSubCommand,
}

#[derive(clap::Subcommand, strum::Display)]
pub(crate) enum BundleSubCommand {
    /// Export warm cache roots into an environment bundle file.
    ///
    /// Typical workflow: run your application once so autotune and the
    /// compilation caches are warm, then export.
    Export(ExportArgs),
    /// Print what a bundle file contains.
    Inspect(InspectArgs),
}

#[derive(clap::Args)]
pub(crate) struct ExportArgs {
    /// Human-chosen bundle name, e.g. "H100 Linux".
    #[arg(long)]
    pub name: String,
    /// Output bundle file. An existing bundle is replaced.
    #[arg(long)]
    pub out: PathBuf,
    /// Cache roots to snapshot, either a cache root directory or a cache
    /// database file. Defaults to the standard cache root (the project target
    /// directory).
    #[arg(long)]
    pub root: Vec<PathBuf>,
    /// Only export these stores, e.g. `autotune` or `cuda`. Repeatable.
    /// Defaults to every store in the cache.
    #[arg(long = "store")]
    pub stores: Vec<String>,
}

#[derive(clap::Args)]
pub(crate) struct InspectArgs {
    /// The bundle file to describe.
    pub path: PathBuf,
}

impl BundleArgs {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        match &self.command {
            BundleSubCommand::Export(args) => export(args),
            BundleSubCommand::Inspect(args) => inspect(args),
        }
    }
}

fn export(args: &ExportArgs) -> anyhow::Result<()> {
    let roots = if args.root.is_empty() {
        vec![CacheConfig::default().root()]
    } else {
        args.root.clone()
    };

    let options = ExportOptions {
        name: args.name.clone(),
        stores: args.stores.clone(),
        ..Default::default()
    };
    let manifest = cubecl_environment::bundle::export(&roots, &args.out, &options)?;

    info!(
        "Exported bundle '{}' (cubecl {}) to {:?}",
        manifest.name, manifest.cubecl_version, args.out
    );
    describe(&Bundle::open(&args.out)?);

    Ok(())
}

fn inspect(args: &InspectArgs) -> anyhow::Result<()> {
    describe(&Bundle::open(&args.path)?);
    Ok(())
}

fn describe(bundle: &Bundle) {
    let manifest = bundle.manifest();

    info!("Bundle '{}'", manifest.name);
    info!("  cubecl version: {}", manifest.cubecl_version);
    for environment in &manifest.environments {
        info!(
            "  environment: {} {}/{} {}",
            environment.label,
            environment.os,
            environment.arch,
            environment.devices.join(", ")
        );
    }

    let summary = bundle.database().summary();
    if summary.is_empty() {
        info!("  (no entries)");
        return;
    }

    let total: u64 = summary.iter().map(|store| store.entries).sum();
    info!("  {total} entries:");
    for store in summary {
        info!(
            "    {} — {} entries, {:.1} KiB",
            store.store,
            store.entries,
            store.bytes as f64 / 1024.0
        );
    }
}
