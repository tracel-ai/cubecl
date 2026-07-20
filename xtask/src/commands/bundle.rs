use std::path::{Path, PathBuf};

use tracel_xtask::prelude::*;

use cubecl_environment::bundle::{
    Bundle, BundleFormat, BundleManifest, EmbeddedBundle, ExportOptions, SqliteBundle,
};
use cubecl_environment::bytes::Bytes;
use cubecl_environment::environment;
use cubecl_environment::persistence::{CacheConfig, NamespaceSummary};

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
    /// Print what a bundle file contains, in either format.
    Inspect(InspectArgs),
    /// Import a bundle into the local environment.
    ///
    /// This is the only thing a bundle is for: it fills the local store, and
    /// afterwards the file can be deleted. Nothing consults it at runtime.
    Import(ImportArgs),
    /// List the namespaces the local environment currently holds, which is
    /// what you consult before exporting.
    Namespaces(NamespacesArgs),
    /// List the environments under a cache root.
    Environments(NamespacesArgs),
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub(crate) enum Format {
    /// One SQLite file, for native targets.
    #[default]
    Sqlite,
    /// One flat blob, for wasm and no-std targets that have no file system.
    /// Embed it with `include_bytes!` or fetch it at runtime.
    Flat,
}

impl From<Format> for BundleFormat {
    fn from(format: Format) -> Self {
        match format {
            Format::Sqlite => BundleFormat::Sqlite,
            Format::Flat => BundleFormat::Flat,
        }
    }
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
    /// Only export these namespaces, e.g. `autotune` or `cuda`. Matches whole
    /// segments, so `autotune` selects everything under it. Repeatable.
    /// Defaults to every namespace in the cache.
    #[arg(long = "namespace")]
    pub namespaces: Vec<String>,
    /// The bundle layout to write.
    #[arg(long, value_enum, default_value_t = Format::Sqlite)]
    pub format: Format,
    /// Environment to export from. Defaults to the active one.
    #[arg(long)]
    pub environment: Option<String>,
}

#[derive(clap::Args)]
pub(crate) struct InspectArgs {
    /// The bundle file to describe.
    pub path: PathBuf,
}

#[derive(clap::Args)]
pub(crate) struct ImportArgs {
    /// The bundle file to import, in either format.
    pub path: PathBuf,
    /// Cache root to fill. Defaults to the standard cache root.
    #[arg(long)]
    pub root: Option<PathBuf>,
    /// Environment to import into. Defaults to the active one.
    #[arg(long)]
    pub environment: Option<String>,
}

#[derive(clap::Args)]
pub(crate) struct NamespacesArgs {
    /// Cache root to inspect. Defaults to the standard cache root.
    #[arg(long)]
    pub root: Option<PathBuf>,
    /// Environment to inspect. Defaults to the active one.
    #[arg(long)]
    pub environment: Option<String>,
}

impl BundleArgs {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        match &self.command {
            BundleSubCommand::Export(args) => export(args),
            BundleSubCommand::Inspect(args) => inspect(&args.path),
            BundleSubCommand::Import(args) => import(args),
            BundleSubCommand::Namespaces(args) => namespaces(args),
            BundleSubCommand::Environments(args) => environments(args),
        }
    }
}

fn cache_root(root: &Option<PathBuf>) -> PathBuf {
    root.clone()
        .unwrap_or_else(|| CacheConfig::default().root())
}

fn import(args: &ImportArgs) -> anyhow::Result<()> {
    if let Some(name) = &args.environment {
        environment::activate(name);
    }
    let root = cache_root(&args.root);

    let bundle = open_bundle(&args.path)?;
    let report = cubecl_environment::bundle::import(bundle.as_ref(), root.to_str());

    info!(
        "Imported {} entries into environment '{}' at {:?} ({} already present)",
        report.imported,
        environment::active(),
        root,
        report.skipped,
    );
    for namespace in &report.namespaces {
        info!("    {namespace}");
    }

    Ok(())
}

fn namespaces(args: &NamespacesArgs) -> anyhow::Result<()> {
    if let Some(name) = &args.environment {
        environment::activate(name);
    }
    let root = cache_root(&args.root);
    let summary = environment::namespaces(&root);

    info!("Environment '{}' at {:?}", environment::active(), root);
    if summary.is_empty() {
        info!("  (no namespaces)");
        return Ok(());
    }
    report(&summary);

    Ok(())
}

fn environments(args: &NamespacesArgs) -> anyhow::Result<()> {
    let root = cache_root(&args.root);
    let names = environment::list(&root);

    info!("Environments at {root:?}");
    if names.is_empty() {
        info!("  (none)");
        return Ok(());
    }

    let active = environment::active();
    for name in names {
        let marker = if name == active { "*" } else { " " };
        info!("  {marker} {name}");
    }

    Ok(())
}

/// Opens a bundle file whichever format it is in.
fn open_bundle(path: &Path) -> anyhow::Result<Box<dyn Bundle>> {
    if let Ok(bundle) = SqliteBundle::open(path) {
        return Ok(Box::new(bundle));
    }

    let bytes = Bytes::from_bytes_vec(std::fs::read(path)?);
    let bundle = EmbeddedBundle::open(bytes).map_err(|err| anyhow::anyhow!("{path:?}: {err}"))?;

    Ok(Box::new(bundle))
}

fn export(args: &ExportArgs) -> anyhow::Result<()> {
    if let Some(name) = &args.environment {
        environment::activate(name);
    }

    let roots = if args.root.is_empty() {
        vec![CacheConfig::default().root()]
    } else {
        args.root.clone()
    };

    let options = ExportOptions {
        name: args.name.clone(),
        namespaces: args.namespaces.clone(),
        format: args.format.into(),
        ..Default::default()
    };
    let manifest = cubecl_environment::bundle::export(&roots, &args.out, &options)?;

    info!(
        "Exported bundle '{}' (cubecl {}) from environment '{}' to {:?}",
        manifest.name,
        manifest.cubecl_version,
        environment::active(),
        args.out
    );
    inspect(&args.out)
}

/// Describes a bundle file whichever format it is in.
fn inspect(path: &Path) -> anyhow::Result<()> {
    if let Ok(bundle) = SqliteBundle::open(path) {
        describe(bundle.manifest(), &bundle.database().summary());
        return Ok(());
    }

    let bundle = EmbeddedBundle::open(Bytes::from_bytes_vec(std::fs::read(path)?))
        .map_err(|err| anyhow::anyhow!("{path:?}: {err}"))?;
    // The metadata blob is the same manifest the SQLite format stores in a row.
    let manifest: BundleManifest = serde_json::from_slice(bundle.metadata())
        .map_err(|err| anyhow::anyhow!("{path:?}: unreadable manifest: {err}"))?;

    describe(&manifest, &bundle.summary());

    Ok(())
}

fn describe(manifest: &BundleManifest, summary: &[NamespaceSummary]) {
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

    if summary.is_empty() {
        info!("  (no entries)");
        return;
    }

    report(summary);
}

fn report(summary: &[NamespaceSummary]) {
    let total: u64 = summary.iter().map(|namespace| namespace.entries).sum();
    info!("  {total} entries:");
    for namespace in summary {
        info!(
            "    {} — {} entries, {:.1} KiB",
            namespace.namespace,
            namespace.entries,
            namespace.bytes as f64 / 1024.0
        );
    }
}
