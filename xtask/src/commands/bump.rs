use std::process::Command;

use anyhow::{anyhow, Ok};
use clap::{Args, Subcommand};
use strum::{Display, EnumIter, EnumString};

use crate::{endgroup, group, utils::cargo::ensure_cargo_crate_is_installed};

#[derive(Args)]
pub(crate) struct BumpCmdArgs {
    #[command(subcommand)]
    command: BumpCommand,
}

#[derive(EnumString, EnumIter, Display, Clone, PartialEq, Subcommand)]
#[strum(serialize_all = "lowercase")]
enum BumpCommand {
    /// Run unit tests.
    Major,
    /// Run integration tests.
    Minor,
    /// Run documentation tests.
    Patch,
}

pub(crate) fn handle_command(args: BumpCmdArgs) -> anyhow::Result<()> {
    bump(&args.command)
}

fn bump(command: &BumpCommand) -> anyhow::Result<()> {
    group!("Bump version: {command}");
    ensure_cargo_crate_is_installed("cargo-edit", None, false)?;
    let status = Command::new("cargo")
        .args(["set-version", "--bump", &command.to_string()])
        .status()
        .map_err(|e| anyhow!("Failed to execute cargo set-version: {}", e))?;
    if !status.success() {
        return Err(anyhow!("Cannot set new {command} version"));
    }
    endgroup!();
    Ok(())
}
