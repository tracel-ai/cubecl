use std::process::Command;

use anyhow::{anyhow, Ok, Result};
use clap::{Args, Subcommand};
use strum::{Display, EnumIter, EnumString, IntoEnumIterator};

use crate::{
    endgroup, group,
    utils::{
        cargo::ensure_cargo_crate_is_installed,
        workspace::{get_workspace_members, WorkspaceMemberType},
    },
};

use super::{
    test::{run_documentation, run_integration, run_unit},
    Target,
};

#[derive(Args)]
pub(crate) struct CICmdArgs {
    /// Target to check for.
    #[arg(short, long, value_enum)]
    pub target: Target,
    #[command(subcommand)]
    pub command: CICommand,
}

#[derive(EnumString, EnumIter, Display, Clone, PartialEq, Subcommand)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum CICommand {
    /// Run audit command.
    Audit,
    /// Run format command.
    Format,
    /// Run lint command.
    Lint,
    /// Run unit tests.
    UnitTests,
    /// Run integration tests.
    IntegrationTests,
    /// Run documentation tests.
    DocTests,
    /// Run all tests.
    AllTests,
    /// Run all the checks.
    All,
}

pub(crate) fn handle_command(args: CICmdArgs) -> anyhow::Result<()> {
    match args.command {
        CICommand::Audit => run_audit(&args.target),
        CICommand::Format => run_format(&args.target),
        CICommand::Lint => run_lint(&args.target),
        CICommand::UnitTests => run_unit_tests(&args.target),
        CICommand::IntegrationTests => run_integration_tests(&args.target),
        CICommand::DocTests => run_doc_tests(&args.target),
        CICommand::AllTests => run_all_tests(&args.target),
        CICommand::All => CICommand::iter()
            .filter(|c| *c != CICommand::All && *c != CICommand::AllTests)
            .try_for_each(|c| {
                handle_command(CICmdArgs {
                    command: c,
                    target: args.target.clone(),
                })
            }),
    }
}

fn run_audit(target: &Target) -> anyhow::Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            group!("Audit: Crates and Examples");
            ensure_cargo_crate_is_installed("cargo-audit", Some("fix"), false)?;
            info!("Command line: cargo audit");
            let status = Command::new("cargo")
                .args(["audit", "-q", "--color", "always"])
                .status()
                .map_err(|e| anyhow!("Failed to execute cargo audit: {}", e))?;
            if !status.success() {
                return Err(anyhow!("Audit check execution failed"));
            }
            endgroup!();
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All && *t != Target::Examples)
                .try_for_each(|t| run_audit(&t))?;
        }
    }
    Ok(())
}

fn run_format(target: &Target) -> Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            let members = match target {
                Target::Crates => get_workspace_members(WorkspaceMemberType::Crate),
                Target::Examples => get_workspace_members(WorkspaceMemberType::Example),
                _ => unreachable!(),
            };

            for member in members {
                group!("Format: {}", member.name);
                info!("Command line: cargo fmt --check -p {}", &member.name);
                let status = Command::new("cargo")
                    .args(["fmt", "--check", "-p", &member.name])
                    .status()
                    .map_err(|e| anyhow!("Failed to execute cargo fmt: {}", e))?;
                if !status.success() {
                    return Err(anyhow!(
                        "Format check execution failed for {}",
                        &member.name
                    ));
                }
                endgroup!();
            }
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All)
                .try_for_each(|t| run_format(&t))?;
        }
    }
    Ok(())
}

fn run_lint(target: &Target) -> anyhow::Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            let members = match target {
                Target::Crates => get_workspace_members(WorkspaceMemberType::Crate),
                Target::Examples => get_workspace_members(WorkspaceMemberType::Example),
                _ => unreachable!(),
            };

            for member in members {
                group!("Lint: {}", member.name);
                info!(
                    "Command line: cargo clippy --no-deps -p {} -- --deny warnings",
                    &member.name
                );
                let status = Command::new("cargo")
                    .args([
                        "clippy",
                        "--no-deps",
                        "-p",
                        &member.name,
                        "--",
                        "--deny",
                        "warnings",
                    ])
                    .status()
                    .map_err(|e| anyhow!("Failed to execute cargo clippy: {}", e))?;
                if !status.success() {
                    return Err(anyhow!("Lint fix execution failed for {}", &member.name));
                }
                endgroup!();
            }
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All)
                .try_for_each(|t| run_lint(&t))?;
        }
    }
    Ok(())
}

fn run_unit_tests(target: &Target) -> anyhow::Result<()> {
    run_unit(&target)
}

fn run_integration_tests(target: &Target) -> anyhow::Result<()> {
    run_integration(&target)
}

fn run_doc_tests(target: &Target) -> anyhow::Result<()> {
    run_documentation(&target)
}

fn run_all_tests(target: &Target) -> anyhow::Result<()> {
    run_unit_tests(&target)?;
    run_integration_tests(&target)?;
    run_doc_tests(&target)?;
    Ok(())
}
