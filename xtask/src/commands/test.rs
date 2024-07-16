use std::process::Command;

use anyhow::{anyhow, Ok, Result};
use clap::{Args, Subcommand};
use strum::{Display, EnumIter, EnumString, IntoEnumIterator};

use crate::{
    endgroup, group,
    utils::workspace::{get_workspace_members, WorkspaceMember, WorkspaceMemberType},
};

use super::Target;

const PROJECT_UUID: &str = "331a3907-bfd8-45e5-af54-1fee73a3c1b1";
const API_KEY: &str = "dcaf7eb9-5acc-47d7-8b93-ca0fbb234096";

#[derive(Args)]
pub(crate) struct TestCmdArgs {
    /// Target to test for.
    #[arg(short, long, value_enum)]
    target: Target,
    #[command(subcommand)]
    command: TestCommand,
}

#[derive(EnumString, EnumIter, Display, Clone, PartialEq, Subcommand)]
#[strum(serialize_all = "lowercase")]
enum TestCommand {
    /// Run unit tests.
    Unit,
    /// Run integration tests.
    Integration,
    /// Run documentation tests.
    Documentation,
    /// Run guide test against Heat dev stack.
    Guide,
    /// Run all the checks.
    All,
}

pub(crate) fn handle_command(args: TestCmdArgs) -> anyhow::Result<()> {
    match args.command {
        TestCommand::Unit => run_unit(&args.target),
        TestCommand::Integration => run_integration(&args.target),
        TestCommand::Documentation => run_documentation(&args.target),
        TestCommand::Guide => run_guide(),
        TestCommand::All => TestCommand::iter()
            .filter(|c| *c != TestCommand::All)
            .try_for_each(|c| {
                handle_command(TestCmdArgs {
                    command: c,
                    target: args.target.clone(),
                })
            }),
    }
}

pub(crate) fn run_guide() -> Result<()> {
    group!("Guide Test");
    info!("Command line: cargo run --release --bin guide -- --key \"...\" --project \"...\"");
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "guide",
            "--",
            "--key",
            API_KEY,
            "--project",
            PROJECT_UUID,
        ])
        .status()
        .map_err(|e| anyhow!("Failed to execute guide example: {}", e))?;
    if !status.success() {
        return Err(anyhow!("Failed to execute guide example"));
    }
    endgroup!();
    Ok(())
}

pub(crate) fn run_unit(target: &Target) -> Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            let members = match target {
                Target::Crates => get_workspace_members(WorkspaceMemberType::Crate),
                Target::Examples => get_workspace_members(WorkspaceMemberType::Example),
                _ => unreachable!(),
            };

            for member in members {
                run_unit_test(&member)?;
            }
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All)
                .try_for_each(|t| run_unit(&t))?;
        }
    }
    Ok(())
}

fn run_unit_test(member: &WorkspaceMember) -> Result<(), anyhow::Error> {
    group!("Unit Tests: {}", member.name);
    info!("Command line: cargo test --lib --bins -p {}", &member.name);
    let status = Command::new("cargo")
        .args(["test", "--lib", "--bins", "-p", &member.name])
        .status()
        .map_err(|e| anyhow!("Failed to execute unit test: {}", e))?;
    if !status.success() {
        return Err(anyhow!("Failed to execute unit test for {}", &member.name));
    }
    endgroup!();
    Ok(())
}

pub(crate) fn run_documentation(target: &Target) -> Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            let members = match target {
                Target::Crates => get_workspace_members(WorkspaceMemberType::Crate),
                Target::Examples => get_workspace_members(WorkspaceMemberType::Example),
                _ => unreachable!(),
            };

            for member in members {
                run_doc_test(&member)?;
            }
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All)
                .try_for_each(|t| run_documentation(&t))?;
        }
    }
    Ok(())
}

fn run_doc_test(member: &WorkspaceMember) -> Result<(), anyhow::Error> {
    group!("Doc Tests: {}", member.name);
    info!("Command line: cargo test --doc -p {}", &member.name);
    let status = Command::new("cargo")
        .args(["test", "--doc", "-p", &member.name])
        .status()
        .map_err(|e| anyhow!("Failed to execute documentation test: {}", e))?;
    if !status.success() {
        return Err(anyhow!(
            "Failed to execute documentation test for {}",
            &member.name
        ));
    }
    endgroup!();
    Ok(())
}

pub(crate) fn run_integration(target: &Target) -> anyhow::Result<()> {
    match target {
        Target::Crates | Target::Examples => {
            let members = match target {
                Target::Crates => get_workspace_members(WorkspaceMemberType::Crate),
                Target::Examples => get_workspace_members(WorkspaceMemberType::Example),
                _ => unreachable!(),
            };

            for member in members {
                run_integration_test(&member)?;
            }
        }
        Target::All => {
            Target::iter()
                .filter(|t| *t != Target::All)
                .try_for_each(|t| run_integration(&t))?;
        }
    }
    Ok(())
}

fn run_integration_test(member: &WorkspaceMember) -> Result<()> {
    group!("Integration Tests: {}", &member.name);
    info!(
        "Command line: cargo test --test \"test_*\" -p {}",
        &member.name
    );
    let output = Command::new("cargo")
        .args(["test", "--test", "test_*", "-p", &member.name])
        .output()
        .map_err(|e| anyhow!("Failed to execute integration test: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("no test target matches pattern") {
            warn!(
                "No tests found matching the pattern `test_*` for {}",
                &member.name
            );
            endgroup!();
            return Ok(());
        }
        return Err(anyhow!(
            "Failed to execute integration test for {}: {}",
            &member.name,
            stderr
        ));
    }
    endgroup!();
    Ok(())
}
