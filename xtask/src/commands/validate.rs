use tracel_xtask::prelude::*;

use crate::commands::{build::CubeCLBuildCmdArgs, test::CubeCLTestCmdArgs};

pub fn handle_command(args: &ValidateCmdArgs) -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    // checks
    [
        CheckSubCommand::Audit,
        CheckSubCommand::Format,
        CheckSubCommand::Lint,
        CheckSubCommand::Typos,
    ]
    .iter()
    .try_for_each(|c| {
        base_commands::check::handle_command(CheckCmdArgs {
            target: target.clone(),
            exclude: exclude.clone(),
            only: only.clone(),
            command: Some(c.clone()),
            ignore_audit: args.ignore_audit,
        })
    })?;

    // build
    super::build::handle_command(CubeCLBuildCmdArgs {
        target: target.clone(),
        exclude: exclude.clone(),
        only: only.clone(),
        ci: true,
    })?;

    // tests
    super::test::handle_command(CubeCLTestCmdArgs {
        target: target.clone(),
        exclude: exclude.clone(),
        only: only.clone(),
        threads: None,
        jobs: None,
        command: Some(TestSubCommand::All),
        ci: true,
        features: None,
    })?;

    // documentation
    [DocSubCommand::Build, DocSubCommand::Tests]
        .iter()
        .try_for_each(|c| {
            base_commands::doc::handle_command(DocCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                command: Some(c.clone()),
            })
        })?;

    Ok(())
}
