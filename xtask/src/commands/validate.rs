use tracel_xtask::prelude::*;

use crate::commands::{build::CubeCLBuildCmdArgs, test::CubeCLTestCmdArgs};

pub fn handle_command(
    args: &ValidateCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
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
            base_commands::check::handle_command(
                CheckCmdArgs {
                    command: Some(c.clone()),
                    exclude: exclude.clone(),
                    ignore_audit: args.ignore_audit,
                    only: only.clone(),
                    target: target.clone(),
                },
                env.clone(),
                context.clone(),
        )
    })?;

    // build
    super::build::handle_command(
         CubeCLBuildCmdArgs {
             ci: true,
             exclude: exclude.clone(),
             only: only.clone(),
             release: false,
             target: target.clone(),
         },
        env.clone(),
        context.clone(),
    )?;

    // tests
    super::test::handle_command(
        CubeCLTestCmdArgs {
            ci: true,
            command: Some(TestSubCommand::All),
            exclude: exclude.clone(),
            features: None,
            force: false,
            jobs: None,
            no_capture: false,
            no_default_features: false,
            only: only.clone(),
            release: args.release,
            target: target.clone(),
            test: None,
            threads: None,
        },
        env.clone(),
        context.clone(),
    )?;

    // documentation
    [DocSubCommand::Build, DocSubCommand::Tests]
        .iter()
        .try_for_each(|c| {
            base_commands::doc::handle_command(
                DocCmdArgs {
                    target: target.clone(),
                    exclude: exclude.clone(),
                    only: only.clone(),
                    command: Some(c.clone()),
                },
                env.clone(),
                context.clone(),
            )
        })?;

    Ok(())
}
