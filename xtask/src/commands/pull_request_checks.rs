use strum::IntoEnumIterator;

use super::{
    ci::{self, CICmdArgs, CICommand},
    test::run_guide,
};

pub(crate) fn handle_command() -> anyhow::Result<()> {
    CICommand::iter()
        // Skip audit command
        .filter(|c| *c != CICommand::All && *c != CICommand::AllTests && *c != CICommand::Audit)
        .try_for_each(|c| {
            ci::handle_command(CICmdArgs {
                target: super::Target::All,
                command: c.clone(),
            })
        })?;
    // Execute the guide example
    run_guide()
}
