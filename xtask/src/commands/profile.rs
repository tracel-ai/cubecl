use std::path::Path;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct ProfileArgs {
    #[command(subcommand)]
    command: ProfileSubCommand,
}

#[derive(clap::Subcommand, strum::Display)]
pub(crate) enum ProfileSubCommand {
    /// Profole matmul on cuda.
    Matmul(ProfileOptionsArgs),
}

#[derive(clap::Args)]
pub(crate) struct ProfileOptionsArgs {
    #[arg(long, default_value = "/usr/local/cuda/bin/ncu")]
    pub ncu_path: String,
    #[arg(long, default_value = "/usr/local/cuda/bin/ncu-ui")]
    pub ncu_ui_path: String,
}

pub(crate) struct Profile {}

impl ProfileArgs {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        Profile::run(&self.command)
    }
}

impl Profile {
    pub(crate) fn run(args: &ProfileSubCommand) -> anyhow::Result<()> {
        Profile {}.execute(args)
    }

    fn execute(&self, command: &ProfileSubCommand) -> anyhow::Result<()> {
        ensure_cargo_crate_is_installed("mdbook", None, None, false)?;
        group!("Profile: {}", command);
        match command {
            ProfileSubCommand::Matmul(options) => self.matmul(options),
        }?;
        endgroup!();
        Ok(())
    }

    fn matmul(&self, options: &ProfileOptionsArgs) -> anyhow::Result<()> {
        run_process(
            "cargo",
            &[
                "build",
                "--bin",
                "matmul-profile",
                "--release",
                "--features",
                "cuda",
            ],
            None,
            None,
            "Can build matmul-profile.",
        )?;

        Self::profile("matmul-profile", options)
    }

    fn profile(name: &str, options: &ProfileOptionsArgs) -> anyhow::Result<()> {
        const TARGET_RELEASE_PATH: &'static str = "./target/release";
        let target_path = Path::new(TARGET_RELEASE_PATH);

        run_process(
            "sudo",
            &[
                &options.ncu_path,
                "--config-file",
                "off",
                "--export",
                name,
                "--force-overwrite",
                name,
            ],
            None,
            Some(&target_path),
            format!("Can profile {name}").as_str(),
        )?;

        let output = format!("{name}.ncu-rep");
        run_process(
            &options.ncu_ui_path,
            &[&output],
            None,
            Some(target_path),
            format!("Can open results for {name}").as_str(),
        )
    }
}
