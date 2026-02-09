use tracel_xtask::prelude::*;

const CI_EXCLUDED_CRATES: &[&str] = &["cubecl-cuda", "cubecl-hip", "cubecl-metal"];

#[macros::extend_command_args(CheckCmdArgs, Target, CheckSubCommand)]
pub struct CubeCLCheckCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLCheckCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    let command = args.command.clone();

    if args.ci {
        args.exclude
            .extend(CI_EXCLUDED_CRATES.iter().map(|s| s.to_string()));
    }

    let ci_lint = args.ci
        && matches!(
            command,
            None | Some(CheckSubCommand::All) | Some(CheckSubCommand::Lint)
        );

    if ci_lint {
        if matches!(command, None | Some(CheckSubCommand::All)) {
            for sub in &[
                CheckSubCommand::Audit,
                CheckSubCommand::Format,
                CheckSubCommand::Typos,
            ] {
                let mut sub_args = args.clone();
                sub_args.command = Some(sub.clone());
                base_commands::check::handle_command(
                    sub_args.try_into().unwrap(),
                    env.clone(),
                    context.clone(),
                )?;
            }
        }
        run_ci_lint()?;
    } else {
        base_commands::check::handle_command(args.try_into().unwrap(), env, context)?;
    }

    // Additional feature-specific checks
    #[cfg(not(target_os = "macos"))]
    helpers::custom_crates_check(
        vec!["cubecl-wgpu"],
        vec!["--features", "spirv"],
        None,
        None,
        "std with SPIR-V compiler",
    )?;
    helpers::custom_crates_check(
        vec!["cubecl-wgpu"],
        vec!["--features", "exclusive-memory-only"],
        None,
        None,
        "std with exclusive_memory_only",
    )?;
    helpers::custom_crates_check(
        vec!["cubecl-runtime"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}

/// Run workspace lint, excluding platform-specific crates.
fn run_ci_lint() -> anyhow::Result<()> {
    let mut cmd_args: Vec<&str> = vec!["clippy", "--workspace", "--no-deps", "--color=always"];
    for crate_name in CI_EXCLUDED_CRATES {
        cmd_args.push("--exclude");
        cmd_args.push(crate_name);
    }
    cmd_args.extend(&["--", "--deny", "warnings"]);

    run_process("cargo", &cmd_args, None, None, "Workspace lint failed")
}
