use std::{
    collections::HashMap,
    path::Path,
    process::{Command, Stdio},
};

use crate::{endgroup, group};

use super::{process::run_process_command, Params};

/// Return a cargo command without executing it
pub fn cargo_command(command: &str, params: Params, envs: HashMap<&str, &str>) -> Command {
    cargo_command_with_path::<String>(command, params, envs, None)
}

/// Run a cargo command with the passed directory as the current directory
pub fn cargo_command_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    envs: HashMap<&str, &str>,
    path: Option<P>,
) -> Command {
    let mut cargo = Command::new("cargo");
    cargo
        .env("CARGO_INCREMENTAL", "0")
        .envs(&envs)
        .arg(command)
        .args(&params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal

    if let Some(path) = path {
        cargo.current_dir(path);
    }

    cargo
}

/// Run a cargo command
pub fn run_cargo(
    command: &str,
    params: Params,
    envs: HashMap<&str, &str>,
    error: &str,
) -> anyhow::Result<()> {
    let mut cargo = cargo_command(command, params.clone(), envs);
    run_process_command(&mut cargo, error)
}

/// Ensure that a cargo crate is installed
pub fn ensure_cargo_crate_is_installed(
    crate_name: &str,
    features: Option<&str>,
    locked: bool,
) -> anyhow::Result<()> {
    if !is_cargo_crate_installed(crate_name) {
        group!("Cargo: install crate '{}'", crate_name);
        let mut args = vec![crate_name];
        if locked {
            args.push("--locked");
        }
        if let Some(features) = features {
            if !features.is_empty() {
                args.push("--features");
                args.push(&features);
            }
        }
        run_cargo(
            "install",
            args.into(),
            HashMap::new(),
            &format!("crate '{}' should be installed", crate_name),
        )?;
        endgroup!();
    }
    Ok(())
}

/// Returns true if the passed cargo crate is installed locally
fn is_cargo_crate_installed(crate_name: &str) -> bool {
    let output = Command::new("cargo")
        .arg("install")
        .arg("--list")
        .output()
        .expect("Should get the list of installed cargo commands");
    let output_str = String::from_utf8_lossy(&output.stdout);
    output_str.lines().any(|line| line.contains(crate_name))
}
