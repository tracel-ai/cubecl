use std::process::{Child, Command};

use crate::utils::get_command_line_from_command;

/// Spawn a process from passed command
pub fn run_process_command(command: &mut Command, error: &str) -> anyhow::Result<()> {
    // Handle cargo child process
    let command_line = get_command_line_from_command(command);
    info!("{command_line}\n");
    let process = command.spawn().expect(error);
    let error = format!(
        "{} process should run flawlessly",
        command.get_program().to_str().unwrap()
    );
    handle_child_process(process, &error)
}

/// Handle child process
pub fn handle_child_process(mut child: Child, error: &str) -> anyhow::Result<()> {
    // Wait for the child process to finish
    let status = child.wait().expect(error);

    // If exit status is not a success, terminate the process with an error
    if !status.success() {
        // Use the exit code associated to a command to terminate the process,
        // if any exit code had been found, use the default value 1
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
