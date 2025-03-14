use std::{io::Write, process::Command};

/// Format C++ code, useful when debugging.
pub fn format_cpp(code: &str) -> Result<String, std::io::Error> {
    let mut child = Command::new("clang-format")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(code.as_bytes())?;
    }

    let output = child.wait_with_output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(std::io::Error::other("clang-format failed"))
    }
}
