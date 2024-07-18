use serde_json::Value;
use std::{path::Path, process::Command};

const MEMBER_PATH_PREFIX: &str = if cfg!(target_os = "windows") {
    "path+file:///"
} else {
    "path+file://"
};

pub(crate) enum WorkspaceMemberType {
    Crate,
    Example,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct WorkspaceMember {
    pub(crate) name: String,
    pub(crate) path: String,
}

impl WorkspaceMember {
    fn new(name: String, path: String) -> Self {
        Self { name, path }
    }
}

/// Get workspace crates
pub(crate) fn get_workspace_members(w_type: WorkspaceMemberType) -> Vec<WorkspaceMember> {
    let cuda_available = Command::new("nvcc").arg("--version").output().is_ok();
    // Run `cargo metadata` command to get project metadata
    let output = Command::new("cargo")
        .arg("metadata")
        .output()
        .expect("Failed to execute command");
    // Parse the JSON output
    let metadata: Value = serde_json::from_slice(&output.stdout).expect("Failed to parse JSON");
    // Extract workspace members from the metadata
    let workspaces = metadata["workspace_members"]
        .as_array()
        .expect("Expected an array of workspace members")
        .iter()
        .filter_map(|member| {
            let member_str = member.as_str()?;
            let has_whitespace = member_str.chars().any(|c| c.is_whitespace());
            let (name, path) = if has_whitespace {
                parse_workspace_member0(member_str)?
            } else {
                parse_workspace_member1(member_str)?
            };

            if !cuda_available {
                if path.contains("cuda") {
                    return None;
                }
            }

            match w_type {
                WorkspaceMemberType::Crate if path.contains("crates/") => {
                    Some(WorkspaceMember::new(name.to_string(), path.to_string()))
                }
                WorkspaceMemberType::Example if path.contains("examples/") => {
                    Some(WorkspaceMember::new(name.to_string(), path.to_string()))
                }
                _ => None,
            }
        })
        .collect();

    workspaces
}

/// Legacy cargo metadata format for member specs (rust < 1.77)
/// Example:
/// "backend-comparison 0.13.0 (path+file:///Users/username/burn/backend-comparison)"
fn parse_workspace_member0(specs: &str) -> Option<(String, String)> {
    let parts: Vec<_> = specs.split_whitespace().collect();
    let (name, path) = (parts.first()?.to_owned(), parts.last()?.to_owned());
    // skip the first character because it is a '('
    let path = path
        .chars()
        .skip(1)
        .collect::<String>()
        .replace(MEMBER_PATH_PREFIX, "")
        .replace(')', "");
    Some((name.to_string(), path.to_string()))
}

/// Cargo metadata format for member specs (rust >= 1.77)
/// Example:
/// "path+file:///Users/username/burn/backend-comparison#0.13.0"
fn parse_workspace_member1(specs: &str) -> Option<(String, String)> {
    let no_prefix = specs.replace(MEMBER_PATH_PREFIX, "").replace(')', "");
    let path = Path::new(no_prefix.split_once('#')?.0);
    let name = path.file_name()?.to_str()?;
    let path = path.to_str()?;
    Some((name.to_string(), path.to_string()))
}
