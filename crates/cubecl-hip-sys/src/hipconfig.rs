use std::process::Command;

use regex::Regex;

pub const HIPCONFIG: &str = "hipconfig";

/// Retrieve the ROCM_PATH with `hipconfig -R` command.
pub fn get_rocm_path() -> std::io::Result<String> {
    exec_hipconfig(&["-R"])
}

/// Retrieve the HIP_PATH with `hipconfig -p` command.
pub fn get_hip_path() -> std::io::Result<String> {
    exec_hipconfig(&["-p"])
}

/// Retrieve the HIP patch number from the `hipconfig --version` output
pub fn get_hip_patch_version() -> std::io::Result<String> {
    let hip_version = exec_hipconfig(&["--version"])?;
    parse_hip_patch_number(&hip_version)
}

/// Return the HIP path suitable for LD_LIBRARY_PATH.
pub fn get_hip_ld_library_path() -> std::io::Result<String> {
    let rocm_path = get_rocm_path()?;
    Ok(format!("{rocm_path}/lib"))
}

/// Return the include path for HIP
pub fn get_hip_include_path() -> std::io::Result<String> {
    let hip_path = get_hip_path()?;
    Ok(format!("{hip_path}/include"))
}

/// Read the file at `path`, then return the latest `hip_<patch>` feature if any.
pub fn extract_latest_hip_feature_from_path<P: AsRef<std::path::Path>>(
    path: P,
) -> std::io::Result<String> {
    let s = std::fs::read_to_string(path)?;
    match extract_latest_hip_feature_from_contents(&s) {
        Some(feature) => Ok(feature),
        None => {
            let feature = "hip_43482";
            println!("cargo::warning=Failed to retrieve the latest feature from 'cubecl-hip-sys' cargo file.\nThis is a bug, open an issue if you can. Fallback to feature '{feature}'.");
            Ok(feature.to_owned())
        }
    }
}

/// Execute hipconfig
fn exec_hipconfig(args: &[&str]) -> std::io::Result<String> {
    match Command::new(HIPCONFIG).args(args).output() {
        Ok(output) => {
            if output.stderr.is_empty() {
                Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                panic!(
                    "Error executing {HIPCONFIG}. The process returned:\n{}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("cargo::warning=Could not find '{HIPCONFIG}' in your PATH. You should install ROCm HIP or ensure '{HIPCONFIG}' is available. For more information please visit https://rocm.docs.amd.com/projects/install-on-linux/en/latest/.");
            Err(e)
        }
        Err(e) => panic!(
            "Failed to run '{HIPCONFIG}' with args '{args:?}', reason: {}",
            e
        ),
    }
}

/// Extract the latest `hip_<patch>` feature from the given Cargo.toml contents.
/// Return `Some("hip_<max_patch>")` or `None` if no such feature is found.
fn extract_latest_hip_feature_from_contents(toml: &str) -> Option<String> {
    // Matches lines like `hip_12345 = []`, capturing the digits.
    let re = Regex::new(r"(?m)^\s*hip_(\d+)\s*=\s*\[\]").expect("regex should compile");
    let mut max_patch: Option<u32> = None;

    for cap in re.captures_iter(toml) {
        if let Some(m) = cap.get(1) {
            if let Ok(n) = m.as_str().parse::<u32>() {
                max_patch = Some(max_patch.map_or(n, |cur| cur.max(n)));
            }
        }
    }

    max_patch.map(|n| format!("hip_{n}"))
}

/// Extract the HIP patch number from hipconfig version output
fn parse_hip_patch_number(version: &str) -> std::io::Result<String> {
    let re = Regex::new(r"\d+\.\d+\.(\d+)-").expect("regex should compile");
    if let Some(caps) = re.captures(version) {
        if let Some(m) = caps.get(1) {
            return Ok(m.as_str().to_string());
        }
    }
    // cannot parse for the patch number
    panic!("Error retrieving HIP patch number from value '{version}'")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case::standard("6.4.43482-0f2d60242", Some("43482"))]
    #[case::with_rc_suffix("10.20.54321-rc1", Some("54321"))]
    #[case::leading_zeros("6.4.00099-test", Some("00099"))]
    #[case::missing_hyphen("6.4.43482", None)]
    #[case::completely_invalid("no numbers", None)]
    fn test_parse_hip_patch_number(#[case] input: &str, #[case] expected: Option<&str>) {
        let result = std::panic::catch_unwind(|| parse_hip_patch_number(input));
        match expected {
            Some(expected_str) => {
                let output = result.expect("should not panic for valid version");
                assert_eq!(
                    output.unwrap(),
                    expected_str,
                    "parsed patch number should match expected"
                );
            }
            None => {
                assert!(result.is_err(), "should panic for invalid version output");
            }
        }
    }

    #[rstest]
    #[case::standard(
        r#"[features]
default = []
hip_41134 = []
hip_42131 = []
hip_42133 = []
hip_42134 = []
hip_43482 = []
"#,
        Some("hip_43482")
    )]
    #[case::unordered(
        r#"[features]
hip_42133 = []
hip_43482 = []
hip_42131 = []
hip_42134 = []
hip_41134 = []
default = []
"#,
        Some("hip_43482")
    )]
    #[case::with_comments(
        r#"[features]
# Supported HIP patch versions
hip_10000 = []
# legacy
hip_02000 = []
"#,
        Some("hip_10000")
    )]
    #[case::no_hip_features(
        r#"[features]
default = []
foo = []
bar = []
"#,
        None
    )]
    #[case::no_features_section(
        r#"workspace = true
name = "example"
version = "0.1.0"
"#,
        None
    )]
    fn test_extract_latest_hip_feature_from_path(
        #[case] contents: &str,
        #[case] expected: Option<&str>,
    ) {
        let got = extract_latest_hip_feature_from_contents(contents);
        assert_eq!(got.as_deref(), expected);
    }
}
