//! The `autotune-checks` cross-check: every candidate's output against a reference candidate's.

use alloc::string::String;
use alloc::vec::Vec;

use crate::tune::{AutotuneCheckable, AutotuneError, CheckResult};

/// Check every candidate's output of `outputs`, each with its name, against the first one that
/// served the call, or the last one where none did. A mismatch is collected as a failed check where
/// `collect`, where something reports it, and panics otherwise rather than passing silently.
pub fn check_outputs<O: AutotuneCheckable>(
    mut outputs: Vec<(String, Result<O, AutotuneError>)>,
    collect: bool,
) -> Vec<CheckResult> {
    let Some(reference_index) = outputs
        .iter()
        .position(|(_, output)| output.is_ok())
        .or(outputs.len().checked_sub(1))
    else {
        return Vec::new();
    };
    let (reference_name, reference) = outputs.remove(reference_index);
    let mut checks: Vec<CheckResult> = outputs
        .into_iter()
        .map(|(name, other)| check(&reference, name, other, collect))
        .collect();
    checks.push(CheckResult {
        name: reference_name,
        passed: reference.is_ok(),
    });
    checks
}

/// The check of the candidate `name`, whose output is `other`, against `reference`. A candidate
/// that did not serve the call fails it.
fn check<O: AutotuneCheckable>(
    reference: &Result<O, AutotuneError>,
    name: String,
    other: Result<O, AutotuneError>,
    collect: bool,
) -> CheckResult {
    let (Ok(reference), Ok(other)) = (reference, other) else {
        return CheckResult {
            name,
            passed: false,
        };
    };
    let mismatch = reference.check_equivalence(other).err();
    assert!(
        collect || mismatch.is_none(),
        "the output of the autotune candidate `{name}` should match the reference output: {mismatch:?}"
    );
    CheckResult {
        passed: mismatch.is_none(),
        name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    /// An output that matches another of the same value.
    struct Value(u8);

    impl AutotuneCheckable for Value {
        fn check_equivalence(&self, other: Self) -> Result<(), String> {
            if self.0 == other.0 {
                Ok(())
            } else {
                Err(alloc::format!("{} against {}", other.0, self.0))
            }
        }
    }

    fn rejected() -> Result<Value, AutotuneError> {
        Err(AutotuneError::Rejected {
            name: "rejects".into(),
            reason: "not this shape".into(),
        })
    }

    fn named(
        name: &str,
        output: Result<Value, AutotuneError>,
    ) -> (String, Result<Value, AutotuneError>) {
        (name.to_string(), output)
    }

    #[test]
    fn the_first_served_output_is_the_reference_and_comes_last() {
        let checks = check_outputs(
            vec![
                named("rejects", rejected()),
                named("reference", Ok(Value(1))),
                named("same", Ok(Value(1))),
            ],
            true,
        );

        let names: Vec<_> = checks
            .iter()
            .map(|check| (check.name.as_str(), check.passed))
            .collect();
        assert_eq!(
            names,
            [("rejects", false), ("same", true), ("reference", true)]
        );
    }

    #[test]
    fn a_collected_mismatch_fails_its_check() {
        let checks = check_outputs(
            vec![
                named("reference", Ok(Value(1))),
                named("other", Ok(Value(2))),
            ],
            true,
        );

        assert!(!checks[0].passed);
    }

    #[test]
    #[should_panic(expected = "should match the reference output")]
    fn a_mismatch_nothing_reports_panics() {
        check_outputs(
            vec![
                named("reference", Ok(Value(1))),
                named("other", Ok(Value(2))),
            ],
            false,
        );
    }

    #[test]
    fn no_output_checks_nothing() {
        assert!(check_outputs::<Value>(Vec::new(), true).is_empty());
    }
}
