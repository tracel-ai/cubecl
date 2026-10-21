use super::{HostData, ValidationResult};

/// How many mismatches a `Fail` reason lists before it only counts them.
const REPORTED_MISMATCHES: usize = 8;

/// Whether `actual` matches `expected` everywhere within `epsilon`, absolute or relative to the
/// expected value, whichever is looser. Two NaNs match, as do two infinities of one sign.
pub fn assert_equals_approx(
    actual: &HostData,
    expected: &HostData,
    epsilon: f32,
) -> ValidationResult {
    if actual.shape != expected.shape {
        return ValidationResult::Fail(format!(
            "Shape mismatch: got {:?}, expected {:?}",
            actual.shape, expected.shape,
        ));
    }

    let mut mismatches = Mismatches::default();
    compare(
        actual,
        expected,
        epsilon as f64,
        &mut Vec::new(),
        &mut mismatches,
    );

    if mismatches.count == 0 {
        ValidationResult::Pass
    } else {
        ValidationResult::Fail(mismatches.report(&actual.shape))
    }
}

/// The mismatches of one comparison: how many, the worst, and the first few in full.
#[derive(Default)]
struct Mismatches {
    compared: usize,
    count: usize,
    worst: Option<(Vec<usize>, f64)>,
    listed: Vec<(Vec<usize>, String)>,
}

impl Mismatches {
    fn record(&mut self, index: &[usize], delta: f64, reason: String) {
        self.count += 1;
        if self.worst.as_ref().is_none_or(|(_, worst)| delta > *worst) {
            self.worst = Some((index.to_vec(), delta));
        }
        if self.listed.len() < REPORTED_MISMATCHES {
            self.listed.push((index.to_vec(), reason));
        }
    }

    fn report(&self, shape: &[usize]) -> String {
        let mut out = format!(
            "Got incorrect results: {}/{} elements mismatched, shape={shape:?}",
            self.count, self.compared
        );
        if let Some((index, delta)) = &self.worst {
            out.push_str(&format!(", max |Δ|={delta:.6} at {index:?}"));
        }
        out.push_str("\nFirst mismatches:");
        for (index, reason) in &self.listed {
            out.push_str(&format!("\n  {index:?}: {reason}"));
        }
        if self.count > self.listed.len() {
            out.push_str(&format!(
                "\n  ... and {} more",
                self.count - self.listed.len()
            ));
        }
        out
    }
}

fn compare(
    actual: &HostData,
    expected: &HostData,
    epsilon: f64,
    index: &mut Vec<usize>,
    mismatches: &mut Mismatches,
) {
    let dim = index.len();
    if dim < actual.shape.len() {
        for i in 0..actual.shape[dim] {
            index.push(i);
            compare(actual, expected, epsilon, index, mismatches);
            index.pop();
        }
        return;
    }

    mismatches.compared += 1;
    let got = actual.get_f64(index);
    let want = expected.get_f64(index);
    let epsilon = (epsilon * want).abs().max(epsilon);

    if got.is_nan() || want.is_nan() {
        if got.is_nan() != want.is_nan() {
            mismatches.record(index, f64::INFINITY, format!("got {got}, expected {want}"));
        }
    } else if got.is_infinite() && want.is_infinite() {
        if got.signum() != want.signum() {
            mismatches.record(index, f64::INFINITY, format!("got {got}, expected {want}"));
        }
    } else {
        let delta = (got - want).abs();
        if delta > epsilon {
            mismatches.record(
                index,
                delta,
                format!("got {got}, expected {want}, |Δ|={delta} > ε={epsilon}"),
            );
        }
    }
}
