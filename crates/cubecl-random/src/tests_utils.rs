use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(Default, Copy, Clone, Debug)]
pub struct BinStats {
    pub count: usize,
    pub n_runs: usize, // Number of sequences of same bin
}

/// Sorts the data into bins for ranges of equal sizes
pub fn calculate_bin_stats<E: Numeric>(
    numbers: &[E],
    number_of_bins: usize,
    low: f32,
    high: f32,
) -> Vec<BinStats> {
    let range = (high - low) / number_of_bins as f32;
    let mut output: Vec<BinStats> = (0..number_of_bins).map(|_| Default::default()).collect();
    let mut initialized = false;
    let mut current_runs = number_of_bins; // impossible value for starting point
    for number in numbers {
        let num = number.to_f32().unwrap();
        if num < low || num > high {
            continue;
        }
        let index = f32::floor((num - low) / range) as usize;
        output[index].count += 1;
        if initialized && index != current_runs {
            output[current_runs].n_runs += 1;
        }
        initialized = true;
        current_runs = index;
    }
    output[current_runs].n_runs += 1;
    output
}

/// Asserts that the mean of a dataset is approximately equal to an expected value,
/// within 2.5 standard deviations.
/// There is a very small chance this raises a false negative.
pub fn assert_mean_approx_equal<E: Numeric>(data: &[E], expected_mean: f32) {
    let mut sum = 0.;
    for elem in data {
        let elem = elem.to_f32().unwrap();
        sum += elem;
    }
    let mean = sum / (data.len() as f32);

    let mut sum = 0.0;
    for elem in data {
        let elem = elem.to_f32().unwrap();
        let d = elem - mean;
        sum += d * d;
    }
    // sample variance
    let var = sum / ((data.len() - 1) as f32);
    let std = var.sqrt();
    // z-score
    let z = (mean - expected_mean).abs() / std;

    assert!(
        z < 3.,
        "Uniform RNG validation failed: mean={mean}, expected mean={expected_mean}, std={std}",
    );
}

/// Asserts that the distribution follows the 68-95-99 rule of normal distributions,
/// following the given mean and standard deviation.
pub fn assert_normal_respects_68_95_99_rule<E: Numeric>(data: &[E], mu: f32, s: f32) {
    // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    let stats = calculate_bin_stats(data, 6, mu - 3. * s, mu + 3. * s);
    let assert_approx_eq = |count, percent| {
        let expected = percent * data.len() as f32 / 100.;
        assert!(f32::abs(count as f32 - expected) < 2000.);
    };
    assert_approx_eq(stats[0].count, 2.1);
    assert_approx_eq(stats[1].count, 13.6);
    assert_approx_eq(stats[2].count, 34.1);
    assert_approx_eq(stats[3].count, 34.1);
    assert_approx_eq(stats[4].count, 13.6);
    assert_approx_eq(stats[5].count, 2.1);
}

/// For a bernoulli distribution: asserts that the proportion of 1s to 0s is approximately equal
/// to the expected probability.
pub fn assert_number_of_1_proportional_to_prob<E: Numeric>(data: &[E], prob: f32) {
    // High bound slightly over 1 so 1.0 is included in second bin
    let bin_stats = calculate_bin_stats(data, 2, 0., 1.1);
    assert!(f32::abs((bin_stats[1].count as f32 / data.len() as f32) - prob) < 0.05);
}

/// Asserts that the elements of the data, sorted into two bins, are elements of the sequence
/// are mutually independent.
/// There is a very small chance it gives a false negative.
pub fn assert_wald_wolfowitz_runs_test<E: Numeric>(data: &[E], bins_low: f32, bins_high: f32) {
    //https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test
    let stats = calculate_bin_stats(data, 2, bins_low, bins_high);
    let n_0 = stats[0].count as f32;
    let n_1 = stats[1].count as f32;
    let n_runs = (stats[0].n_runs + stats[1].n_runs) as f32;

    let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
    let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
        / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
    let z = (n_runs - expectation) / f32::sqrt(variance);

    // below 2 means we can have good confidence in the randomness
    // we put 2.6 to make sure it passes even when very unlucky.
    // With higher vectorization, adjacent values are more
    // correlated, which makes this test is more flaky.
    assert!(z.abs() < 2.6, "z: {}, var: {}", z, variance);
}

/// Asserts that there is at least one value per bin
pub fn assert_at_least_one_value_per_bin<E: Numeric>(
    data: &[E],
    number_of_bins: usize,
    bins_low: f32,
    bins_high: f32,
) {
    let stats = calculate_bin_stats(data, number_of_bins, bins_low, bins_high);
    assert!(stats[0].count >= 1);
    assert!(stats[1].count >= 1);
    assert!(stats[2].count >= 1);
}
