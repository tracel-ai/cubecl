/// Controls how Streaming Multiprocessors (SMs) are assigned cubes.
///
/// - `Exact`: Balanced allocation using GCD (e.g., 120 cubes, 16 SMs → 4 SMs × 30 cubes)
/// - `Full`: Uses all SMs even if it overallocates (e.g., 120 cubes, 16 SMs → 16 SMs × 8 cubes = 128 total cubes)
/// - `Overallocate`: Allows extra SMs within a specified fraction (e.g., up to 25% overuse)
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SmAllocation {
    /// Balanced: uses GCD to never exceed total cubes.
    Exact,

    /// Uses all SMs, possibly overallocating total cubes.
    Full,

    /// Allows overallocating SMs up to a ratio.
    Ratio {
        max_extra_numerator: u32,
        max_extra_denominator: u32,
    },
}

impl SmAllocation {
    /// Returns a pair (`num_sms_used`, `cubes_per_sm`) depending on the strategy
    pub fn allocate(&self, num_sms: u32, total_cubes: u32) -> (u32, u32) {
        match self {
            SmAllocation::Exact => SmAllocation::Ratio {
                max_extra_numerator: 0,
                max_extra_denominator: 1,
            }
            .allocate(num_sms, total_cubes),

            SmAllocation::Full => SmAllocation::Ratio {
                max_extra_numerator: u32::MAX,
                max_extra_denominator: 1,
            }
            .allocate(num_sms, total_cubes),

            SmAllocation::Ratio {
                max_extra_numerator,
                max_extra_denominator,
            } => {
                let max_slack = num_sms
                    .saturating_mul(*max_extra_numerator)
                    .div_ceil(*max_extra_denominator);

                let fallback_cubes_per_sm = total_cubes.div_ceil(num_sms);
                let mut best = (num_sms, fallback_cubes_per_sm);

                // Generate divisors in descending order
                let divisors_desc = |n: u32| {
                    let mut divs = Vec::new();
                    let mut i = 1;

                    while i * i <= n {
                        if n.is_multiple_of(i) {
                            divs.push(i);
                            if i != n / i {
                                divs.push(n / i);
                            }
                        }
                        i += 1;
                    }

                    divs.sort_by(|a, b| b.cmp(a)); // descending
                    divs.into_iter()
                };

                for sms_used in divisors_desc(num_sms) {
                    let cubes_per_sm = total_cubes.div_ceil(sms_used);
                    let total_allocated = cubes_per_sm * sms_used;
                    let slack = total_allocated.saturating_sub(total_cubes);

                    if slack <= max_slack {
                        best = (sms_used, cubes_per_sm);
                        break;
                    }
                }

                best
            }
        }
    }
}
