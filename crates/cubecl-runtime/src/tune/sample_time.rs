use core::time::Duration;
use cubecl_common::profile::ProfileTicks;

/// A resolved sample's time, or `None` where it is no time: unmeasured, or zero. No launch that
/// ran takes zero; a zero is a window that timed nothing, such as a candidate whose kernel never
/// launched, and recorded it would be the fastest sample in the round and end it on the spot.
pub fn measured(ticks: Option<ProfileTicks>) -> Option<Duration> {
    ticks
        .map(|ticks| ticks.duration())
        .filter(|duration| !duration.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::profile::Instant;

    #[test]
    fn a_zero_sample_is_no_time() {
        let now = Instant::now();
        assert_eq!(measured(Some(ProfileTicks::from_start_end(now, now))), None);
        assert_eq!(measured(None), None);
    }

    #[test]
    fn a_timed_sample_is_its_duration() {
        let now = Instant::now();
        let later = now + Duration::from_micros(3);
        assert_eq!(
            measured(Some(ProfileTicks::from_start_end(now, later))),
            Some(Duration::from_micros(3))
        );
    }
}
