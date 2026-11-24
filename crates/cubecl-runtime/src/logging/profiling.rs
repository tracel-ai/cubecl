use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Display;
use core::time::Duration;
use cubecl_environment::collections::HashMap;

/// Control the amount of info being display when profiling.
#[derive(Debug, Copy, Clone)]
pub enum ProfileLevel {
    /// Provide only the summary information about kernels being run.
    Basic,
    /// Provide the summary information about kernels being run with their trace.
    Medium,
    /// Provide more information about kernels being run.
    Full,
    /// Only the execution are logged.
    ExecutionOnly,
}

/// Aggregates the profiled launches by name, for the profiling logger's
/// summary.
#[derive(Debug, Default)]
pub struct ProfileAggregator {
    /// Each name's launches, written by name and read whole, slowest first,
    /// when the summary is shown.
    launches: HashMap<String, ProfiledLaunches>,
}

/// The launches of one name the profiling logger timed.
#[derive(Debug, Default, Clone, Copy)]
struct ProfiledLaunches {
    /// How long they took together.
    pub total_duration: Duration,
    /// How many there were.
    pub launch_count: usize,
}

impl ProfiledLaunches {
    /// These launches, and one more that took `duration`.
    fn adding_duration(self, duration: Duration) -> Self {
        Self {
            total_duration: self.total_duration + duration,
            launch_count: self.launch_count + 1,
        }
    }
}

impl ProfileAggregator {
    /// Whether nothing was profiled since it was last cleared.
    pub fn is_empty(&self) -> bool {
        self.launches.is_empty()
    }

    /// Adds a launch of `name`, which took `duration`. A name over several
    /// lines aggregates under its first.
    pub fn aggregate(&mut self, name: &str, duration: Duration) {
        let name = name.split_once('\n').map_or(name, |(first, _)| first);
        if let Some(launches) = self.launches.get_mut(name) {
            *launches = launches.adding_duration(duration);
            return;
        }
        let first = ProfiledLaunches::default().adding_duration(duration);
        self.launches.insert(name.to_string(), first);
    }

    /// Forgets every launch aggregated so far.
    pub fn clear(&mut self) {
        self.launches.clear();
    }
}

impl Display for ProfileAggregator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let header_name = "Name";
        let header_launch_count = "Num Computed";
        let header_duration = "Duration";
        let header_ratio = "Ratio";

        let mut ratio_len = header_ratio.len();
        let mut name_len = header_name.len();
        let mut launch_count_len = header_launch_count.len();
        let mut duration_len = header_duration.len();

        let mut total_duration = Duration::from_secs(0);
        let mut total_launch_count = 0;

        let mut rows: Vec<(String, String, String, Duration)> = self
            .launches
            .iter()
            .map(|(name, launches)| {
                let launch_count = format!("{}", launches.launch_count);
                let duration = format!("{:?}", launches.total_duration);

                name_len = usize::max(name_len, name.len());
                launch_count_len = usize::max(launch_count_len, launch_count.len());
                duration_len = usize::max(duration_len, duration.len());

                total_duration += launches.total_duration;
                total_launch_count += launches.launch_count;

                (
                    name.clone(),
                    launch_count,
                    duration,
                    launches.total_duration,
                )
            })
            .collect();

        let total_duration_fmt = format!("{total_duration:?}");
        let total_launch_count_fmt = format!("{total_launch_count:?}");
        let total_ratio_fmt = "100 %";

        duration_len = usize::max(duration_len, total_duration_fmt.len());
        launch_count_len = usize::max(launch_count_len, total_launch_count_fmt.len());
        ratio_len = usize::max(ratio_len, total_ratio_fmt.len());

        let line_length = name_len + duration_len + launch_count_len + ratio_len + 11;

        let write_line = |char: &str, f: &mut core::fmt::Formatter<'_>| {
            writeln!(f, "|{}| ", char.repeat(line_length))
        };
        rows.sort_by(|(_, _, _, a), (_, _, _, b)| b.cmp(a));

        write_line("⎺", f)?;

        writeln!(
            f,
            "| {header_name:<name_len$} | {header_duration:<duration_len$} | {header_launch_count:<launch_count_len$} | {header_ratio:<ratio_len$} |",
        )?;

        write_line("⎼", f)?;

        for (name, launch_count, duration, num) in rows {
            let ratio = (100 * num.as_micros()) / total_duration.as_micros().max(1);

            writeln!(
                f,
                "| {:<width_name$} | {:<width_duration$} | {:<width_launch_count$} | {:<width_ratio$} |",
                name,
                duration,
                launch_count,
                format!("{} %", ratio),
                width_name = name_len,
                width_duration = duration_len,
                width_launch_count = launch_count_len,
                width_ratio = ratio_len,
            )?;
        }

        write_line("⎼", f)?;

        writeln!(
            f,
            "| {:<width_name$} | {:<width_duration$} | {:<width_launch_count$} | {:<width_ratio$} |",
            "Total",
            total_duration_fmt,
            total_launch_count_fmt,
            total_ratio_fmt,
            width_name = name_len,
            width_duration = duration_len,
            width_launch_count = launch_count_len,
            width_ratio = ratio_len,
        )?;

        write_line("⎯", f)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_aggregates_its_launches() {
        let mut aggregator = ProfileAggregator::default();
        aggregator.aggregate("kernel", Duration::from_micros(3));
        aggregator.aggregate("kernel", Duration::from_micros(4));
        assert_eq!(
            aggregator.launches["kernel"].total_duration,
            Duration::from_micros(7)
        );
    }

    #[test]
    fn a_name_over_several_lines_aggregates_under_its_first() {
        let mut aggregator = ProfileAggregator::default();
        aggregator.aggregate("kernel\nwith its trace", Duration::from_micros(3));
        assert!(aggregator.launches.contains_key("kernel"));
    }

    #[test]
    fn a_cleared_aggregator_is_empty() {
        let mut aggregator = ProfileAggregator::default();
        aggregator.aggregate("kernel", Duration::from_micros(3));
        aggregator.clear();
        assert!(aggregator.is_empty());
    }

    #[test]
    fn one_more_launch_adds_its_duration_and_counts() {
        let launches = ProfiledLaunches::default().adding_duration(Duration::from_micros(3));
        assert_eq!(launches.launch_count, 1);
    }

    #[test]
    fn the_summary_names_every_launch_and_the_total() {
        let mut aggregator = ProfileAggregator::default();
        aggregator.aggregate("kernel", Duration::from_micros(3));
        let summary = aggregator.to_string();
        assert!(summary.contains("kernel") && summary.contains("Total"));
    }
}
