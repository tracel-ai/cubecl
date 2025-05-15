use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Display;
use hashbrown::HashMap;

#[derive(Debug, Default)]
pub(crate) struct Profiled {
    durations: HashMap<String, ProfileItem>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct ProfileItem {
    total_duration: core::time::Duration,
    num_computed: usize,
}

#[derive(Debug, Copy, Clone)]
/// Control the amount of info being display when profiling.
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

impl Profiled {
    /// If some computation was profiled.
    pub fn is_empty(&self) -> bool {
        self.durations.is_empty()
    }
    pub fn update(&mut self, name: &String, duration: core::time::Duration) {
        let name = if name.contains("\n") {
            name.split("\n").next().unwrap()
        } else {
            name
        };
        if let Some(item) = self.durations.get_mut(name) {
            item.update(duration);
        } else {
            self.durations.insert(
                name.to_string(),
                ProfileItem {
                    total_duration: duration,
                    num_computed: 1,
                },
            );
        }
    }
}

impl Display for Profiled {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let header_name = "Name";
        let header_num_computed = "Num Computed";
        let header_duration = "Duration";
        let header_ratio = "Ratio";

        let mut ratio_len = header_ratio.len();
        let mut name_len = header_name.len();
        let mut num_computed_len = header_num_computed.len();
        let mut duration_len = header_duration.len();

        let mut total_duration = core::time::Duration::from_secs(0);
        let mut total_computed = 0;

        let mut items: Vec<(String, String, String, core::time::Duration)> = self
            .durations
            .iter()
            .map(|(key, item)| {
                let name = key.clone();
                let num_computed = format!("{}", item.num_computed);
                let duration = format!("{:?}", item.total_duration);

                name_len = usize::max(name_len, name.len());
                num_computed_len = usize::max(num_computed_len, num_computed.len());
                duration_len = usize::max(duration_len, duration.len());

                total_duration += item.total_duration;
                total_computed += item.num_computed;

                (name, num_computed, duration, item.total_duration)
            })
            .collect();

        let total_duration_fmt = format!("{total_duration:?}");
        let total_compute_fmt = format!("{total_computed:?}");
        let total_ratio_fmt = "100 %";

        duration_len = usize::max(duration_len, total_duration_fmt.len());
        num_computed_len = usize::max(num_computed_len, total_compute_fmt.len());
        ratio_len = usize::max(ratio_len, total_ratio_fmt.len());

        let line_length = name_len + duration_len + num_computed_len + ratio_len + 11;

        let write_line = |char: &str, f: &mut core::fmt::Formatter<'_>| {
            writeln!(f, "|{}| ", char.repeat(line_length))
        };
        items.sort_by(|(_, _, _, a), (_, _, _, b)| b.cmp(a));

        write_line("⎺", f)?;

        writeln!(
            f,
            "| {header_name:<name_len$} | {header_duration:<duration_len$} | {header_num_computed:<num_computed_len$} | {header_ratio:<ratio_len$} |",
        )?;

        write_line("⎼", f)?;

        for (name, num_computed, duration, num) in items {
            let ratio = (100 * num.as_micros()) / total_duration.as_micros();

            writeln!(
                f,
                "| {:<width_name$} | {:<width_duration$} | {:<width_num_computed$} | {:<width_ratio$} |",
                name,
                duration,
                num_computed,
                format!("{} %", ratio),
                width_name = name_len,
                width_duration = duration_len,
                width_num_computed = num_computed_len,
                width_ratio = ratio_len,
            )?;
        }

        write_line("⎼", f)?;

        writeln!(
            f,
            "| {:<width_name$} | {:<width_duration$} | {:<width_num_computed$} | {:<width_ratio$} |",
            "Total",
            total_duration_fmt,
            total_compute_fmt,
            total_ratio_fmt,
            width_name = name_len,
            width_duration = duration_len,
            width_num_computed = num_computed_len,
            width_ratio = ratio_len,
        )?;

        write_line("⎯", f)?;

        Ok(())
    }
}

impl ProfileItem {
    pub fn update(&mut self, duration: core::time::Duration) {
        self.total_duration += duration;
        self.num_computed += 1;
    }
}
