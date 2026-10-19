use super::paint::Role;
use super::table::{Align, Table};
use super::{Bytes, Maybe, Text, Wall};
use crate::report::{Listing, Summary};
use cubecl_environment::bundle::BundleManifest;
use std::fmt;

impl fmt::Display for Text<'_, Summary> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let summary = self.0;
        if let Some(name) = summary.name() {
            Role::Label.write(f, "name         ")?;
            writeln!(f, "{name}")?;
        }
        Role::Label.write(f, "environment  ")?;
        writeln!(
            f,
            "{} ({})",
            summary.path.display(),
            Bytes(summary.file_bytes)
        )?;
        Role::Label.write(f, "built        ")?;
        match &summary.manifest {
            Some(manifest) => writeln!(f, "{}", Built(manifest))?,
            None => writeln!(f, "- (a live environment carries no manifest)")?,
        }
        Role::Label.write(f, "entries      ")?;
        writeln!(
            f,
            "{} in {} namespaces, {} autotune keys",
            summary.entries(),
            summary.namespaces.len(),
            summary.autotune_keys()
        )?;

        if !summary.sessions.is_empty() {
            let mut sessions = Table::new(&[
                ("session", Align::Left),
                ("started", Align::Left),
                ("label", Align::Left),
                ("tunes", Align::Right),
                ("tuning", Align::Right),
                ("kernels", Align::Right),
                ("compiling", Align::Right),
                ("other", Align::Right),
                ("span", Align::Right),
            ])
            .paint(&["tuning", "compiling", "other", "span"], Role::Measure);
            for row in &summary.sessions {
                sessions.row(vec![
                    row.session.id.to_string(),
                    UnixTime(row.session.started_unix_ms / 1000).to_string(),
                    Maybe(row.session.label.as_ref()).to_string(),
                    row.tunes.to_string(),
                    Wall(row.tuning).to_string(),
                    row.compilations.to_string(),
                    Wall(row.compiling).to_string(),
                    Wall(row.other()).to_string(),
                    Wall(row.span).to_string(),
                ]);
            }
            writeln!(f)?;
            fmt::Display::fmt(&sessions, f)?;
            Role::Note.write(
                f,
                "compiling includes what ran inside tunes; other is the span less both.",
            )?;
            writeln!(f)?;
        }

        let mut roots = Table::new(&[
            ("root", Align::Left),
            ("entries", Align::Right),
            ("bytes", Align::Right),
        ]);
        for root in summary.roots() {
            roots.row(vec![
                root.namespace,
                root.entries.to_string(),
                Bytes(root.bytes).to_string(),
            ]);
        }
        writeln!(f)?;
        fmt::Display::fmt(&roots, f)?;

        let mut namespaces = Table::new(&[
            ("namespace", Align::Left),
            ("entries", Align::Right),
            ("bytes", Align::Right),
        ]);
        for namespace in &summary.namespaces {
            namespaces.row(vec![
                namespace.namespace.clone(),
                namespace.entries.to_string(),
                Bytes(namespace.bytes).to_string(),
            ]);
        }
        writeln!(f)?;
        fmt::Display::fmt(&namespaces, f)
    }
}

impl fmt::Display for Text<'_, Listing> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let listing = self.0;
        if listing.environments.is_empty() && listing.unreadable.is_empty() {
            let directories: Vec<String> = listing
                .directories
                .iter()
                .map(|directory| directory.display().to_string())
                .collect();
            return writeln!(f, "no environment in {}", directories.join(", "));
        }
        fmt::Display::fmt(&Table::from(listing.environments.as_slice()), f)?;
        for unreadable in &listing.unreadable {
            writeln!(
                f,
                "unreadable: {} ({})",
                unreadable.path.display(),
                unreadable.reason
            )?;
        }
        Ok(())
    }
}

/// The listing's table: one row per environment, in the order given.
impl From<&[Summary]> for Table {
    fn from(environments: &[Summary]) -> Self {
        let mut table = Table::new(&[
            ("environment", Align::Left),
            ("built", Align::Left),
            ("cubecl", Align::Left),
            ("devices", Align::Left),
            ("keys", Align::Right),
            ("entries", Align::Right),
            ("size", Align::Right),
            ("name", Align::Left),
        ]);
        for summary in environments {
            let manifest = summary.manifest.as_ref();
            table.row(vec![
                summary.stem(),
                Maybe(
                    manifest
                        .and_then(|manifest| manifest.created_unix_secs)
                        .map(UnixTime),
                )
                .to_string(),
                Maybe(manifest.map(|manifest| &manifest.cubecl_version)).to_string(),
                manifest
                    .map(devices)
                    .filter(|devices| !devices.is_empty())
                    .unwrap_or_else(|| "-".to_string()),
                summary.autotune_keys().to_string(),
                summary.entries().to_string(),
                Bytes(summary.file_bytes).to_string(),
                Maybe(summary.name()).to_string(),
            ]);
        }
        table
    }
}

/// What a manifest says of the build: when, by which cubecl, for what.
struct Built<'a>(&'a BundleManifest);

impl fmt::Display for Built<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let manifest = self.0;
        match manifest.created_unix_secs {
            Some(secs) => fmt::Display::fmt(&UnixTime(secs), f)?,
            None => f.write_str("-")?,
        }
        write!(f, " by cubecl {}", manifest.cubecl_version)?;
        for environment in &manifest.environments {
            write!(f, "; ")?;
            if !environment.label.is_empty() {
                write!(f, "{} ", environment.label)?;
            }
            write!(f, "on {}/{}", environment.os, environment.arch)?;
            if !environment.devices.is_empty() {
                write!(f, " ({})", environment.devices.join(", "))?;
            }
        }
        Ok(())
    }
}

fn devices(manifest: &BundleManifest) -> String {
    manifest
        .environments
        .iter()
        .flat_map(|environment| environment.devices.iter().cloned())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Seconds since the Unix epoch, as a UTC date and time to the minute.
struct UnixTime(u64);

impl fmt::Display for UnixTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let days = (self.0 / 86_400) as i64;
        let minutes = (self.0 % 86_400) / 60;
        // Howard Hinnant's `civil_from_days`: the proleptic Gregorian date of
        // a day count, exact for every date a file can carry.
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let day_of_era = z.rem_euclid(146_097);
        let year_of_era =
            (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
        let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
        let shifted_month = (5 * day_of_year + 2) / 153;
        let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
        let month = if shifted_month < 10 {
            shifted_month + 3
        } else {
            shifted_month - 9
        };
        let year = year_of_era + era * 400 + i64::from(month <= 2);
        write!(
            f,
            "{year:04}-{month:02}-{day:02} {:02}:{:02} UTC",
            minutes / 60,
            minutes % 60
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_time_reads_as_a_utc_date() {
        assert_eq!(UnixTime(0).to_string(), "1970-01-01 00:00 UTC");
        assert_eq!(UnixTime(951_782_400).to_string(), "2000-02-29 00:00 UTC");
        assert_eq!(UnixTime(1_789_488_372).to_string(), "2026-09-15 16:06 UTC");
    }
}
