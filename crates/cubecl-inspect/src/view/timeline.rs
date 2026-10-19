use super::paint::Role;
use super::table::{Align, Table};
use super::{Bytes, KeyText, Maybe, Text, Wall};
use crate::report::{AutotuneTable, MemorySnapshots, Timeline};
use cubecl_server::memory_management::{
    MemoryPoolKind, MemoryPoolReport, MemoryPoolsReport, MemoryUsage,
};
use std::fmt;

impl fmt::Display for Text<'_, Timeline> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let timeline = self.0;
        if timeline
            .sessions
            .iter()
            .all(|session| session.spans.is_empty())
        {
            return writeln!(f, "no marks recorded");
        }
        for session in timeline.sessions.iter().filter(|s| !s.spans.is_empty()) {
            match &session.session.label {
                Some(label) => Role::Title
                    .write(f, format_args!("session {} — {label}", session.session.id))?,
                None => Role::Title.write(f, format_args!("session {}", session.session.id))?,
            }
            writeln!(f)?;
            let mut table = Table::new(&[
                ("at", Align::Right),
                ("span", Align::Left),
                ("wall", Align::Right),
                ("tunes", Align::Right),
                ("tuning", Align::Right),
                ("kernels", Align::Right),
                ("compiling", Align::Right),
                ("slowest tune", Align::Left),
            ])
            .paint(&["wall", "tuning", "compiling"], Role::Measure);
            for span in &session.spans {
                let slowest = span.slowest.as_ref().map(|slowest| {
                    let tuner = AutotuneTable::parse(&slowest.table)
                        .map_or_else(|| slowest.table.clone(), |table| table.tuner);
                    format!("{} {tuner} {}", Wall(slowest.wall), KeyText(&slowest.key))
                });
                let role = match (span.depth, span.tunes + span.compilations) {
                    (0, _) => Some(Role::Strong),
                    (_, 0) => Some(Role::Idle),
                    _ => None,
                };
                table.row_in(
                    role,
                    vec![
                        format!("+{:.1} s", span.stamp.offset.as_secs_f64()),
                        format!("{}{}", "  ".repeat(span.depth), span.label),
                        Wall(span.wall).to_string(),
                        span.tunes.to_string(),
                        Wall(span.tuning).to_string(),
                        span.compilations.to_string(),
                        Wall(span.compiling).to_string(),
                        Maybe(slowest).to_string(),
                    ],
                );
            }
            fmt::Display::fmt(&table, f)?;
            writeln!(f)?;
        }
        Role::Note.write(f, "a span counts what started inside it, nested spans included; compiling inside a tune is also tuning;\n\
             a span that tuned and compiled nothing is muted.")?;
        writeln!(f)
    }
}

impl fmt::Display for Text<'_, MemorySnapshots> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.snapshots.is_empty() {
            return writeln!(f, "no memory snapshots recorded");
        }
        for snapshot in &self.0.snapshots {
            Role::Title.write(
                f,
                format_args!(
                    "{} — session {} at +{:.1} s",
                    snapshot.record.label,
                    snapshot.stamp.session,
                    snapshot.stamp.offset.as_secs_f64()
                ),
            )?;
            writeln!(f)?;
            let mut table = Table::new(&[
                ("pool", Align::Left),
                ("kind", Align::Left),
                ("pages", Align::Right),
                ("peak", Align::Right),
                ("unmapped", Align::Right),
                ("in use", Align::Right),
                ("reserved", Align::Right),
                ("used", Align::Right),
                ("padding", Align::Right),
                ("largest", Align::Right),
            ]);
            let mut total = MemoryUsage::default();
            let mut add = |name: String, pool: &MemoryPoolReport| {
                total = total.combine(pool.usage.clone());
                let role = (pool.usage.bytes_reserved == 0).then_some(Role::Idle);
                table.row_in(role, pool_row(name, pool));
            };
            for stream in &snapshot.record.report.streams {
                let prefix = format!("stream {}", stream.stream.value);
                for (name, pool) in named_pools(&prefix, &stream.pools) {
                    add(name, pool);
                }
                for memory in &stream.auxiliary {
                    let prefix = format!("{prefix} {}", memory.name);
                    for (name, pool) in named_pools(&prefix, &memory.pools) {
                        add(name, pool);
                    }
                }
            }
            table.row_in(Some(Role::Strong), total_row(&total));
            fmt::Display::fmt(&table, f)?;
            writeln!(f)?;
        }
        Role::Note.write(f, "peak: the most pages held at once; unmapped: carved while launches were discarded, never backed;\n\
             used: in use over reserved, the share of the reservation holding live bytes; a pool with nothing reserved is muted.")?;
        writeln!(f)
    }
}

/// Every pool of `pools`, named after `prefix` and its role.
fn named_pools<'a>(
    prefix: &'a str,
    pools: &'a MemoryPoolsReport,
) -> impl Iterator<Item = (String, &'a MemoryPoolReport)> + 'a {
    pools
        .dynamic
        .iter()
        .enumerate()
        .map(move |(index, pool)| (format!("{prefix} dynamic {index}"), pool))
        .chain([
            (format!("{prefix} persistent"), &pools.persistent),
            (format!("{prefix} dedicated"), &pools.dedicated),
        ])
}

fn pool_row(name: String, pool: &MemoryPoolReport) -> Vec<String> {
    let usage = &pool.usage;
    vec![
        name,
        PoolKind(&pool.kind).to_string(),
        pool.pages.to_string(),
        pool.pages_peak.to_string(),
        pool.pages_unmapped.to_string(),
        Bytes(usage.bytes_in_use).to_string(),
        Bytes(usage.bytes_reserved).to_string(),
        Share::of(usage.bytes_in_use, usage.bytes_reserved).to_string(),
        Share::of(usage.bytes_padding, usage.bytes_in_use).to_string(),
        Bytes(pool.largest_alloc).to_string(),
    ]
}

/// Every pool of a snapshot summed: the pages are left out, since each pool
/// counts its own size of page and peaks at its own moment.
fn total_row(usage: &MemoryUsage) -> Vec<String> {
    let blank = || "".to_string();
    vec![
        "total".to_string(),
        blank(),
        blank(),
        blank(),
        blank(),
        Bytes(usage.bytes_in_use).to_string(),
        Bytes(usage.bytes_reserved).to_string(),
        Share::of(usage.bytes_in_use, usage.bytes_reserved).to_string(),
        Share::of(usage.bytes_padding, usage.bytes_in_use).to_string(),
        blank(),
    ]
}

/// A part over a whole as a percentage, `-` when the whole is empty.
struct Share(Option<f64>);

impl Share {
    fn of(part: u64, whole: u64) -> Self {
        Self((whole > 0).then(|| part as f64 / whole as f64 * 100.0))
    }
}

impl fmt::Display for Share {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Maybe(self.0.map(|percent| format!("{percent:.1}%"))).fmt(f)
    }
}

/// A pool's shape, briefly.
struct PoolKind<'a>(&'a MemoryPoolKind);

impl fmt::Display for PoolKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            MemoryPoolKind::Sliced {
                page_size,
                max_slice_size,
            } => write!(
                f,
                "sliced, {} pages, slices up to {}",
                Bytes(*page_size),
                Bytes(*max_slice_size)
            ),
            MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            } => {
                write!(f, "adaptive, {} pages", Bytes(*page_size))?;
                match outdated_pages {
                    0 => Ok(()),
                    outdated => write!(f, ", {outdated} outdated"),
                }
            }
            MemoryPoolKind::Exclusive { max_alloc_size } => {
                write!(f, "exclusive, up to {}", Bytes(*max_alloc_size))
            }
            MemoryPoolKind::Direct => f.write_str("direct"),
            MemoryPoolKind::Persistent => f.write_str("persistent"),
        }
    }
}
