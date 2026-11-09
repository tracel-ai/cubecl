use super::paint::Role;
use super::table::{Align, Table};
use super::{ByteSize, Percent, Text};
use crate::report::MemorySnapshots;
use cubecl_server::memory_management::{
    MemoryPoolKind, MemoryPoolReport, MemoryPoolsReport, MemoryUsage,
};
use std::fmt;

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
        ByteSize(usage.bytes_in_use).to_string(),
        ByteSize(usage.bytes_reserved).to_string(),
        format!(
            "{:.1}",
            Percent::new(usage.bytes_in_use, usage.bytes_reserved)
        ),
        format!(
            "{:.1}",
            Percent::new(usage.bytes_padding, usage.bytes_in_use)
        ),
        ByteSize(pool.largest_alloc).to_string(),
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
        ByteSize(usage.bytes_in_use).to_string(),
        ByteSize(usage.bytes_reserved).to_string(),
        format!(
            "{:.1}",
            Percent::new(usage.bytes_in_use, usage.bytes_reserved)
        ),
        format!(
            "{:.1}",
            Percent::new(usage.bytes_padding, usage.bytes_in_use)
        ),
        blank(),
    ]
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
                ByteSize(*page_size),
                ByteSize(*max_slice_size)
            ),
            MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            } => {
                write!(f, "adaptive, {} pages", ByteSize(*page_size))?;
                match outdated_pages {
                    0 => Ok(()),
                    outdated => write!(f, ", {outdated} outdated"),
                }
            }
            MemoryPoolKind::Exclusive { max_alloc_size } => {
                write!(f, "exclusive, up to {}", ByteSize(*max_alloc_size))
            }
            MemoryPoolKind::Direct => f.write_str("direct"),
            MemoryPoolKind::Persistent => f.write_str("persistent"),
        }
    }
}
