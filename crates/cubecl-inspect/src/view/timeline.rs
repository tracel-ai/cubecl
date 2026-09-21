use super::table::{Align, Table};
use super::{Bytes, KeyText, Maybe, Text, Wall};
use crate::report::{AutotuneTable, MemorySnapshots, Timeline};
use cubecl_server::memory_management::{MemoryPoolKind, MemoryPoolReport};
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
            writeln!(
                f,
                "session {} — {}",
                session.session.id,
                Maybe(session.session.label.as_ref())
            )?;
            let mut table = Table::new(&[
                ("at", Align::Right),
                ("span", Align::Left),
                ("wall", Align::Right),
                ("tunes", Align::Right),
                ("tuning", Align::Right),
                ("kernels", Align::Right),
                ("compiling", Align::Right),
                ("slowest tune", Align::Left),
            ]);
            for span in &session.spans {
                let slowest = span.slowest.as_ref().map(|slowest| {
                    let tuner = AutotuneTable::parse(&slowest.table)
                        .map_or_else(|| slowest.table.clone(), |table| table.tuner);
                    format!("{} {tuner} {}", Wall(slowest.wall), KeyText(&slowest.key))
                });
                table.row(vec![
                    format!("+{:.1} s", span.stamp.offset.as_secs_f64()),
                    format!("{}{}", "  ".repeat(span.depth), span.label),
                    Wall(span.wall).to_string(),
                    span.tunes.to_string(),
                    Wall(span.tuning).to_string(),
                    span.compilations.to_string(),
                    Wall(span.compiling).to_string(),
                    Maybe(slowest).to_string(),
                ]);
            }
            write!(f, "{table}")?;
            writeln!(f)?;
        }
        writeln!(
            f,
            "a span counts what started inside it, nested spans included; compiling inside a tune is also tuning."
        )
    }
}

impl fmt::Display for Text<'_, MemorySnapshots> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.snapshots.is_empty() {
            return writeln!(f, "no memory snapshots recorded");
        }
        for snapshot in &self.0.snapshots {
            writeln!(
                f,
                "{} — session {} at +{:.1} s",
                snapshot.record.label,
                snapshot.stamp.session,
                snapshot.stamp.offset.as_secs_f64()
            )?;
            let mut table = Table::new(&[
                ("pool", Align::Left),
                ("kind", Align::Left),
                ("pages", Align::Right),
                ("peak", Align::Right),
                ("unmapped", Align::Right),
                ("in use", Align::Right),
                ("reserved", Align::Right),
                ("padding", Align::Right),
                ("largest", Align::Right),
            ]);
            let report = &snapshot.record.report;
            let pools = report
                .dynamic
                .iter()
                .enumerate()
                .map(|(index, pool)| (format!("dynamic {index}"), pool))
                .chain([("persistent".to_string(), &report.persistent)]);
            for (name, pool) in pools {
                table.row(pool_row(name, pool));
            }
            write!(f, "{table}")?;
            writeln!(f)?;
        }
        writeln!(
            f,
            "peak: the most pages held at once; unmapped: carved under a dry run, never backed."
        )
    }
}

fn pool_row(name: String, pool: &MemoryPoolReport) -> Vec<String> {
    let usage = &pool.usage;
    let padding = (usage.bytes_in_use > 0)
        .then(|| usage.bytes_padding as f64 / usage.bytes_in_use as f64 * 100.0);
    vec![
        name,
        PoolKind(&pool.kind).to_string(),
        pool.pages.to_string(),
        pool.pages_peak.to_string(),
        pool.pages_unmapped.to_string(),
        Bytes(usage.bytes_in_use).to_string(),
        Bytes(usage.bytes_reserved).to_string(),
        Maybe(padding.map(|percent| format!("{percent:.1}%"))).to_string(),
        Bytes(pool.largest_alloc).to_string(),
    ]
}

/// A pool's shape, briefly.
struct PoolKind<'a>(&'a MemoryPoolKind);

impl fmt::Display for PoolKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            MemoryPoolKind::Sliced {
                page_size,
                max_pool_size,
                ..
            } => {
                write!(f, "sliced, {} pages", Bytes(*page_size))?;
                match max_pool_size {
                    Some(cap) => write!(f, ", capped at {}", Bytes(*cap)),
                    None => write!(f, ", growable"),
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
