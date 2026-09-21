use super::table::{Align, Table};
use super::{Bytes, Maybe, Text, Wall};
use crate::report::{KernelReport, KernelRow};
use std::fmt;

impl fmt::Display for Text<'_, KernelReport> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.0;
        let mut families = Table::new(&[
            ("kernel", Align::Left),
            ("instances", Align::Right),
            ("compiled", Align::Right),
            ("loaded", Align::Right),
            ("compiling", Align::Right),
            ("loading", Align::Right),
            ("stored", Align::Right),
        ]);
        for family in &report.families {
            families.row(vec![
                family.short_name().to_string(),
                family.instances.to_string(),
                family.compiled.to_string(),
                family.loaded.to_string(),
                Wall(family.compiling).to_string(),
                Wall(family.loading).to_string(),
                Bytes(family.bytes).to_string(),
            ]);
        }
        write!(f, "{families}")?;

        let mut kernels = Table::new(&[
            ("id", Align::Left),
            ("build", Align::Left),
            ("kernel", Align::Left),
            ("compiled", Align::Right),
            ("loaded", Align::Right),
            ("compiling", Align::Right),
            ("stored", Align::Right),
        ]);
        for row in &report.kernels {
            kernels.row(vec![
                row.id.to_string(),
                row.build.to_string(),
                row.short_name().to_string(),
                row.compiled.to_string(),
                row.loaded.to_string(),
                Wall(row.compiling).to_string(),
                Maybe(row.bytes.map(Bytes)).to_string(),
            ]);
        }
        writeln!(f)?;
        write!(f, "{kernels}")?;
        write!(
            f,
            "{} kernels, compiled in {}",
            report.kernels.len(),
            Wall(report.compiling())
        )?;
        if report.unrecorded > 0 {
            write!(
                f,
                "; {} more stored by builds that recorded nothing",
                report.unrecorded
            )?;
        }
        writeln!(f)
    }
}

impl fmt::Display for Text<'_, KernelRow> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row = self.0;
        writeln!(f, "id         {}", row.id)?;
        writeln!(f, "build      {}", row.build)?;
        writeln!(f, "kernel     {}", row.kernel)?;
        if let Some(bytes) = row.bytes {
            writeln!(f, "stored     {}", Bytes(bytes))?;
        }
        writeln!(
            f,
            "compiled   {} times in {}, loaded {} times in {}",
            row.compiled,
            Wall(row.compiling),
            row.loaded,
            Wall(row.loading)
        )?;
        match &row.ir {
            Some(ir) => writeln!(f, "\nir\n{}", ir.trim_end())?,
            None => writeln!(
                f,
                "ir         not recorded (only a fresh compile defines the kernel)"
            )?,
        }
        match &row.source {
            Some(source) => writeln!(f, "\n{source}"),
            None => writeln!(
                f,
                "source     not recorded (`[environment.records] level = \"full\"` keeps it)"
            ),
        }
    }
}
