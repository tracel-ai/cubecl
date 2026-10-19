use super::paint::Role;
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
        ])
        .paint(&["compiling", "loading"], Role::Measure);
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
        fmt::Display::fmt(&families, f)?;

        let mut kernels = Table::new(&[
            ("id", Align::Left),
            ("build", Align::Left),
            ("kernel", Align::Left),
            ("compiled", Align::Right),
            ("loaded", Align::Right),
            ("compiling", Align::Right),
            ("stored", Align::Right),
        ])
        .paint(&["compiling"], Role::Measure);
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
        fmt::Display::fmt(&kernels, f)?;
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
        Role::Label.write(f, "id         ")?;
        writeln!(f, "{}", row.id)?;
        Role::Label.write(f, "build      ")?;
        writeln!(f, "{}", row.build)?;
        Role::Label.write(f, "kernel     ")?;
        writeln!(f, "{}", row.kernel)?;
        if let Some(bytes) = row.bytes {
            Role::Label.write(f, "stored     ")?;
            writeln!(f, "{}", Bytes(bytes))?;
        }
        Role::Label.write(f, "compiled   ")?;
        writeln!(
            f,
            "{} times in {}, loaded {} times in {}",
            row.compiled,
            Wall(row.compiling),
            row.loaded,
            Wall(row.loading)
        )?;
        match &row.ir {
            Some(ir) => {
                writeln!(f)?;
                Role::Title.write(f, "ir")?;
                writeln!(f, "\n{}", ir.trim_end())?;
            }
            None => {
                Role::Label.write(f, "ir         ")?;
                match row.compiled {
                    0 => writeln!(
                        f,
                        "not recorded (a kernel loaded from the store is not defined again)"
                    )?,
                    _ => writeln!(
                        f,
                        "not recorded (`[environment.records] level = \"full\"` keeps a fresh compile's)"
                    )?,
                }
            }
        }
        match &row.source {
            Some(source) => {
                writeln!(f)?;
                Role::Title.write(f, "source")?;
                writeln!(f, "\n{source}")
            }
            None => {
                Role::Label.write(f, "source     ")?;
                writeln!(
                    f,
                    "not recorded (`[environment.records] level = \"full\"` keeps it)"
                )
            }
        }
    }
}
