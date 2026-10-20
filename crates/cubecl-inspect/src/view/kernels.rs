use super::fields::Fields;
use super::paint::Role;
use super::table::{Align, Table};
use super::{ByteSize, Maybe, Text, Wall};
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
                family.cost.compiled.to_string(),
                family.cost.loaded.to_string(),
                Wall(family.cost.compiling).to_string(),
                Wall(family.cost.loading).to_string(),
                ByteSize(family.bytes).to_string(),
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
                row.cost.compiled.to_string(),
                row.cost.loaded.to_string(),
                Wall(row.cost.compiling).to_string(),
                Maybe(row.bytes.map(ByteSize)).to_string(),
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
                "; {} more stored while nothing was recorded",
                report.unrecorded
            )?;
        }
        writeln!(f)
    }
}

impl fmt::Display for Text<'_, KernelRow> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row = self.0;
        let fields = Fields::new(&[
            "id", "build", "kernel", "stored", "compiled", "ir", "source",
        ]);
        fields.line(f, "id", row.id)?;
        fields.line(f, "build", row.build)?;
        fields.line(f, "kernel", &row.kernel)?;
        if let Some(bytes) = row.bytes {
            fields.line(f, "stored", ByteSize(bytes))?;
        }
        fields.line(
            f,
            "compiled",
            format_args!(
                "{} times in {}, loaded {} times in {}",
                row.cost.compiled,
                Wall(row.cost.compiling),
                row.cost.loaded,
                Wall(row.cost.loading)
            ),
        )?;
        match (&row.ir, row.cost.compiled) {
            (Some(ir), _) => {
                writeln!(f)?;
                Role::Title.write(f, "ir")?;
                writeln!(f, "\n{}", ir.trim_end())?;
            }
            (None, 0) => fields.line(
                f,
                "ir",
                "not recorded (a kernel loaded from the store is not defined again)",
            )?,
            (None, _) => fields.line(
                f,
                "ir",
                "not recorded (`[environment.records] level = \"full\"` keeps a fresh compile's)",
            )?,
        }
        match &row.source {
            Some(source) => {
                writeln!(f)?;
                Role::Title.write(f, "source")?;
                writeln!(f, "\n{source}")
            }
            None => fields.line(
                f,
                "source",
                "not recorded (`[environment.records] level = \"full\"` keeps it)",
            ),
        }
    }
}
