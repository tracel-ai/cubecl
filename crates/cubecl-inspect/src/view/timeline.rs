use super::paint::Role;
use super::table::{Align, Table};
use super::{KeyText, Maybe, Text, Wall};
use crate::report::{AutotuneTable, Timeline};
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
                let slowest = span.activity.slowest.as_ref().map(|slowest| {
                    let tuner = AutotuneTable::new(&slowest.table)
                        .map_or_else(|| slowest.table.clone(), |table| table.tuner);
                    format!("{} {tuner} {}", Wall(slowest.wall), KeyText(&slowest.key))
                });
                let role = match (span.depth, span.activity.tunes + span.activity.compilations) {
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
                        span.activity.tunes.to_string(),
                        Wall(span.activity.tuning).to_string(),
                        span.activity.compilations.to_string(),
                        Wall(span.activity.compiling).to_string(),
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
