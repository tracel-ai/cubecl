//! The command line: what to read out of an environment ([`Inspection`]), how
//! a report is written ([`Output`]), and the help's look. The binary resolves
//! `--env` to a file and hands the [`Inspection`] an [`Inspector`]; a program
//! that finds its environments another way can do the same.

use crate::report::{
    AutotuneReport, CandidateReport, KernelOrder, KernelSelector, KeyId, KeyOrder,
};
use crate::view::Text;
use crate::{InspectError, Inspector, palette, wordmark};
use clap::builder::PossibleValue;
use clap::builder::styling::{Color, Style, Styles};
use clap::{Args, Subcommand, ValueEnum};
use serde::Serialize;
use std::fmt::Display;
use std::io::{IsTerminal as _, Write as _};
use std::path::PathBuf;

/// What to read out of one environment.
#[derive(Subcommand, Debug, Clone)]
pub enum Inspection {
    /// What the file holds, namespace by namespace, and what it says it was
    /// built for.
    Summary,
    /// Every tuned key: its winner, how many candidates raced, how close it
    /// was.
    Autotune(AutotuneArgs),
    /// Every kernel the sessions compiled or loaded, per type and per
    /// instance, and what that cost.
    Kernels(KernelArgs),
    /// The spans each session marked, in order, with the tuning and
    /// compiling inside each: where a session's time went, phase by phase.
    Timeline,
    /// Every memory snapshot the sessions recorded: each pool's pages against
    /// its peak, padding, and largest allocation.
    Memory,
    /// This environment's autotune answers against another's: keys added
    /// and gone, winners that changed, and how each tune's wall moved.
    Diff {
        /// The environment compared against, taken as the later one.
        other: PathBuf,
    },
    /// Drop the records of every session but the newest, in place.
    Prune {
        /// How many sessions to keep.
        #[arg(long, default_value_t = 1)]
        keep: usize,
    },
    /// Write a copy of the environment without its records, for
    /// distribution: the caches stay, the account of how they were built
    /// goes.
    Strip {
        /// Where the copy goes.
        out: PathBuf,
    },
}

/// What `autotune` reads: the keys, one key, or the candidates across them,
/// filtered and ordered as asked.
#[derive(Args, Debug, Clone)]
pub struct AutotuneArgs {
    #[command(subcommand)]
    pub view: Option<AutotuneView>,
    /// The order keys are listed in.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub sort: KeyOrder,
    /// Only the tuners whose name contains this.
    #[arg(long, global = true, value_name = "NAME")]
    pub tuner: Option<String>,
}

/// What `kernels` reads: every instance, or one, ordered as asked.
#[derive(Args, Debug, Clone)]
pub struct KernelArgs {
    #[command(subcommand)]
    pub view: Option<KernelView>,
    /// The order instances are listed in.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub sort: KernelOrder,
}

/// The views of `kernels` beside its list.
#[derive(Subcommand, Debug, Clone)]
pub enum KernelView {
    /// One instance: what it is, what it cost, and its source when recorded.
    Show {
        /// The instance's id as `kernels` lists it, or a prefix of it.
        id: String,
        /// The build that recorded it, as `kernels` lists it, or a prefix of
        /// it: needed only when several builds recorded the kernel.
        #[arg(long)]
        build: Option<String>,
    },
}

/// The views of `autotune` beside its list.
#[derive(Subcommand, Debug, Clone)]
pub enum AutotuneView {
    /// One key: every candidate's result, ranked, with the winner marked.
    Show {
        /// The key's id, as `autotune` lists it.
        id: KeyId,
    },
    /// Every candidate across the keys its tuner raced: how often it won and
    /// how far behind it typically ran — the table that says what to cut.
    Candidates,
}

/// How a report is written: text for a person, or JSON of the same report.
#[derive(Args, Debug, Clone, Copy, Default)]
pub struct Output {
    /// Write the report as JSON.
    #[arg(long, global = true)]
    json: bool,
}

impl Output {
    /// Write `report` to standard output, painted when that is a terminal
    /// and `NO_COLOR` is unset. A closed pipe — `| head` — ends the write
    /// quietly rather than as an error.
    pub fn print<R>(&self, report: &R) -> Result<(), InspectError>
    where
        R: Serialize,
        for<'a> Text<'a, R>: Display,
    {
        let rendered = if self.json {
            let mut json = serde_json::to_string_pretty(report)?;
            json.push('\n');
            json
        } else if paints() {
            format!("{:#}", Text(report))
        } else {
            Text(report).to_string()
        };
        match std::io::stdout().lock().write_all(rendered.as_bytes()) {
            Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
            written => Ok(written?),
        }
    }
}

/// The help's styles, in the `CubeCL` page's colours: headings in the
/// accent, what is typed in bold, what is filled in muted. The rest keeps the
/// terminal's own colour.
pub fn styles() -> Styles {
    let accent = Style::new()
        .fg_color(Some(Color::from(palette::ACCENT)))
        .bold();
    Styles::styled()
        .header(accent)
        .usage(accent)
        .literal(Style::new().bold())
        .placeholder(Style::new().fg_color(Some(Color::from(palette::MUTED))))
}

/// The `CubeCL` wordmark, the mark in the accent: what the help opens with.
/// The help strips it of its colour where it strips its own.
pub fn wordmark() -> String {
    let accent = Style::new().fg_color(Some(Color::from(palette::ACCENT)));
    wordmark::MARK
        .iter()
        .zip(wordmark::LETTERS)
        .map(|(mark, letters)| format!("{accent}{mark}{accent:#}{letters}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The orders as `--sort` spells them. Implemented here rather than derived
/// on the report, which stays a plain value no command line shapes.
impl ValueEnum for KeyOrder {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Wall, Self::Trials, Self::Margin, Self::Table]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Wall => PossibleValue::new("wall").help(
                "Slowest tune first; the keys whose tune was not recorded after them, most candidates measured first",
            ),
            Self::Trials => PossibleValue::new("trials").help("Most candidates measured first"),
            Self::Margin => PossibleValue::new("margin")
                .help("Widest margin first: the races whose answer was never in doubt"),
            Self::Table => PossibleValue::new("table").help("Grouped by tuner, as the file stores them"),
        })
    }
}

/// See the [`KeyOrder`] implementation.
impl ValueEnum for KernelOrder {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Compile, Self::Size, Self::Name]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Compile => PossibleValue::new("compile").help("Costliest to compile first"),
            Self::Size => PossibleValue::new("size").help("Largest artifact first"),
            Self::Name => PossibleValue::new("name").help("By kernel type"),
        })
    }
}

/// Whether text written to standard output is painted: on a terminal, unless
/// `NO_COLOR` asks otherwise.
fn paints() -> bool {
    std::io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty())
}

impl Inspection {
    /// Takes the inspector because [`Prune`](Self::Prune) closes it: its
    /// file cannot be written while it is open for reading.
    pub fn run(&self, inspector: Inspector, output: Output) -> Result<(), InspectError> {
        match self {
            Self::Summary => output.print(&inspector.summary()),
            Self::Autotune(args) => args.run(&inspector, output),
            Self::Kernels(args) => args.run(&inspector, output),
            Self::Timeline => output.print(&inspector.timeline()),
            Self::Memory => output.print(&inspector.memory()),
            Self::Diff { other } => output.print(&inspector.diff(&Inspector::open(other)?)),
            Self::Prune { keep } => output.print(&inspector.prune(*keep)?),
            Self::Strip { out } => output.print(&inspector.strip(out)?),
        }
    }
}

impl KernelArgs {
    fn run(&self, inspector: &Inspector, output: Output) -> Result<(), InspectError> {
        match &self.view {
            Some(KernelView::Show { id, build }) => {
                output.print(&inspector.kernel(KernelSelector {
                    id,
                    build: build.as_deref(),
                })?)
            }
            None => {
                let mut report = inspector.kernels();
                report.sort(self.sort);
                output.print(&report)
            }
        }
    }
}

impl AutotuneArgs {
    fn run(&self, inspector: &Inspector, output: Output) -> Result<(), InspectError> {
        match &self.view {
            Some(AutotuneView::Show { id }) => output.print(&inspector.autotune_key(*id)?),
            Some(AutotuneView::Candidates) => {
                output.print(&CandidateReport::from(&self.report(inspector)))
            }
            None => output.print(&self.report(inspector)),
        }
    }

    /// The file's keys, filtered and ordered as asked.
    fn report(&self, inspector: &Inspector) -> AutotuneReport {
        let mut report = inspector.autotune();
        if let Some(pattern) = &self.tuner {
            report.retain_tuners(pattern);
        }
        report.sort(self.sort);
        report
    }
}
