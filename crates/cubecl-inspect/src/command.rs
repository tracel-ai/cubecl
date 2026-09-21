//! The command line, shared by every front door: the `cubecl-inspect` binary
//! names the file with `--env <path>`, and an application that knows where its
//! environments live resolves the file its own way and hands the same
//! [`Inspection`] an [`Inspector`].

use crate::report::{AutotuneReport, CandidateReport, KernelOrder, KeyId, KeyOrder};
use crate::view::Text;
use crate::{InspectError, Inspector};
use clap::{Args, Subcommand};
use serde::Serialize;
use std::fmt::Display;
use std::io::Write as _;
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
    /// Every kernel the builds compiled or loaded, per type and per
    /// instance, and what that cost.
    Kernels(KernelArgs),
    /// The spans each session marked, in order, with the tuning and
    /// compiling inside each: where a build's time went, phase by phase.
    Timeline,
    /// Every memory snapshot the builds recorded: each pool's pages against
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
    /// Browse the reports in the terminal, read again whenever the file
    /// changes: open it beside a build to watch the build fill it.
    #[cfg(feature = "tui")]
    Tui,
    /// Write a copy of the environment without its records, for
    /// distribution: the caches stay, the account of how they were built
    /// goes.
    Strip {
        /// Where the copy goes.
        out: PathBuf,
    },
}

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

#[derive(Args, Debug, Clone)]
pub struct KernelArgs {
    #[command(subcommand)]
    pub view: Option<KernelView>,
    /// The order instances are listed in.
    #[arg(long, global = true, value_enum, default_value_t)]
    pub sort: KernelOrder,
}

#[derive(Subcommand, Debug, Clone)]
pub enum KernelView {
    /// One instance: what it is, what it cost, and its source when recorded.
    Show {
        /// The instance's id as `kernels` lists it, or a prefix of it.
        id: String,
    },
}

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
    /// Write `report` to standard output. A closed pipe — `| head` — ends the
    /// write quietly rather than as an error.
    pub fn print<R>(&self, report: &R) -> Result<(), InspectError>
    where
        R: Serialize,
        for<'a> Text<'a, R>: Display,
    {
        let rendered = if self.json {
            let mut json = serde_json::to_string_pretty(report)?;
            json.push('\n');
            json
        } else {
            Text(report).to_string()
        };
        match std::io::stdout().lock().write_all(rendered.as_bytes()) {
            Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
            written => Ok(written?),
        }
    }
}

impl Inspection {
    pub fn run(&self, inspector: &Inspector, output: Output) -> Result<(), InspectError> {
        match self {
            Self::Summary => output.print(&inspector.summary()),
            Self::Autotune(args) => args.run(inspector, output),
            Self::Kernels(args) => args.run(inspector, output),
            Self::Timeline => output.print(&inspector.timeline()),
            Self::Memory => output.print(&inspector.memory()),
            Self::Diff { other } => output.print(&inspector.diff(&Inspector::open(other)?)),
            Self::Prune { keep } => output.print(&inspector.prune(*keep)?),
            Self::Strip { out } => output.print(&inspector.strip(out)?),
            #[cfg(feature = "tui")]
            Self::Tui => crate::tui::run(inspector.path()),
        }
    }
}

impl KernelArgs {
    fn run(&self, inspector: &Inspector, output: Output) -> Result<(), InspectError> {
        match &self.view {
            Some(KernelView::Show { id }) => output.print(&inspector.kernel(id)?),
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
