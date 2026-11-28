//! The kinds of work a collector counts, and how each can end.

/// How a registered kernel ended.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOutcome {
    /// Compiled by the backend's compiler.
    Compiled,
    /// Read from the compilation store.
    Loaded,
    /// Failed to compile.
    Failed,
    /// Dropped before it settled.
    Abandoned,
}

/// What the server did with a settled kernel: the device refused it, or the
/// compilation store kept it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelHandling {
    /// The device refused it.
    Refused,
    /// The compilation store kept it.
    Stored,
}

/// How a registered autotune key ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneOutcome {
    /// Tuned by measuring its candidates.
    Measured,
    /// Loaded from the persistent autotune cache instead.
    Loaded,
    /// Every candidate failed: the pick was made unmeasured.
    Failed,
    /// Dropped before it settled.
    Abandoned,
}

/// What became of a settled key's pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneHandling {
    /// The persistent autotune cache kept it.
    Persisted,
}

/// A kind of work a collector counts.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountedWork {
    /// Kernels obtained.
    Kernel,
    /// Autotune keys tuned.
    Tune,
}

/// A kind of work a collector can count: how one item of it can end, and
/// what can happen to it after.
///
/// Sealed: it is `pub` only so the public
/// [`Registration`](super::Registration) can name it, in a module nothing
/// outside the crate reaches, so no kind of work is added from outside.
#[doc(hidden)]
pub trait Countable: Copy + core::fmt::Debug + 'static {
    /// What can happen to an item once it settled.
    type Later: Copy;
    /// The work it counts.
    const WORK: CountedWork;
    /// How many outcomes it has, which a collector's tally has room for.
    const OUTCOME_COUNT: usize;
    /// How many things can happen to it once settled.
    const LATER_COUNT: usize;
    /// The outcome of an item dropped before it settled.
    const ABANDONED: Self;

    /// Its slot in a tally's outcomes, below
    /// [`OUTCOME_COUNT`](Self::OUTCOME_COUNT).
    fn index(self) -> usize;
    /// `later`'s slot in a tally's afterwards, below
    /// [`LATER_COUNT`](Self::LATER_COUNT).
    fn later_index(later: Self::Later) -> usize;
}

impl Countable for KernelOutcome {
    type Later = KernelHandling;
    const WORK: CountedWork = CountedWork::Kernel;
    const OUTCOME_COUNT: usize = 4;
    const LATER_COUNT: usize = 2;
    const ABANDONED: Self = Self::Abandoned;

    fn index(self) -> usize {
        match self {
            Self::Compiled => 0,
            Self::Loaded => 1,
            Self::Failed => 2,
            Self::Abandoned => 3,
        }
    }

    fn later_index(later: KernelHandling) -> usize {
        match later {
            KernelHandling::Refused => 0,
            KernelHandling::Stored => 1,
        }
    }
}

impl Countable for TuneOutcome {
    type Later = TuneHandling;
    const WORK: CountedWork = CountedWork::Tune;
    const OUTCOME_COUNT: usize = 4;
    const LATER_COUNT: usize = 1;
    const ABANDONED: Self = Self::Abandoned;

    fn index(self) -> usize {
        match self {
            Self::Measured => 0,
            Self::Failed => 1,
            Self::Abandoned => 2,
            Self::Loaded => 3,
        }
    }

    fn later_index(later: TuneHandling) -> usize {
        match later {
            TuneHandling::Persisted => 0,
        }
    }
}
