//! What a launch is issued for: the work itself, or an autotune race measuring
//! or gathering candidates. Decided where the launch is issued and carried to
//! the thread that loads its kernel by its [`IssuedLaunch`].
//!
//! A kernel the work launched is one a later run of that work launches again,
//! whatever shape it was launched for; a kernel only a race launched is a
//! candidate the work may never pick. That is what keeps an environment's
//! prune from removing anything the work needs: the server records each
//! kernel the work launches, and nothing a race launches.

use super::{IssuedLaunch, with_value};

/// What a launch is issued for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchPurpose {
    /// The work: a call of an operation, its tuned pick included.
    Work,
    /// An autotune race: a candidate run to be measured, or to gather its
    /// kernels before it is.
    Race,
    /// The work, run by a candidate of a tune that has not decided its pick
    /// yet — its kernels only gathered, or its measurement still pending. The
    /// work's as far as what it launched goes, but a kernel of it that fails
    /// is the tune's to drop when it measures, not a failure of the work.
    Provisional,
}

impl LaunchPurpose {
    /// Whether a kernel launched for it is one a later run of the work
    /// launches again: the work's, or a candidate run in place of a pick not
    /// decided yet.
    pub fn is_works(self) -> bool {
        match self {
            Self::Work | Self::Provisional => true,
            Self::Race => false,
        }
    }

    /// The purpose of a kernel two launches asked for, one issued for
    /// `self` and one for `other`: the work's when either was, a race's only
    /// when both were.
    pub fn joined(self, other: Self) -> Self {
        match (self, other) {
            (Self::Work, _) | (_, Self::Work) => Self::Work,
            (Self::Provisional, _) | (_, Self::Provisional) => Self::Provisional,
            (Self::Race, Self::Race) => Self::Race,
        }
    }

    /// The purpose of the launch whose kernel this thread is loading. A kernel
    /// loaded outside a launch counts as the work's: keeping a kernel a race
    /// launched costs space, losing one the work launches costs a compile.
    pub fn running() -> Self {
        IssuedLaunch::running_purpose().unwrap_or(Self::Work)
    }
}

impl LaunchPurpose {
    /// The purpose of a launch issued on this thread now: a race inside
    /// [`race`], provisional inside [`run_provisionally`], the work otherwise.
    pub(crate) fn issued() -> Self {
        issuing::get().unwrap_or(Self::Work)
    }

    /// The purpose of a launch issued inside a run for `inner` opened while
    /// `self` was in force: a race's anywhere inside a race, provisional
    /// anywhere inside a provisional run otherwise.
    fn nested(self, inner: Self) -> Self {
        match (self, inner) {
            (Self::Race, _) | (_, Self::Race) => Self::Race,
            (Self::Provisional, _) | (_, Self::Provisional) => Self::Provisional,
            (Self::Work, Self::Work) => Self::Work,
        }
    }
}

/// Runs `candidate`, a race's run of one candidate: the launches it issues
/// on this thread are issued for the race. The thread is the one that
/// issues them, the device's own when the race runs there.
pub fn race<R>(candidate: impl FnOnce() -> R) -> R {
    issue_for(LaunchPurpose::Race, candidate)
}

/// Runs `candidate`, a candidate of a tune not decided yet run as the
/// work: the launches it issues on this thread are
/// [provisional](LaunchPurpose::Provisional), unless a race inside it issues them.
pub fn run_provisionally<R>(candidate: impl FnOnce() -> R) -> R {
    issue_for(LaunchPurpose::Provisional, candidate)
}

/// Runs `candidate` with the launches it issues on this thread issued for
/// `purpose`, nested in the purpose in force.
fn issue_for<R>(purpose: LaunchPurpose, candidate: impl FnOnce() -> R) -> R {
    with_value(
        issuing::replace,
        LaunchPurpose::issued().nested(purpose),
        candidate,
    )
}

/// The purpose of the launches this thread issues now, inside a race or a
/// provisional run.
#[cfg(feature = "std")]
mod issuing {
    use super::LaunchPurpose;
    use core::cell::Cell;

    std::thread_local! {
        static ISSUING: Cell<Option<LaunchPurpose>> = const { Cell::new(None) };
    }

    pub fn replace(purpose: Option<LaunchPurpose>) -> Option<LaunchPurpose> {
        ISSUING.with(|issuing| issuing.replace(purpose))
    }

    pub fn get() -> Option<LaunchPurpose> {
        ISSUING.with(Cell::get)
    }
}

/// Without std nothing records a work launch, so nothing tells a race's
/// launch from the work's: every launch is the work's, and a launch pays
/// nothing to say so.
#[cfg(not(feature = "std"))]
mod issuing {
    use super::LaunchPurpose;

    pub fn replace(_purpose: Option<LaunchPurpose>) -> Option<LaunchPurpose> {
        None
    }

    pub fn get() -> Option<LaunchPurpose> {
        None
    }
}

// Without std every launch is the work's: there is nothing to tell apart.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn a_launch_is_the_works_outside_a_race() {
        assert_eq!(LaunchPurpose::issued(), LaunchPurpose::Work);
    }

    #[test]
    fn a_launch_issued_by_a_race_is_the_races() {
        assert_eq!(race(LaunchPurpose::issued), LaunchPurpose::Race);
    }

    #[test]
    fn a_launch_issued_by_a_provisional_run_is_provisional() {
        assert_eq!(
            run_provisionally(LaunchPurpose::issued),
            LaunchPurpose::Provisional
        );
    }

    #[test]
    fn a_race_inside_a_provisional_run_is_still_a_race() {
        let nested = run_provisionally(|| race(LaunchPurpose::issued));
        assert_eq!(nested, LaunchPurpose::Race);
    }

    #[test]
    fn a_provisional_run_inside_a_race_is_still_a_race() {
        let nested = race(|| run_provisionally(LaunchPurpose::issued));
        assert_eq!(nested, LaunchPurpose::Race);
    }

    #[test]
    fn a_provisional_launch_is_the_works() {
        assert!(LaunchPurpose::Provisional.is_works());
    }

    #[test]
    fn a_race_launch_is_not_the_works() {
        assert!(!LaunchPurpose::Race.is_works());
    }

    #[test]
    fn a_kernel_two_launches_asked_for_is_the_works_when_either_was() {
        assert_eq!(
            LaunchPurpose::Race.joined(LaunchPurpose::Work),
            LaunchPurpose::Work
        );
    }

    #[test]
    fn a_kernel_two_races_asked_for_is_a_races() {
        assert_eq!(
            LaunchPurpose::Race.joined(LaunchPurpose::Race),
            LaunchPurpose::Race
        );
    }

    #[test]
    fn a_kernel_loaded_outside_a_launch_is_the_works() {
        assert_eq!(LaunchPurpose::running(), LaunchPurpose::Work);
    }
}
