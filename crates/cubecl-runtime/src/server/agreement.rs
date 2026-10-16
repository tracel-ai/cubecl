use super::ServerError;
use alloc::{format, sync::Arc, vec::Vec};
use cubecl_environment::{backtrace::BackTrace, sync::Mutex};

/// Whether every member of one collective runs its part.
///
/// A part one device skips leaves the others waiting in it, and their next collective pairs with
/// that device's, so the members vote and either all run or all skip.
#[derive(Debug)]
pub struct Agreement {
    members: usize,
    tally: Arc<Tally>,
}

/// One member's say in an [`Agreement`]. Dropped without being cast, it refuses.
#[derive(Debug)]
pub struct Vote {
    member: usize,
    tally: Option<Arc<Tally>>,
}

#[derive(Debug)]
struct Tally {
    count: Mutex<Count>,
    #[cfg(feature = "std")]
    decided: std::sync::Condvar,
}

#[derive(Debug)]
struct Count {
    pending: usize,
    refusal: Option<Refusal>,
}

#[derive(Debug, Clone)]
struct Refusal {
    member: usize,
    error: ServerError,
}

impl Agreement {
    /// An agreement among `members`.
    pub fn new(members: usize) -> Self {
        Self {
            members,
            tally: Arc::new(Tally {
                count: Mutex::new(Count {
                    pending: members,
                    refusal: None,
                }),
                #[cfg(feature = "std")]
                decided: std::sync::Condvar::new(),
            }),
        }
    }

    /// One vote per member, member `i`'s at `i`.
    pub fn votes(self) -> Vec<Vote> {
        (0..self.members)
            .map(|member| Vote {
                member,
                tally: Some(self.tally.clone()),
            })
            .collect()
    }
}

impl Vote {
    /// The vote of a part whose peers queue theirs without agreeing, which decides alone.
    pub fn alone() -> Self {
        Self {
            member: 0,
            tally: None,
        }
    }

    /// Casts `verdict` and waits for every other member's, returning it when no member refused.
    ///
    /// # Errors
    ///
    /// This member's refusal, or else the first another member cast.
    pub fn cast<T>(mut self, verdict: Result<T, ServerError>) -> Result<T, ServerError> {
        let Some(tally) = self.tally.take() else {
            return verdict;
        };
        let refusal = verdict.as_ref().err().map(|error| Refusal {
            member: self.member,
            error: error.clone(),
        });
        let decided = tally.decide(refusal);
        match (verdict, decided) {
            (Ok(value), None) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Some(refusal)) => Err(ServerError::Generic {
                reason: format!(
                    "member {} refused the collective, so every member skipped it: {}",
                    refusal.member, refusal.error
                ),
                backtrace: BackTrace::capture(),
            }),
        }
    }
}

impl Drop for Vote {
    fn drop(&mut self) {
        if let Some(tally) = self.tally.take() {
            // Recorded without waiting: the thread dropping it may be the one that would flush the
            // others' parts.
            tally.record(Some(Refusal {
                member: self.member,
                error: ServerError::Generic {
                    reason: format!("member {}'s part never ran", self.member),
                    backtrace: BackTrace::capture(),
                },
            }));
        }
    }
}

impl Tally {
    /// Records one member's refusal, if any.
    fn record(&self, refusal: Option<Refusal>) {
        let mut count = self.count.lock();
        count.pending -= 1;
        if count.refusal.is_none() {
            count.refusal = refusal;
        }
        #[cfg(feature = "std")]
        if count.pending == 0 {
            self.decided.notify_all();
        }
    }

    /// Records one member's refusal, if any, then waits for every member and returns the first.
    fn decide(&self, refusal: Option<Refusal>) -> Option<Refusal> {
        self.record(refusal);
        #[cfg(feature = "std")]
        {
            let count = self
                .decided
                .wait_while(self.count.lock(), |count| count.pending > 0)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            count.refusal.clone()
        }
        #[cfg(not(feature = "std"))]
        loop {
            let count = self.count.lock();
            if count.pending == 0 {
                return count.refusal.clone();
            }
            drop(count);
            core::hint::spin_loop();
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec};

    fn refusal(reason: &str) -> ServerError {
        ServerError::Generic {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        }
    }

    fn cast_on_threads(verdicts: Vec<Result<(), ServerError>>) -> Vec<Result<(), ServerError>> {
        let votes = Agreement::new(verdicts.len()).votes();
        let casts: Vec<_> = votes
            .into_iter()
            .zip(verdicts)
            .map(|(vote, verdict)| std::thread::spawn(move || vote.cast(verdict)))
            .collect();
        casts
            .into_iter()
            .map(|cast| cast.join().unwrap())
            .collect()
    }

    #[test]
    fn members_that_all_accept_run() {
        let casts = cast_on_threads(vec![Ok(()), Ok(()), Ok(())]);

        assert!(casts.iter().all(Result::is_ok));
    }

    #[test]
    fn one_refusal_reaches_every_member() {
        let casts = cast_on_threads(vec![Ok(()), Err(refusal("tainted source")), Ok(())]);

        for cast in casts {
            let error = cast.unwrap_err().to_string();
            assert!(error.contains("tainted source"), "{error}");
        }
    }

    #[test]
    fn a_vote_dropped_uncast_refuses() {
        let mut votes = Agreement::new(2).votes();
        drop(votes.pop());

        let error = votes.pop().unwrap().cast(Ok(())).unwrap_err().to_string();

        assert!(error.contains("member 1's part never ran"), "{error}");
    }

    #[test]
    fn a_vote_alone_keeps_its_verdict() {
        assert!(Vote::alone().cast(Ok(())).is_ok());
        assert!(Vote::alone().cast::<()>(Err(refusal("alone"))).is_err());
    }
}
