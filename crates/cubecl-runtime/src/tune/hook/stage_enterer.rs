use crate::tune::{TuneInputs, TuneStage};
use alloc::boxed::Box;
use core::any::Any;

/// Enters every stage of a set's tune, so a candidate that also launches somewhere the tuner does
/// not see, another device or another stream, can be put in step with the stage.
///
/// The tuner only says when a stage starts and ends: what the enterer does there is the set's
/// business, and its "after" is the `Drop` of the guard it returns. Stages nest: a gathering stage
/// may open inside a measuring one, and its guard drops first.
///
/// Candidates run outside every stage twice: as the call's work, once the tune decided or while it
/// is pending, and, under the `autotune-checks` feature, when every candidate is cross-checked.
///
/// # Assumptions
///
/// - Every stage is entered inside the hold the set's
///   [`TuneDeviceAccess`](crate::tune::TuneDeviceAccess) takes, on the thread that runs the
///   candidates: the calling thread under `Shared`, which never holds.
///
/// # Guarantees
///
/// - The guard `enter` hands back undoes exactly what `enter` did when it drops, so stages nest
///   in any combination.
/// - An implementation never waits on the tuned device's exclusive access.
/// - The guard's `Drop` compiles the kernels a gathering stage leaves queued on other devices, as
///   the tuner compiles the tuned device's.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid stage enterer",
    label = "invalid stage enterer"
)]
pub trait StageEnterer<K, I: TuneInputs>: Send + Sync + 'static {
    /// Enter `stage` of the tune of `key`, with the reference inputs the tune was called with.
    fn enter<'a>(&self, stage: TuneStage, key: &K, inputs: &I::At<'a>) -> Box<dyn Any>;
}

impl<K, Func, A, G> StageEnterer<K, A> for Func
where
    A: Clone + Send + Sync + 'static,
    K: 'static,
    G: 'static,
    Func: Send + Sync + 'static + Fn(TuneStage, &K, &A) -> G,
{
    #[inline]
    fn enter<'a>(
        &self,
        stage: TuneStage,
        key: &K,
        inputs: &<A as TuneInputs>::At<'a>,
    ) -> Box<dyn Any> {
        Box::new((self)(stage, key, inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use cubecl_environment::sync::Mutex;

    /// What a stage did, in order: entered, or ended.
    type StageLog = Arc<Mutex<Vec<(TuneStage, bool)>>>;

    /// Logs the end of its stage.
    struct StageCloser {
        log: StageLog,
        stage: TuneStage,
    }

    impl StageCloser {
        fn close(&self) {
            self.log.lock().push((self.stage, false));
        }
    }

    impl Drop for StageCloser {
        fn drop(&mut self) {
            self.close();
        }
    }

    /// An enterer that logs each stage it enters, and its end through the guard.
    fn logging(log: &StageLog) -> impl Fn(TuneStage, &String, &String) -> StageCloser + 'static {
        let log = log.clone();
        move |stage, _key, _inputs| {
            log.lock().push((stage, true));
            StageCloser {
                log: log.clone(),
                stage,
            }
        }
    }

    #[test]
    fn a_stage_ends_when_its_guard_drops() {
        let log = StageLog::default();
        let enterer = logging(&log);

        let guard = StageEnterer::<String, String>::enter(
            &enterer,
            TuneStage::Measuring,
            &String::new(),
            &String::new(),
        );
        core::mem::drop(guard);

        assert_eq!(
            *log.lock(),
            [(TuneStage::Measuring, true), (TuneStage::Measuring, false)]
        );
    }
}
