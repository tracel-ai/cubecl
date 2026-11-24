use crate::tune::TuneInputs;
use alloc::boxed::Box;
use core::any::Any;

/// What a tune does with its candidates while a stage of it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneStage {
    /// Each candidate runs with its launches discarded on the tuned stream, so the kernels it
    /// reaches queue to compile together.
    Gathering,
    /// The candidates run for real on the tuned stream: their warm-ups and their timed samples.
    Measuring,
}

/// Enters every stage of a set's tune, so a candidate that also launches somewhere the tuner does
/// not see, another device or another stream, can be put in step with the stage.
///
/// The tuner only says when a stage starts and ends: what the enterer does there is the set's
/// business, and its "after" is the `Drop` of the guard it returns. Stages nest: a gathering stage
/// may open inside a measuring one, and its guard drops first.
///
/// Candidates run outside every stage twice: as the call's work, once the tune decided or while it
/// is pending, and, under the `autotune-checks` feature, when every candidate is cross-checked.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid stage enterer",
    label = "invalid stage enterer"
)]
pub trait StageEnterer<K, I: TuneInputs>: Send + Sync + 'static {
    /// Enter `stage` of the tune of `key`, with the reference inputs the tune was called with.
    ///
    /// Every implementation hands back a guard whose `Drop` undoes exactly what `enter` did, so
    /// stages nest in any combination. A measuring stage is entered where the set's
    /// [`TuneDeviceAccess`](super::TuneDeviceAccess) holds the tuned device, on the thread that
    /// runs the candidates: an implementation never waits on that device's exclusive access.
    /// Kernels a gathering stage leaves queued on other devices are compiled by the guard's
    /// `Drop`, as the tuner compiles the tuned device's.
    fn enter<'a>(&self, stage: TuneStage, key: &K, inputs: &I::At<'a>) -> Box<dyn Any>;
}

/// `Fn(TuneStage, &K, &A) -> G` acts as a [`StageEnterer`] when `A` is an owned type, `G` being the
/// guard dropped as the stage ends.
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

/// A [`StageEnterer`] bound to one key and the reference inputs of one tune: what the tuner enters
/// at each stage.
///
/// `FnMut` for the reason [`Evictor`](crate::tune::Evictor) is: it holds the inputs, which are only
/// `Send`, and is lent out exclusively to the device thread the stage runs on.
pub(crate) type BoundStageEnterer<'i> = dyn FnMut(TuneStage) -> Box<dyn Any> + Send + 'i;

/// The guard of `stage` entered by `enterer`, or none where the set registered no enterer.
pub(crate) fn enter_stage(
    enterer: Option<&mut BoundStageEnterer<'_>>,
    stage: TuneStage,
) -> Option<Box<dyn Any>> {
    enterer.map(|enter| enter(stage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{CloneInputGenerator, TunableSet};
    use alloc::string::{String, ToString};
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

    #[test]
    fn a_set_without_an_enterer_enters_no_stage() {
        assert!(enter_stage(None, TuneStage::Gathering).is_none());
    }

    #[test]
    fn a_bound_enterer_enters_the_stage_it_is_given() {
        let mut bound: Box<BoundStageEnterer<'_>> = Box::new(|stage| Box::new(stage));

        let guard = enter_stage(Some(&mut *bound), TuneStage::Gathering);

        assert_eq!(
            guard.and_then(|guard| guard.downcast::<TuneStage>().ok()),
            Some(Box::new(TuneStage::Gathering))
        );
    }

    #[test]
    fn a_set_binds_its_enterer_to_the_key_and_the_reference_inputs() {
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let recorded = seen.clone();
        let set = TunableSet::<String, String, ()>::new(
            |inputs: &String| inputs.clone(),
            CloneInputGenerator,
        )
        .with_stage_enterer(Arc::new(
            move |_stage: TuneStage, key: &String, inputs: &String| {
                recorded.lock().push((key.clone(), inputs.clone()));
            },
        ));

        let mut bound = set
            .bound_stage_enterer(&"key".to_string(), &"inputs".to_string())
            .expect("the set should hold an enterer");
        bound(TuneStage::Measuring);

        assert_eq!(*seen.lock(), [("key".to_string(), "inputs".to_string())]);
    }
}
