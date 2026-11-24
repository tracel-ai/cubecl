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

/// Runs around every stage of a set's tune, so a candidate that also launches somewhere the tuner
/// does not see, another device or another stream, can be put in step with the stage.
///
/// The tuner only says when a stage starts and ends: what the hook does there is the set's
/// business. The guard [`enter`](Self::enter) returns is dropped as the stage ends, so a hook does
/// its "after" in the guard's `Drop`. Stages nest: a gathering stage may open inside a measuring
/// one, and its guard drops first.
///
/// A candidate run as the call's work, once the tune decided or while it is pending, runs in no
/// stage, and the hook is not entered for it.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid stage hook",
    label = "invalid stage hook"
)]
pub trait StageHook<K, I: TuneInputs>: Send + Sync + 'static {
    /// Enter `stage` of the tune of `key`, called with the reference inputs the tune was called
    /// with. What it returns is dropped as the stage ends.
    fn enter<'a>(&self, stage: TuneStage, key: &K, inputs: &I::At<'a>) -> Box<dyn Any>;
}

/// `Fn(TuneStage, &K, &A) -> G` acts as a [`StageHook`] when `A` is an owned type, `G` being the
/// guard dropped as the stage ends.
impl<K, Func, A, G> StageHook<K, A> for Func
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

/// A [`StageHook`] bound to one key and the reference inputs of one tune: what the tuner enters at
/// each stage.
///
/// `FnMut` for the reason [`Evictor`](crate::tune::Evictor) is: it holds the inputs, which are only
/// `Send`, and is lent out exclusively to the device thread the stage runs on.
pub(crate) type StageEnterer<'i> = dyn FnMut(TuneStage) -> Box<dyn Any> + Send + 'i;

/// The guard of `stage` entered by `enterer`, or none where the set registered no hook.
pub(crate) fn enter_stage(
    enterer: Option<&mut StageEnterer<'_>>,
    stage: TuneStage,
) -> Option<Box<dyn Any>> {
    enterer.map(|enter| enter(stage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use cubecl_environment::sync::Mutex;

    /// Records each stage as it enters and ends.
    struct Events(Arc<Mutex<Vec<(TuneStage, bool)>>>);

    struct Guard(Arc<Mutex<Vec<(TuneStage, bool)>>>, TuneStage);

    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.lock().push((self.1, false));
        }
    }

    impl Events {
        fn hook(&self) -> impl Fn(TuneStage, &u32, &u32) -> Guard + Send + Sync + 'static {
            let events = self.0.clone();
            move |stage, _key, _inputs| {
                events.lock().push((stage, true));
                Guard(events.clone(), stage)
            }
        }
    }

    #[test]
    fn a_stage_ends_when_its_guard_drops() {
        let events = Events(Arc::default());
        let hook = events.hook();

        let guard = StageHook::<u32, u32>::enter(&hook, TuneStage::Measuring, &0, &0);
        core::mem::drop(guard);

        assert_eq!(
            *events.0.lock(),
            [(TuneStage::Measuring, true), (TuneStage::Measuring, false)]
        );
    }

    #[test]
    fn a_set_without_a_hook_enters_no_stage() {
        assert!(enter_stage(None, TuneStage::Gathering).is_none());
    }
}
