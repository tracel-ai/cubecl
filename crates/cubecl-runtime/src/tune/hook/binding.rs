//! What the tuner calls as a tune runs: the hooks a set registered, bound to one key and the
//! reference inputs of one tune.

use alloc::boxed::Box;
use core::any::Any;

use crate::tune::{AutotuneError, TuneStage};

/// An [`Evictor`](crate::tune::Evictor) bound to one key and the reference inputs of one tune: what a benchmark loop
/// calls before each sample.
///
/// `FnMut` rather than `Fn` because the loop lends it out exclusively: the closure holds the
/// inputs, which are only `Send`, and a shared borrow would need them `Sync` to reach the device
/// thread the samples are launched from.
pub type BoundEvictor<'i> = dyn FnMut() -> Result<(), AutotuneError> + Send + 'i;

/// A [`StageEnterer`](crate::tune::StageEnterer) bound to one key and the reference inputs of one tune: what the tuner enters
/// at each stage.
///
/// `FnMut` for the reason [`BoundEvictor`] is.
pub type BoundStageEnterer<'i> = dyn FnMut(TuneStage) -> Box<dyn Any> + Send + 'i;

/// The guard of `stage` entered by `enterer`, or none where the set registered no enterer.
pub fn enter_stage(
    enterer: Option<&mut BoundStageEnterer<'_>>,
    stage: TuneStage,
) -> Option<Box<dyn Any>> {
    enterer.map(|enter| enter(stage))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
