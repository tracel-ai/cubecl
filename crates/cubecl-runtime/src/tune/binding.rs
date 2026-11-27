//! The hooks a set registered, bound to one key and the reference inputs of one tune: what the
//! tuner calls as the tune runs.

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::any::Any;

use super::{AutotuneError, Bounds, BoundsGenerator, Evictor, StageEnterer, TuneInputs, TuneStage};

/// An [`Evictor`] bound to one key and the reference inputs of one tune: what a benchmark loop
/// calls before each sample.
///
/// `FnMut` rather than `Fn` because the loop lends it out exclusively: the closure holds the
/// inputs, which are only `Send`, and a shared borrow would need them `Sync` to reach the device
/// thread the samples are launched from.
pub type BoundEvictor<'i> = dyn FnMut() -> Result<(), AutotuneError> + Send + 'i;

/// `evictor`, when a set registered one, bound to `key` and the reference `inputs`.
pub fn bind_evictor<'i, K: Clone + Send + 'static, I: TuneInputs>(
    evictor: Option<&Arc<dyn Evictor<K, I> + Send + Sync>>,
    key: &K,
    inputs: &I::At<'i>,
) -> Option<Box<BoundEvictor<'i>>> {
    let evictor = evictor?.clone();
    let key = key.clone();
    let inputs = inputs.clone();
    Some(Box::new(move || evictor.evict(&key, &inputs)))
}

/// A [`StageEnterer`] bound to one key and the reference inputs of one tune: what the tuner enters
/// at each stage.
///
/// `FnMut` for the reason [`BoundEvictor`] is: it holds the inputs,
/// which are only `Send`, and is lent out exclusively to the device thread the stage runs on.
pub type BoundStageEnterer<'i> = dyn FnMut(TuneStage) -> Box<dyn Any> + Send + 'i;

/// `enterer`, when a set registered one, bound to `key` and the reference `inputs`.
pub fn bind_stage_enterer<'i, K: Clone + Send + 'static, I: TuneInputs>(
    enterer: Option<&Arc<dyn StageEnterer<K, I> + Send + Sync>>,
    key: &K,
    inputs: &I::At<'i>,
) -> Option<Box<BoundStageEnterer<'i>>> {
    let enterer = enterer?.clone();
    let key = key.clone();
    let inputs = inputs.clone();
    Some(Box::new(move |stage| enterer.enter(stage, &key, &inputs)))
}

/// The bounds `generator`, when a set registered one, generates for `key` and the reference
/// `inputs`.
pub fn generate_bounds<K: 'static, I: TuneInputs>(
    generator: Option<&Arc<dyn BoundsGenerator<K, I> + Send + Sync>>,
    key: &K,
    inputs: &I::At<'_>,
) -> Option<Bounds> {
    generator.map(|generator| generator.generate(key, inputs))
}

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
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use cubecl_environment::sync::Mutex;

    #[test]
    fn a_bound_evictor_evicts_for_its_key_and_reference_inputs() {
        let seen: Arc<Mutex<Vec<(u8, u8)>>> = Arc::default();
        let recorded = seen.clone();
        let evictor: Arc<dyn Evictor<u8, u8> + Send + Sync> =
            Arc::new(move |key: &u8, inputs: &u8| {
                recorded.lock().push((*key, *inputs));
                Ok::<_, String>(())
            });

        let mut bound =
            bind_evictor(Some(&evictor), &1, &2).expect("a registered evictor should bind");
        assert!(bound().is_ok());

        assert_eq!(*seen.lock(), [(1, 2)]);
    }

    #[test]
    fn a_set_without_an_evictor_binds_none() {
        assert!(bind_evictor::<u8, u8>(None, &1, &2).is_none());
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
    fn a_bound_enterer_enters_for_its_key_and_reference_inputs() {
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let recorded = seen.clone();
        let enterer: Arc<dyn StageEnterer<String, String> + Send + Sync> =
            Arc::new(move |_stage: TuneStage, key: &String, inputs: &String| {
                recorded.lock().push((key.clone(), inputs.clone()));
            });

        let mut bound =
            bind_stage_enterer(Some(&enterer), &"key".to_string(), &"inputs".to_string())
                .expect("a registered enterer should bind");
        bound(TuneStage::Measuring);

        assert_eq!(*seen.lock(), [("key".to_string(), "inputs".to_string())]);
    }
}
