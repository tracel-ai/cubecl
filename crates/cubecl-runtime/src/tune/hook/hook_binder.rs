//! What the tuner calls as a tune runs: the hooks a set registered, bound to one key and the
//! reference inputs of one tune.

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::any::Any;

use crate::tune::{
    AutotuneError, Bounds, BoundsGenerator, Evictor, StageEnterer, TuneInputs, TuneStage,
};

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

/// Binds the optional hooks a set registered to one tune.
pub struct HookBinder<K, I: TuneInputs> {
    bounds_generator: Option<Arc<dyn BoundsGenerator<K, I> + Send + Sync>>,
    evictor: Option<Arc<dyn Evictor<K, I> + Send + Sync>>,
    stage_enterer: Option<Arc<dyn StageEnterer<K, I> + Send + Sync>>,
}

impl<K: Clone + Send + 'static, I: TuneInputs> HookBinder<K, I> {
    /// A binder of no hook.
    pub fn new() -> Self {
        Self {
            bounds_generator: None,
            evictor: None,
            stage_enterer: None,
        }
    }

    pub fn register_bounds_generator(
        &mut self,
        bounds_generator: Arc<dyn BoundsGenerator<K, I> + Send + Sync>,
    ) {
        self.bounds_generator = Some(bounds_generator);
    }

    pub fn register_evictor(&mut self, evictor: Arc<dyn Evictor<K, I> + Send + Sync>) {
        self.evictor = Some(evictor);
    }

    pub fn register_stage_enterer(&mut self, enterer: Arc<dyn StageEnterer<K, I> + Send + Sync>) {
        self.stage_enterer = Some(enterer);
    }

    /// The bounds the registered generator generates for `key` and the reference `inputs`.
    pub fn generate_bounds(&self, key: &K, inputs: &I::At<'_>) -> Option<Bounds> {
        self.bounds_generator
            .as_ref()
            .map(|generator| generator.generate(key, inputs))
    }

    /// The registered evictor, bound to `key` and the reference `inputs`.
    pub fn bind_evictor<'i>(&self, key: &K, inputs: &I::At<'i>) -> Option<Box<BoundEvictor<'i>>> {
        let evictor = self.evictor.clone()?;
        let key = key.clone();
        let inputs = inputs.clone();
        Some(Box::new(move || evictor.evict(&key, &inputs)))
    }

    /// The registered stage enterer, bound to `key` and the reference `inputs`.
    pub fn bind_stage_enterer<'i>(
        &self,
        key: &K,
        inputs: &I::At<'i>,
    ) -> Option<Box<BoundStageEnterer<'i>>> {
        let enterer = self.stage_enterer.clone()?;
        let key = key.clone();
        let inputs = inputs.clone();
        Some(Box::new(move |stage| enterer.enter(stage, &key, &inputs)))
    }
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
        let mut binder = HookBinder::<u8, u8>::new();
        binder.register_evictor(Arc::new(move |key: &u8, inputs: &u8| {
            recorded.lock().push((*key, *inputs));
            Ok::<_, String>(())
        }));

        let mut bound = binder
            .bind_evictor(&1, &2)
            .expect("a registered evictor should bind");
        assert!(bound().is_ok());

        assert_eq!(*seen.lock(), [(1, 2)]);
    }

    #[test]
    fn a_binder_without_an_evictor_binds_none() {
        assert!(HookBinder::<u8, u8>::new().bind_evictor(&1, &2).is_none());
    }

    #[test]
    fn a_bound_enterer_enters_for_its_key_and_reference_inputs() {
        let seen: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let recorded = seen.clone();
        let mut binder = HookBinder::<String, String>::new();
        binder.register_stage_enterer(Arc::new(
            move |_stage: TuneStage, key: &String, inputs: &String| {
                recorded.lock().push((key.clone(), inputs.clone()));
            },
        ));

        let mut bound = binder
            .bind_stage_enterer(&"key".to_string(), &"inputs".to_string())
            .expect("a registered enterer should bind");
        bound(TuneStage::Measuring);

        assert_eq!(*seen.lock(), [("key".to_string(), "inputs".to_string())]);
    }

    #[test]
    fn a_binder_without_an_enterer_binds_none() {
        assert!(
            HookBinder::<u8, u8>::new()
                .bind_stage_enterer(&1, &2)
                .is_none()
        );
    }

    #[test]
    fn a_binder_without_a_bounds_generator_generates_no_bounds() {
        let key = "key".to_string();

        assert!(
            HookBinder::<String, String>::new()
                .generate_bounds(&key, &key)
                .is_none()
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
}
