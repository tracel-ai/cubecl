use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use cubecl_common::hash::StableHash;

use crate::tune::hook::binding::{
    BoundEvictor, BoundStageEnterer, bind_evictor, bind_stage_enterer, generate_bounds,
};
use crate::tune::hook::clone_input_generator::CloneInputGenerator;
use crate::tune::plan::planner::TunePlanner;
use crate::tune::set::candidate_planner::{CandidatePlanner, names_hash};
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{
    AutotuneKeyable, Bounds, BoundsGenerator, Evictor, InputGenerator, KeyGenerator, StageEnterer,
    Tunable, TuneDeviceAccess, TuneInputs,
};

/// A set of candidate tunable functions for autotune, sharing a key generator and an input
/// generator. See [`TuneInputs`] for the `F` parameter.
///
/// `Id` is what its tunables are [identified](Tunable::identified) by, inferred from the tunables
/// it is given: `()` for tunables built by [`Tunable::new`], named and nothing more.
///
/// # Assumptions
///
/// - Two sets whose candidates have the same names, in the same order, hold the same candidates:
///   a set is told apart from another, in memory and in the persistent cache, by its
///   [checksum](Self::compute_checksum) alone, so a kernel changed behind an unchanged name keeps
///   the answers tuned before the change.
pub struct TunableSet<K: AutotuneKeyable, F: TuneInputs, Output: 'static, Id = ()> {
    candidates: CandidatePlanner<K, F, Output, Id>,
    /// The hash of the candidates' names, kept in step with them: every launch looks its tuner up
    /// by it.
    checksum: StableHash,
    key_gen: Arc<dyn KeyGenerator<K, F> + Send + Sync>,
    input_gen: Arc<dyn InputGenerator<K, F> + Send + Sync>,
    bounds_gen: Option<Arc<dyn BoundsGenerator<K, F> + Send + Sync>>,
    evictor: Option<Arc<dyn Evictor<K, F> + Send + Sync>>,
    stage_enterer: Option<Arc<dyn StageEnterer<K, F> + Send + Sync>>,
    device_access: TuneDeviceAccess,
    short_circuit: bool,
}

impl<K: AutotuneKeyable, F: TuneInputs, Output: 'static, Id> TunableSet<K, F, Output, Id> {
    /// A set with no candidate, keyed by `key_gen` and measured on what `input_gen` makes of the
    /// reference inputs.
    pub fn new(key_gen: impl KeyGenerator<K, F>, input_gen: impl InputGenerator<K, F>) -> Self {
        Self {
            candidates: CandidatePlanner::new(),
            checksum: names_hash::<K, F, Output, Id>(&[]),
            input_gen: Arc::new(input_gen),
            key_gen: Arc::new(key_gen),
            bounds_gen: None,
            evictor: None,
            stage_enterer: None,
            device_access: TuneDeviceAccess::Exclusive,
            short_circuit: true,
        }
    }

    /// [`new`](Self::new), measured on clones of the real call inputs.
    pub fn new_cloning_inputs(key_gen: impl KeyGenerator<K, F>) -> Self {
        Self::new(key_gen, CloneInputGenerator)
    }

    /// The number of candidates, which bounds every index a tune of the set answers.
    pub fn len(&self) -> usize {
        self.candidates.functions().len()
    }

    /// Whether the set holds no candidate, so that no call of it can execute.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The set with `tunable` after its other candidates.
    ///
    /// # Guarantees
    ///
    /// - The set plans every key afresh, `tunable` included, and its checksum names `tunable`.
    pub fn with(mut self, tunable: Tunable<K, F, Output, Id>) -> Self {
        self.checksum = self.candidates.add(tunable);
        self
    }

    /// The set with `bounds` setting the time limit a candidate is close enough under.
    pub fn with_bounds(
        mut self,
        bounds_generator: Arc<dyn BoundsGenerator<K, F> + Send + Sync>,
    ) -> Self {
        self.bounds_gen = Some(bounds_generator);
        self
    }

    /// The set with `evictor` running before every measured sample, so the candidates are timed
    /// reading memory rather than the cache the previous sample left warm.
    ///
    /// # Guarantees
    ///
    /// - The eviction is issued outside the profiled region, so it costs the round time rather
    ///   than the sample.
    /// - A failed eviction is logged and the sample is measured anyway, warm: an eviction is a
    ///   measurement aid, and losing one is not worth failing the tune for.
    pub fn with_evictor(mut self, evictor: Arc<dyn Evictor<K, F> + Send + Sync>) -> Self {
        self.evictor = Some(evictor);
        self
    }

    /// The set with `enterer` entering every stage of its tunes, so candidates that launch where
    /// the tuner does not see are kept in step with it.
    pub fn with_stage_enterer(
        mut self,
        enterer: Arc<dyn StageEnterer<K, F> + Send + Sync>,
    ) -> Self {
        self.stage_enterer = Some(enterer);
        self
    }

    /// The set with its tunes holding the device as `access` does.
    pub fn with_device_access(mut self, access: TuneDeviceAccess) -> Self {
        self.device_access = access;
        self
    }

    /// The set with the short circuit, which ends a batch at its first candidate under the
    /// bounds' time limit, turned on where `enabled`.
    pub fn with_short_circuit(mut self, enabled: bool) -> Self {
        self.short_circuit = enabled;
        self
    }

    /// The checksum a cached answer is stored under: an answer to another candidate list is an
    /// answer to another question.
    pub fn compute_checksum(&self) -> String {
        format!("{:x}", self.checksum)
    }

    /// The key of the problem `inputs` pose.
    pub fn generate_key<'a>(&self, inputs: &F::At<'a>) -> K {
        self.key_gen.generate(inputs)
    }

    /// What the candidates are measured on for `key`, made from the reference `inputs`.
    pub fn generate_inputs<'a>(&self, key: &K, inputs: &F::At<'a>) -> F::At<'a> {
        self.input_gen.generate(key, inputs)
    }

    /// The bounds the candidates of `key` are measured against, if the set has a bounds generator.
    pub fn bounds(&self, key: &K, inputs: &F::At<'_>) -> Option<Bounds> {
        generate_bounds(self.bounds_gen.as_ref(), key, inputs)
    }
}

impl<K: AutotuneKeyable, F: TuneInputs, Output: 'static, Id> TunableSet<K, F, Output, Id> {
    pub(crate) fn device_access(&self) -> TuneDeviceAccess {
        self.device_access
    }

    pub(crate) fn is_short_circuit_enabled(&self) -> bool {
        self.short_circuit
    }

    /// Every candidate's function, in registration order.
    pub(crate) fn autotunables(&self) -> impl Iterator<Item = &TuneFn<F, Output>> {
        self.candidates.functions()
    }

    pub(crate) fn plan(&self, key: &K) -> TunePlanner {
        self.candidates.plan(key)
    }

    /// The candidates of `key` in the order an untuned key runs them.
    pub(crate) fn run_order(&self, key: &K) -> Arc<[usize]> {
        self.candidates.run_order(key)
    }

    /// The function of the candidate at `index`, an index the tuner answered for this set.
    pub(crate) fn function_at(&self, index: usize) -> &TuneFn<F, Output> {
        self.candidates.tunable_at(index).function()
    }

    /// What the candidate at `index` is [identified](Tunable::identified) by, an index the tuner
    /// answered for this set.
    pub(crate) fn identity_at(&self, index: usize) -> &Id {
        self.candidates.tunable_at(index).identity()
    }

    /// The low half of the [checksum](Self::compute_checksum), without hashing the names again.
    pub(crate) fn checksum(&self) -> u64 {
        self.checksum as u64
    }

    /// The set's evictor, if any, bound to `key` and the reference `inputs`.
    pub(crate) fn evictor<'i>(&self, key: &K, inputs: &F::At<'i>) -> Option<Box<BoundEvictor<'i>>> {
        bind_evictor(self.evictor.as_ref(), key, inputs)
    }

    /// The set's stage enterer, if any, bound to `key` and the reference `inputs`.
    pub(crate) fn bound_stage_enterer<'i>(
        &self,
        key: &K,
        inputs: &F::At<'i>,
    ) -> Option<Box<BoundStageEnterer<'i>>> {
        bind_stage_enterer(self.stage_enterer.as_ref(), key, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn set() -> TunableSet<String, String, ()> {
        TunableSet::new_cloning_inputs(|inputs: &String| inputs.clone())
    }

    fn runs(_: String) -> Result<(), String> {
        Ok(())
    }

    #[test]
    fn a_candidate_changes_the_checksum() {
        let empty = set();
        let one = set().with(Tunable::new("one", runs));

        assert_ne!(empty.compute_checksum(), one.compute_checksum());
        assert_ne!(empty.checksum(), one.checksum());
    }

    #[test]
    fn a_set_with_no_bounds_generator_has_no_bounds() {
        let key = "key".to_string();

        assert!(set().bounds(&key, &key).is_none());
    }

    #[test]
    fn a_cloning_set_measures_on_the_reference_inputs() {
        let inputs = "inputs".to_string();

        assert_eq!(set().generate_inputs(&"key".to_string(), &inputs), inputs);
    }
}
