use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Display;

use crate::tune::set::group::TuneGroup;
use crate::tune::set::membership::GroupMembership;
use crate::tune::set::tune_fn::TuneFn;
use crate::tune::{AutotuneError, TuneInputs};

/// A single candidate for autotune: a named function plus the [groups](TuneGroup) it belongs to.
///
/// `Id` is what the tunables of a set are [identified](Self::identified) by, `()` for a set whose
/// tunables are [named](Self::new) and nothing more.
///
/// # Assumptions
///
/// - Its function launches the same kernels for equal inputs: a tune measures it on generated
///   inputs and keeps the answer for every call of the key.
///
/// # Guarantees
///
/// - It is autotuned whenever any of its groups is prioritized.
pub struct Tunable<K, F: TuneInputs, Output, Id = ()> {
    function: TuneFn<F, Output>,
    /// The groups it joined, in the order it joined them (Arrival Usage).
    groups: Vec<GroupMembership<K>>,
    identity: Id,
}

impl<K, F: TuneInputs, Output: 'static> Tunable<K, F, Output> {
    /// A tunable named `name`, running `func`.
    ///
    /// The `for<'a> Fn(F::At<'a>) -> _` bound is spelled out in the `where`-clause rather than
    /// hidden behind a helper trait, so that closure inference sees it: otherwise `move |input| …`
    /// picks one concrete lifetime and fails with `implementation of FnOnce is not general enough`
    /// wherever `F::At<'a>` depends on `'a`.
    ///
    /// For multi-input kernels, destructure a tuple:
    /// `Tunable::new("name", |(lhs, rhs, out)| body)`.
    ///
    /// A tunable in no [group](Tunable::group) states no priority, so it trails the grouped
    /// candidates of its round and a short circuit among them leaves it unmeasured.
    ///
    /// # Guarantees
    ///
    /// - An error `func` returns reaches the tune as [`AutotuneError::Rejected`]: the candidate
    ///   does not serve the call, and is never decided for it.
    pub fn new<Func, Err>(name: &str, func: Func) -> Self
    where
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<F as TuneInputs>::At<'a>) -> Result<Output, Err> + Send + Sync + 'static,
    {
        Self::build(name, (), func)
    }
}

impl<K, F: TuneInputs, Output: 'static, Id> Tunable<K, F, Output, Id> {
    /// A tunable identified by `identity`, and named by it: the value
    /// [`LocalTuner::fastest_identity`](crate::tune::LocalTuner::fastest_identity) hands back once the
    /// tunable is the fastest for a key.
    ///
    /// For a caller that builds something else from the winner — a variant of the same kernel,
    /// say — and so needs to know *which* candidate won rather than only to run it again. The name
    /// a cached result is checksummed by is `identity`'s own, so the two cannot disagree.
    ///
    /// # Assumptions
    ///
    /// - Two identities of one set display differently: a set is told apart from another by its
    ///   candidates' names alone.
    pub fn identified<Func, Err>(identity: Id, func: Func) -> Self
    where
        Id: Display,
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<F as TuneInputs>::At<'a>) -> Result<Output, Err> + Send + Sync + 'static,
    {
        Self::build(&identity.to_string(), identity, func)
    }

    /// Add this tunable to `group` with the given intra-group priority.
    ///
    /// Groups run in order of their own priority. Within a cutoff group the member priority picks
    /// one level as the round and leaves the levels under it as fallbacks. Within an
    /// [ordered](TuneGroup::ordered) group it orders one round that holds every member.
    ///
    /// # Assumptions
    ///
    /// - `priority` answers from the key alone, as [`TuneGroup`]'s does.
    ///
    /// # Guarantees
    ///
    /// - A negative priority skips the tunable for the key, in this group.
    pub fn group(
        mut self,
        group: &TuneGroup<K>,
        priority: impl Fn(&K) -> i8 + Send + Sync + 'static,
    ) -> Self {
        self.groups.push(GroupMembership::new(group, priority));
        self
    }
}

impl<K, F: TuneInputs, Output: 'static, Id> Tunable<K, F, Output, Id> {
    fn build<Func, Err>(name: &str, identity: Id, func: Func) -> Self
    where
        Err: Into<String> + 'static,
        Func: for<'a> Fn(<F as TuneInputs>::At<'a>) -> Result<Output, Err> + Send + Sync + 'static,
    {
        let name: String = name.into();
        let name_for_err = name.clone();
        Self {
            function: TuneFn::new(
                name,
                Box::new(move |inputs| {
                    func(inputs).map_err(|err| AutotuneError::Rejected {
                        name: name_for_err.to_string(),
                        reason: err.into(),
                    })
                }),
            ),
            groups: Vec::new(),
            identity,
        }
    }

    pub(crate) fn identity(&self) -> &Id {
        &self.identity
    }

    pub(crate) fn function(&self) -> &TuneFn<F, Output> {
        &self.function
    }

    /// Every group the tunable joined, in the order it joined them.
    pub(crate) fn groups(&self) -> &[GroupMembership<K>] {
        &self.groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_of_the_function_is_a_rejection() {
        let tunable = Tunable::<(), (), ()>::new("declines", |_: ()| Err("not this shape"));

        assert!(matches!(
            tunable.function().execute(()),
            Err(AutotuneError::Rejected { .. })
        ));
    }

    #[test]
    fn an_identified_tunable_is_named_by_its_identity() {
        let tunable = Tunable::<(), (), (), u8>::identified(7, |_: ()| Ok::<_, String>(()));

        assert_eq!((tunable.function().name(), *tunable.identity()), ("7", 7));
    }

    #[test]
    fn a_tunable_keeps_the_groups_it_joined_in_order() {
        let first = TuneGroup::<()>::new("first", |_| 1);
        let second = TuneGroup::<()>::new("second", |_| 1);
        let tunable = Tunable::<(), (), ()>::new("joins", |_: ()| Ok::<_, String>(()))
            .group(&first, |_| 2)
            .group(&second, |_| 3);

        let joined: Vec<_> = tunable
            .groups()
            .iter()
            .map(|membership| (membership.group().id(), membership.priority(&())))
            .collect();
        assert_eq!(joined, [(first.id(), 2), (second.id(), 3)]);
    }
}
