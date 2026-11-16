//! Expansion-time erasure of a recipe, so a tile holds one whatever its type.

use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ElemType, ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::constant::{Constant, ConstantExpand};
use super::{FactorReads, Recipe, RecipeCoords, RecipeCoordsExpand, RecipeExpand, SeparableExpand};
use crate::Dimension;
use crate::tile::held::read_as;

/// A recipe as a tile holds it: its values, at the element it computes them in.
#[doc(hidden)]
pub(crate) trait RecipeCall {
    fn call(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> ExpandValue;
    fn element(&self, scope: &Scope) -> ElemType;
}

/// A separable recipe as a tile holds it.
#[doc(hidden)]
pub(crate) trait SeparableCall: RecipeCall {
    fn call_factors(&self, scope: &Scope) -> usize;
    fn call_factor_reads(&self, scope: &Scope, factor: usize, dimension: Dimension) -> bool;
    fn call_factor(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> ExpandValue;
}

/// A recipe computing values of `T`, its element held beside it.
struct Computing<T: Numeric, R> {
    recipe: R,
    _computes: PhantomData<T>,
}

impl<T: Numeric, R: RecipeExpand<T>> RecipeCall for Computing<T, R> {
    fn call(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> ExpandValue {
        self.recipe
            .__expand_evaluate_method(scope, coordinates)
            .expand
    }
    fn element(&self, scope: &Scope) -> ElemType {
        T::elem_type(scope)
    }
}

impl<T: Numeric, R> SeparableCall for Computing<T, R>
where
    R: SeparableExpand<T> + RecipeExpand<T> + FactorReads,
{
    fn call_factors(&self, scope: &Scope) -> usize {
        self.recipe.__expand_factors_method(scope)
    }
    fn call_factor_reads(&self, scope: &Scope, factor: usize, dimension: Dimension) -> bool {
        self.recipe.factor_reads(scope, factor, dimension)
    }
    fn call_factor(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> ExpandValue {
        self.recipe
            .__expand_factor_method(scope, coordinates, factor)
            .expand
    }
}

/// Expansion-time type erasure for a [`Recipe`]: it computes at its own element, and a read names
/// the element it wants. It is never a GPU virtual call.
#[derive(Clone)]
pub(crate) struct ErasedRecipe;

#[derive(Clone)]
pub(crate) struct ErasedRecipeExpand {
    state: Arc<dyn RecipeCall>,
    separable: Option<Arc<dyn SeparableCall>>,
    /// The identity the recipe is everywhere, where it is one: what an op it is handed to may
    /// skip.
    pub(crate) identity: Option<Identity>,
}

/// A value an operation leaves its other operand unchanged by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Identity {
    /// Of a sum ([`Procedural::zero`](super::Procedural::zero)).
    Zero,
    /// Of a product ([`Procedural::one`](super::Procedural::one)).
    One,
}

impl Identity {
    fn value(self) -> i64 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

impl ErasedRecipe {
    pub(crate) fn __expand_new<T: Numeric, R: Recipe<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> ErasedRecipeExpand {
        ErasedRecipeExpand {
            state: Arc::new(Computing::<T, _> {
                recipe,
                _computes: PhantomData,
            }),
            separable: None,
            identity: None,
        }
    }

    /// A recipe that is `identity` everywhere, computing at `T`, and says so.
    pub(crate) fn __expand_identity<T: Numeric>(
        scope: &Scope,
        identity: Identity,
    ) -> ErasedRecipeExpand {
        let everywhere = ConstantExpand::<T> {
            value: T::from_int(identity.value()).into_expand(scope),
        };
        ErasedRecipeExpand {
            identity: Some(identity),
            ..Self::__expand_new::<T, Constant<T>>(scope, everywhere)
        }
    }

    pub(crate) fn __expand_new_separable<T: Numeric, R: super::Separable<T> + 'static>(
        _scope: &Scope,
        recipe: R::ExpandType,
    ) -> ErasedRecipeExpand
    where
        R::ExpandType: SeparableExpand<T> + FactorReads,
    {
        let recipe = Arc::new(Computing::<T, _> {
            recipe,
            _computes: PhantomData,
        });
        ErasedRecipeExpand {
            state: recipe.clone(),
            separable: Some(recipe),
            identity: None,
        }
    }

    /// The value at `coordinates`, as `E`.
    pub(crate) fn evaluate<E: Numeric>(&self, _coordinates: &RecipeCoords) -> E {
        unexpanded!()
    }

    /// One factor per contracted dimension if the recipe is separable, else `None`.
    pub(crate) fn factorization(&self) -> comptime_type!(Option<usize>) {
        unexpanded!()
    }

    /// The identity this recipe is everywhere, where it is one.
    pub(crate) fn identity(&self) -> comptime_type!(Option<Identity>) {
        unexpanded!()
    }

    /// Factor `factor` at `coordinates`, as `E`.
    pub(crate) fn factor<E: Numeric>(&self, _coordinates: &RecipeCoords, _factor: usize) -> E {
        unexpanded!()
    }

    /// Whether one separable factor reads `dimension` from its recipe coordinates.
    #[allow(dead_code)] // Reached through its expand, from [`Procedural::factor_reads`].
    pub(crate) fn factor_reads(
        &self,
        _factor: usize,
        _dimension: Dimension,
    ) -> comptime_type!(bool) {
        unexpanded!()
    }
}

impl ErasedRecipeExpand {
    pub(crate) fn __expand_evaluate_method<E: Numeric>(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
    ) -> NativeExpand<E> {
        let value = self.state.call(scope, coordinates);
        read_as::<E>(scope, value, self.state.element(scope))
    }

    pub(crate) fn __expand_factorization_method(&self, scope: &Scope) -> Option<usize> {
        self.separable
            .as_ref()
            .map(|separable| separable.call_factors(scope))
    }

    pub(crate) fn __expand_identity_method(&self, _scope: &Scope) -> Option<Identity> {
        self.identity
    }

    pub(crate) fn __expand_factor_method<E: Numeric>(
        &self,
        scope: &Scope,
        coordinates: &RecipeCoordsExpand,
        factor: usize,
    ) -> NativeExpand<E> {
        let value = match &self.separable {
            Some(separable) => separable.call_factor(scope, coordinates, factor),
            // An unfactorized recipe is its own factor zero.
            None => {
                assert_eq!(factor, 0, "recipe states no factorization beyond itself");
                self.state.call(scope, coordinates)
            }
        };
        read_as::<E>(scope, value, self.state.element(scope))
    }

    pub(crate) fn __expand_factor_reads_method(
        &self,
        scope: &Scope,
        factor: usize,
        dimension: Dimension,
    ) -> bool {
        match &self.separable {
            Some(separable) => separable.call_factor_reads(scope, factor, dimension),
            None => true,
        }
    }
}

impl CubeType for ErasedRecipe {
    type ExpandType = ErasedRecipeExpand;
}
impl IntoExpand for ErasedRecipeExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for ErasedRecipeExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for ErasedRecipeExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
    fn read_place(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for ErasedRecipeExpand {}
impl AsRefExpand for ErasedRecipeExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for ErasedRecipeExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
