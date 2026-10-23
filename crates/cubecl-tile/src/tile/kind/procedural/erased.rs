//! Expansion-time erasure of a recipe, so a tile holds one whatever its type.

use core::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ElemType, ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::{FactorReads, Recipe, RecipeCoords, RecipeCoordsExpand, RecipeExpand, SeparableExpand};
use crate::Axis;
use crate::tile::held::read_as;

/// A recipe as a tile holds it: its values, at the element it computes them in.
#[doc(hidden)]
pub trait RecipeCall {
    fn call(&self, scope: &Scope, coordinates: &RecipeCoordsExpand) -> ExpandValue;
    fn element(&self, scope: &Scope) -> ElemType;
}

/// A separable recipe as a tile holds it.
#[doc(hidden)]
pub trait SeparableCall: RecipeCall {
    fn call_factors(&self, scope: &Scope) -> usize;
    fn call_factor_reads(&self, scope: &Scope, factor: usize, axis: Axis) -> bool;
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
    fn call_factor_reads(&self, scope: &Scope, factor: usize, axis: Axis) -> bool {
        self.recipe.factor_reads(scope, factor, axis)
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
        }
    }

    /// The value at `coordinates`, as `E`.
    pub(crate) fn evaluate<E: Numeric>(&self, _coordinates: &RecipeCoords) -> E {
        unexpanded!()
    }

    /// One factor per contracted axis if the recipe is separable, else `None`.
    pub(crate) fn factorization(&self) -> comptime_type!(Option<usize>) {
        unexpanded!()
    }

    /// Factor `factor` at `coordinates`, as `E`.
    pub(crate) fn factor<E: Numeric>(&self, _coordinates: &RecipeCoords, _factor: usize) -> E {
        unexpanded!()
    }

    /// Whether one separable factor reads `axis` from its recipe coordinates.
    #[allow(dead_code)] // Reached through its expand, from [`Procedural::factor_reads`].
    pub(crate) fn factor_reads(&self, _factor: usize, _axis: Axis) -> comptime_type!(bool) {
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
        axis: Axis,
    ) -> bool {
        match &self.separable {
            Some(separable) => separable.call_factor_reads(scope, factor, axis),
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
