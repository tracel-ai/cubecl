use core::iter;

use alloc::boxed::Box;
use hi_sparse_bitset::{BitSetBase, BitSetInterface};

use crate::{BitSetExt, SparseBitSet, analyses::dataflow_solver::ChangeResult};

pub mod alias_analysis;
pub mod dataflow_solver;
pub mod dominance;
pub mod liveness;
pub mod memory_ssa;
pub mod pointer_source;
pub mod slices;
pub mod symbol_table;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[allow(clippy::large_enum_variant)]
pub enum MaybeUninitBitset {
    #[default]
    Uninitialized,
    Initialized(SparseBitSet),
}

impl<T> From<T> for MaybeUninitBitset
where
    T: BitSetInterface<Conf = <SparseBitSet as BitSetBase>::Conf>,
{
    fn from(value: T) -> Self {
        MaybeUninitBitset::Initialized(value.into())
    }
}

impl MaybeUninitBitset {
    pub fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (MaybeUninitBitset::Uninitialized, other)
            | (other, MaybeUninitBitset::Uninitialized) => other.clone(),
            (MaybeUninitBitset::Initialized(this), MaybeUninitBitset::Initialized(other)) => {
                MaybeUninitBitset::Initialized(this.union(other).into())
            }
        }
    }

    pub fn unite(&mut self, other: &Self) -> ChangeResult {
        self.set(self.union(other))
    }

    pub fn intersection(&self, other: &Self) -> Self {
        match (self, other) {
            (MaybeUninitBitset::Uninitialized, other)
            | (other, MaybeUninitBitset::Uninitialized) => other.clone(),
            (MaybeUninitBitset::Initialized(this), MaybeUninitBitset::Initialized(other)) => {
                MaybeUninitBitset::Initialized(this.intersection(other).into())
            }
        }
    }

    pub fn intersect(&mut self, other: &Self) -> ChangeResult {
        self.set(self.intersection(other))
    }

    pub fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (MaybeUninitBitset::Uninitialized, _) => MaybeUninitBitset::Uninitialized,
            (this, MaybeUninitBitset::Uninitialized) => this.clone(),
            (MaybeUninitBitset::Initialized(this), MaybeUninitBitset::Initialized(other)) => {
                MaybeUninitBitset::Initialized(this.difference(other).into())
            }
        }
    }

    pub fn raw_difference<Other>(&self, other: Other) -> Self
    where
        Other: BitSetInterface<Conf = <SparseBitSet as BitSetBase>::Conf>,
    {
        match self {
            MaybeUninitBitset::Uninitialized => MaybeUninitBitset::Uninitialized,
            MaybeUninitBitset::Initialized(this) => {
                MaybeUninitBitset::Initialized(this.difference(other).into())
            }
        }
    }

    fn set(&mut self, new: Self) -> ChangeResult {
        match *self == new {
            true => ChangeResult::Unchanged,
            false => {
                *self = new;
                ChangeResult::Changed
            }
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            MaybeUninitBitset::Uninitialized => Box::new(iter::empty()),
            MaybeUninitBitset::Initialized(bit_set) => Box::new(bit_set.iter()),
        }
    }
}
