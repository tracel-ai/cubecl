//! A table the stored values index ([`Tile::lookup`](crate::Tile::lookup)), read back as
//! `table[index]` widened to `f32`.

use std::sync::Arc;

use cubecl::{ir::Scope, prelude::*, unexpanded};

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// A table as the values read it.
pub(crate) trait CodebookRead {
    /// The table's entry at `index`, widened to `f32`.
    fn entry(&self, scope: &Scope, index: NativeExpand<u32>) -> NativeExpand<f32>;
}

impl CodebookRead for TileExpand {
    fn entry(&self, scope: &Scope, index: NativeExpand<u32>) -> NativeExpand<f32> {
        table_entry::expand(scope, self, index)
    }
}

/// `table[index]`, the table read as the one-axis tile it is.
#[cube]
fn table_entry(table: &Tile, index: u32) -> f32 {
    let define!(Served) = table.read_element();
    let entries = table.nd_packed::<Served, Const<1>>(comptime!(Guard::Checked));
    let mut at = CoordsDyn::new();
    at.push(index);
    f32::cast_from(entries.read(at).extract(0usize))
}

/// Each of `indices` replaced by its entry in `codebook`.
#[cube]
fn look_up<T: Numeric, V: Size>(codebook: &Codebook, indices: Vector<T, V>) -> Vector<T, V> {
    let mut entries = Vector::<T, V>::empty();
    #[unroll]
    for j in 0..V::value() {
        let index = u32::cast_from(indices.extract(j));
        entries.insert(j, T::cast_from(codebook.entry(index)));
    }
    entries
}

/// The table these values index, if any. Empty is values that are numbers.
#[derive(Clone)]
pub(crate) struct Codebook;

#[derive(Clone, Default)]
pub(crate) struct CodebookExpand {
    table: Option<Arc<dyn CodebookRead>>,
}

impl Codebook {
    /// No table: what every operand carries until [`Tile::lookup`](crate::Tile::lookup).
    pub(crate) fn none() -> Codebook {
        unexpanded!()
    }

    /// The entry the stored `index` names.
    pub(crate) fn entry(&self, _index: u32) -> f32 {
        unexpanded!()
    }

    /// Every value of `indices` replaced by its entry; unchanged when there is no table.
    pub(crate) fn entries<T: Numeric, V: Size>(&self, _indices: Vector<T, V>) -> Vector<T, V> {
        unexpanded!()
    }

    pub(crate) fn __expand_none(_scope: &Scope) -> CodebookExpand {
        CodebookExpand::default()
    }
}

impl CodebookExpand {
    /// A codebook reading `table`.
    // Expand-time data, held by one thread: `Arc` as the erased tensors hold theirs.
    #[allow(clippy::arc_with_non_send_sync)]
    pub(crate) fn of(table: &TileExpand) -> Self {
        CodebookExpand {
            table: Some(Arc::new(table.clone())),
        }
    }

    /// Whether these values index a table at all.
    pub(crate) fn present(&self) -> bool {
        self.table.is_some()
    }

    pub(crate) fn __expand_entries_method<T: Numeric, V: Size>(
        &self,
        scope: &Scope,
        indices: NativeExpand<Vector<T, V>>,
    ) -> NativeExpand<Vector<T, V>> {
        match self.table {
            Some(_) => look_up::expand::<T, V>(scope, self, indices),
            None => indices,
        }
    }

    pub(crate) fn __expand_entry_method(
        &self,
        scope: &Scope,
        index: NativeExpand<u32>,
    ) -> NativeExpand<f32> {
        self.table
            .as_ref()
            .expect("Codebook::entry: these values index no table")
            .entry(scope, index)
    }
}

impl CubeType for Codebook {
    type ExpandType = CodebookExpand;
}
impl IntoExpand for CodebookExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for CodebookExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for CodebookExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for CodebookExpand {}
impl AsRefExpand for CodebookExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for CodebookExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
