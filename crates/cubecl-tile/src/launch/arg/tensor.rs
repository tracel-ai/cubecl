//! An operand as a single launch argument: its values, a buffer or a TMA tensor map, its comptime
//! [`TileSpec`], the element it serves them at, the runtime half of its projection, and what its
//! values are read through: the table that places them, the table they index and the scales they
//! are under, each a [`TileArg`] of its own.

use std::marker::PhantomData;
use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::frontend::{buffer_len::expand_buffer_length_native, slice};
use cubecl::ir::pliron::value::Value;
use cubecl::ir::{ElemType, Scope, types::VectorType};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    layout::{CoordsDyn, VirtualLayout, VirtualLayoutExpand},
    view::launch::{VirtualViewLayoutCompilationArg, VirtualViewLayoutLaunch},
};

use crate::launch::binding::rebound_tensor;
use crate::tile::packing::base::Packing;
use crate::*;
use cubecl::server::BufferBinding;

/// An operand as a single launch argument: its values, a buffer or a TMA tensor map, its comptime
/// [`TileSpec`], the element its buffer holds, and the scales they are read under.
///
/// The element is data the argument carries, stated where the operand is bound
/// ([`TileArgLaunch::new`]), not a type of the kernel's: what the buffer holds, `width` of them a
/// line. A packed buffer holds `u32` words, its spec's packing naming the fields in them; a reader
/// decodes them into the element it asks for ([`Tile::dequantized`]), or `f32` where it asks for
/// none.
///
/// Scales are levels innermost first, so each covers more than the one inside it: NVFP4's values
/// carry their block scales, which carry the per-tensor scale. A plain level holding `u32` is words
/// of block scales, which only the block-scaled instruction reads; any other, a packed one
/// included, is scales read as numbers.
///
/// Values stored as indices ([`Field::Index`]) read through the table they index, their codebook
/// ([`TileArgLaunch::codebook`]); a gather's runtime coefficients and offsets
/// ([`TileArgLaunch::runtime_map`]) ride the argument too, as does the table that places their
/// windows ([`TileArgLaunch::indexed`]), a routed product's experts or a pool's pages. A kernel
/// serves every operand the same way, with [`tile`](TileArg::tile).
pub struct TileArg<'a> {
    pub spec: TileSpec,
    /// The element the buffer holds.
    pub element: ElemType,
    _values: PhantomData<&'a ()>,
}

impl TileArg<'_> {
    /// Serve the operand as a [`Tile`] over the kernel's one `partitioning`, under its scales.
    pub fn tile(&self, _partitioning: &Partitioning) -> Tile {
        unexpanded!()
    }

    /// The buffer as lines of `V` of `E`, which must be the element it holds and the width it is
    /// bound at: what a write that adds into a line reads and writes it as. A launch that bound
    /// the buffer at another element is refused as the kernel expands.
    pub fn lines<E: Numeric, V: Size>(&self) -> &Tensor<Vector<E, V>> {
        unexpanded!()
    }
}

/// A buffer of values, served at `E`, its projection's runtime half `coefficients` and `offsets`.
#[cube]
fn buffer_tile<E: Numeric>(
    tensor: &Tensor<E>,
    partitioning: &Partitioning,
    coefficients: Coords<u32>,
    offsets: Coords<i32>,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    GlobalOperand::gathered(
        tensor,
        comptime!(space.space().clone()),
        spec,
        coefficients,
        offsets,
    )
    .tile::<E>(comptime!(space.levels().to_vec()))
}

/// A codebook's table, `entries` long along its one axis, outside the kernel's space.
#[cube]
fn table_tile<E: Numeric>(
    tensor: &Tensor<E>,
    #[comptime] entries: usize,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(Space::new(&[(spec.axes()[0], entries)]));
    GlobalOperand::tensor(tensor, comptime!(space.clone()), spec).tile::<E>(comptime!(Vec::new()))
}

/// [`buffer_tile`] over a pool of pages, read as the sequences `table` names.
#[cube]
fn paged_tile<E: Numeric>(
    tensor: &Tensor<E>,
    table: &Tensor<u32>,
    partitioning: &Partitioning,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    GlobalOperand::paged(tensor, table, comptime!(space.space().clone()), spec)
        .tile::<E>(comptime!(space.levels().to_vec()))
}

/// A tensor map, served as a TMA-filled tile.
#[cube]
fn tensor_map_tile<E: Numeric>(
    map: TensorMap<E, Tiled>,
    layout: VirtualLayout<CoordsDyn, Sequence<i32>>,
    partitioning: &Partitioning,
    #[comptime] spec: TileSpec,
    #[comptime] width: usize,
) -> Tile {
    let space = comptime!(partitioning.clone());
    let own = comptime!(space.space().subspace(spec.axes()));
    let data = TmaData::from_tensor_map(
        map,
        layout,
        comptime!(own.rank()),
        comptime!(spec.units),
        comptime!(spec.packing),
        width,
    );
    Tile::new::<E>(
        TileKind::new_TmaGmem(data),
        comptime!(Placement::root(own.clone(), space.levels().to_vec())),
    )
}

/// The device side of a [`TileArg`].
#[derive(Clone)]
pub struct TileArgExpand<'a> {
    values: ValuesExpand<'a>,
    pub spec: TileSpec,
    /// The element the buffer holds.
    pub element: ElemType,
    /// The scales the values are read under: none, or a coarser operand carrying its own.
    scale: Option<Box<TileArgExpand<'static>>>,
    /// The table that places the values' windows, and what its entry places ([`Indexing`]).
    index: Option<(Box<TileArgExpand<'static>>, Indexing)>,
    /// The table the values index, where they are stored as indices.
    codebook: Option<Box<TileArgExpand<'static>>>,
    /// The projection's dynamic coefficients and divisors, then its dynamic offsets.
    runtime_map: (CoordsExpand<u32>, CoordsExpand<i32>),
}

/// The device side of the values: a buffer, kept also as it was declared for
/// [`lines`](TileArg::lines), or a tensor map and the layout landing a position in it.
#[derive(Clone)]
enum ValuesExpand<'a> {
    Tensor {
        /// The values, kept at [`Held`] and named at their element where a tile reads them.
        tensor: &'a TensorExpand<Held>,
        buffer: Value,
        meta: TensorMetaExpand,
        width: usize,
    },
    TensorMap {
        map: Value,
        layout: VirtualLayoutExpand<CoordsDyn, Sequence<i32>>,
        width: usize,
    },
}

impl<'a> TileArgExpand<'a> {
    pub fn __expand_tile_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let values = self.values_tile(scope, partitioning);
        let values = self.looking_up(scope, values);
        let values = self.indexing(scope, values, partitioning);
        self.scaling(scope, values, partitioning)
    }

    pub fn __expand_lines_method<E: Numeric, V: Size>(
        &self,
        scope: &Scope,
    ) -> &'a TensorExpand<Vector<E, V>> {
        let ValuesExpand::Tensor {
            buffer,
            meta,
            width: bound,
            ..
        } = &self.values
        else {
            panic!("TileArg::lines: a tensor map has no lines to add into")
        };
        let named = E::elem_type(scope);
        assert_eq!(
            named, self.element,
            "TileArg::lines: the buffer is bound at {}, not {named}",
            self.element
        );
        let width = V::__expand_value(scope);
        assert_eq!(
            width, *bound,
            "TileArg::lines: the buffer is bound {bound} wide, not {width}"
        );
        let len = expand_buffer_length_native(scope, *buffer);
        let lines = slice::from_raw_parts::<Vector<E, V>>(
            scope,
            *buffer,
            0usize.into_expand(scope),
            len.into(),
        );
        scope.create_kernel_ref(TensorExpand::__expand_from_parts(*meta, lines))
    }

    /// The values alone, under none of their scales: read through their table where they are a
    /// pool of pages, where they lie otherwise.
    fn values_tile(&self, scope: &Scope, partitioning: &PartitioningExpand) -> TileExpand {
        let _held = scope.define_type::<Held>(self.element);
        match &self.values {
            ValuesExpand::Tensor { tensor, meta, .. } => {
                let tensor = scope.create_kernel_ref(TensorExpand::<Held>::__expand_from_parts(
                    *meta,
                    tensor.expand.into(),
                ));
                if let (Some(_), Some((table, _))) = (&self.spec.paging, &self.index) {
                    let ValuesExpand::Tensor {
                        tensor: pages,
                        meta,
                        ..
                    } = &table.values
                    else {
                        unreachable!("TileArgLaunch::indexed binds its table as a buffer")
                    };
                    let pages = scope.create_kernel_ref(TensorExpand::<u32>::__expand_from_parts(
                        *meta,
                        pages.expand.into(),
                    ));
                    return paged_tile::expand::<Held>(
                        scope,
                        tensor,
                        pages,
                        partitioning,
                        self.spec.clone(),
                    );
                }
                let (coefficients, offsets) = self.runtime_map.clone();
                buffer_tile::expand::<Held>(
                    scope,
                    tensor,
                    partitioning,
                    coefficients,
                    offsets,
                    self.spec.clone(),
                )
            }
            ValuesExpand::TensorMap { map, layout, width } => tensor_map_tile::expand::<Held>(
                scope,
                (*map).into(),
                layout.clone(),
                partitioning,
                self.spec.clone(),
                *width,
            ),
        }
    }

    /// `values` read through the table they index, where they are stored as indices.
    fn looking_up(&self, scope: &Scope, values: TileExpand) -> TileExpand {
        let Some(table) = &self.codebook else {
            return values;
        };
        let ValuesExpand::Tensor { tensor, meta, .. } = &table.values else {
            unreachable!("TileArgLaunch::codebook binds a buffer")
        };
        let Packing::Packed {
            field: Field::Index { bits },
        } = self.spec.packing
        else {
            unreachable!("TileArgLaunch::codebook binds values stored as indices")
        };
        let _held = scope.define_type::<Held>(table.element);
        let entries = scope.create_kernel_ref(TensorExpand::<Held>::__expand_from_parts(
            *meta,
            tensor.expand.into(),
        ));
        let table = table_tile::expand::<Held>(scope, entries, 1 << bits, table.spec.clone());
        values.__expand_lookup_method(scope, &table)
    }

    /// `values` carrying the table that places their windows, where the operand has one: read at
    /// the first level of `partitioning` that cuts the table's keys ([`Indexing::read_at`]), or
    /// now, at its first entry, where none does. A pool of pages reads its table as each window
    /// is cut and each line is offset ([`values_tile`](Self::values_tile)), so holds none here.
    fn indexing(
        &self,
        scope: &Scope,
        values: TileExpand,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let Some((table, indexing)) = &self.index else {
            return values;
        };
        if self.spec.paging.is_some() {
            return values;
        }
        let keys = table.spec.axes().to_vec();
        let read_at = indexing
            .read_at(&keys, &partitioning.levels)
            .unwrap_or_else(|refusal| panic!("{refusal}"));
        let ValuesExpand::Tensor { tensor, meta, .. } = &table.values else {
            panic!("{}", Refusal::IndexTableNotU32)
        };
        let entries = scope.create_kernel_ref(TensorExpand::<u32>::__expand_from_parts(
            *meta,
            tensor.expand.into(),
        ));
        let table = IndexTable::__expand_new(scope, entries, keys, indexing.clone());
        let values = values.__expand_with_index_method(scope, table);
        match read_at {
            Some(_) => values,
            None => values.__expand_indexed_whole_method(scope),
        }
    }

    /// `values` under these scales, a tile carrying its own scales in turn. A tensor map's values
    /// land in a stage as they lie, so the scales on them are words of block scales, which a stage
    /// keeps beside them; a scale above the words leaves the operand for the sum (`Tile::factor`)
    /// before the stage is filled, and staging checks what is left.
    fn scaling(
        &self,
        scope: &Scope,
        values: TileExpand,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let Some(level) = &self.scale else {
            return values;
        };
        // Words of block scales, or block scales packed in words a field a scale, which a reader
        // that decodes as it loads asks a stage to keep beside the values.
        let words = level.element == u32::elem_type_native();
        assert!(
            words || !matches!(self.values, ValuesExpand::TensorMap { .. }),
            "TileArg: a tensor map lands its values as they lie, so it rides only scales a stage \
             keeps beside them: words of block scales, or block scales packed in words"
        );
        values.__expand_mul_method(scope, &level.__expand_tile_method(scope, partitioning))
    }
}

/// The host side of a [`TileArg`]: its values, its spec, the element its buffer holds and its
/// scales. The spec is shared: a rebound operand and the compilation argument it registers as
/// read the one its launch was settled with.
pub struct TileArgLaunch {
    values: ValuesLaunch,
    spec: Arc<TileSpec>,
    element: ElemType,
    scale: Option<Box<TileArgLaunch>>,
    index: Option<(Box<TileArgLaunch>, Indexing)>,
    codebook: Option<Box<TileArgLaunch>>,
    runtime_map: (Vec<u32>, Vec<i32>),
}

/// The host side of the values: a tensor and the width it is bound at, or a tensor map, boxed for
/// being several times a tensor's size, its layout and the width a stage it fills is read at.
enum ValuesLaunch {
    Tensor {
        tensor: TensorArg,
        width: usize,
    },
    TensorMap {
        map: Box<TensorMapArg<Tiled>>,
        layout: VirtualViewLayoutLaunch<CoordsDyn, Sequence<i32>, TensorMapArg<Tiled>>,
        width: usize,
    },
}

impl TileArgLaunch {
    /// `tensor`, holding `element`, bound `width` wide, as `spec` reads it, under no scales.
    pub fn new(tensor: TensorArg, element: ElemType, width: usize, spec: TileSpec) -> Self {
        Self::of(ValuesLaunch::Tensor { tensor, width }, element, spec)
    }

    /// A tensor map landing through `layout`, holding the element its descriptor moves, as `spec`
    /// reads it, under no scales.
    pub(crate) fn of_tensor_map(
        map: TensorMapArg<Tiled>,
        layout: VirtualViewLayoutLaunch<CoordsDyn, Sequence<i32>, TensorMapArg<Tiled>>,
        spec: TileSpec,
    ) -> Self {
        let element = map.metadata.elem_ty;
        let map = Box::new(map);
        Self::of(
            ValuesLaunch::TensorMap {
                map,
                layout,
                width: 1,
            },
            element,
            spec,
        )
    }

    /// This tensor map with the stage it fills read `width` values a line: the engine lands a box
    /// as bytes, whatever the line its readers take, so a stage of a tensor map is as wide as
    /// they read it, not one value a line.
    ///
    /// # Panics
    ///
    /// On a tensor, whose width is the one it is bound at ([`new`](Self::new)).
    pub fn served_wide(mut self, width: usize) -> Self {
        match &mut self.values {
            ValuesLaunch::TensorMap { width: served, .. } => *served = width,
            ValuesLaunch::Tensor { .. } => {
                panic!("TileArgLaunch::served_wide: a tensor is read at the width it is bound at")
            }
        }
        self
    }

    fn of(values: ValuesLaunch, element: ElemType, spec: TileSpec) -> Self {
        assert!(
            spec.packing == Packing::Plain || element == u32::elem_type_native(),
            "TileArgLaunch: a packed operand is bound at the u32 words its buffer holds, not \
             {element:?}"
        );
        Self {
            values,
            spec: Arc::new(spec),
            element,
            scale: None,
            index: None,
            codebook: None,
            runtime_map: (Vec::new(), Vec::new()),
        }
    }

    /// This operand bound again over other buffers of the same layout: each tensor it holds, in the
    /// order it holds them (its values, then its scales', then its index table's, then its
    /// codebook's), takes the next
    /// of `buffers` and keeps the shape, strides and tiling it was settled at. A launch that binds
    /// the same operands every step keeps its binding and changes only the addresses.
    ///
    /// `None` for a tensor map, whose descriptor holds its buffer, and where `buffers` runs out.
    pub fn rebound(&self, buffers: &mut impl Iterator<Item = BufferBinding>) -> Option<Self> {
        let values = match &self.values {
            ValuesLaunch::Tensor { tensor, width } => ValuesLaunch::Tensor {
                tensor: rebound_tensor(tensor, buffers.next()?)?,
                width: *width,
            },
            ValuesLaunch::TensorMap { .. } => return None,
        };
        let scale = match &self.scale {
            Some(scale) => Some(Box::new(scale.rebound(buffers)?)),
            None => None,
        };
        let index = match &self.index {
            Some((table, indexing)) => Some((Box::new(table.rebound(buffers)?), indexing.clone())),
            None => None,
        };
        let codebook = match &self.codebook {
            Some(table) => Some(Box::new(table.rebound(buffers)?)),
            None => None,
        };
        Some(Self {
            values,
            spec: self.spec.clone(),
            element: self.element,
            scale,
            index,
            codebook,
            runtime_map: self.runtime_map.clone(),
        })
    }

    /// These values, stored as indices ([`Field::Index`]), read as the entries of `table` they
    /// name: a plain one-dim buffer of at least an entry per index, read one value a line.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] for values not stored as indices, or a table that is not a plain one-dim
    /// buffer holding an entry for every index.
    pub fn codebook(mut self, table: TileArgLaunch) -> Result<Self, Refusal> {
        let Packing::Packed {
            field: Field::Index { bits },
        } = self.spec.packing
        else {
            return Err(Refusal::CodebookOfValuesNotIndices);
        };
        let ValuesLaunch::Tensor { tensor, width } = &table.values else {
            return Err(Refusal::CodebookTableNotPlain);
        };
        let plain = *width == 1
            && tensor.shape().len() == 1
            && table.spec.axes().len() == 1
            && table.spec.packing == Packing::Plain
            && table.scale.is_none()
            && table.index.is_none()
            && table.codebook.is_none();
        if !plain {
            return Err(Refusal::CodebookTableNotPlain);
        }
        let entries = tensor.shape()[0];
        if entries < 1 << bits {
            return Err(Refusal::CodebookTableShort { entries, bits });
        }
        self.codebook = Some(Box::new(table));
        Ok(self)
    }

    /// These values read through their projection's runtime half: a value for each of its
    /// dynamic coefficients and divisors (`Scale::Dynamic`, `Divisor::Dynamic`), physical axis
    /// major, each axis's divisor last, and for each of its dynamic offsets (`Offset::Dynamic`).
    /// Only how many there are is checked, as the kernel expands.
    pub fn runtime_map(mut self, coefficients: Vec<u32>, offsets: Vec<i32>) -> Self {
        self.runtime_map = (coefficients, offsets);
        self
    }

    /// This tensor map's cells read as the words `field` values are packed in, eight `e2m1` to
    /// a word: what a TMA engine moves of a packed operand, which a stage holds as it lies.
    /// `field` is read as stated: the field as the device reads it is
    /// [`Field::read_on`]'s.
    ///
    /// # Panics
    ///
    /// On an argument that is not a tensor map of `u32` words.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        assert!(
            matches!(self.values, ValuesLaunch::TensorMap { .. })
                && self.element == u32::elem_type_native(),
            "TileArgLaunch::packed: a tensor map of u32 words is read as packed values, and this \
             is not one"
        );
        Arc::make_mut(&mut self.spec).packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// These values under `scale`, outside the scales they carry already: words of block scales
    /// where `scale` is a plain level holding `u32`, scales read as numbers otherwise.
    pub fn scales(mut self, scale: TileArgLaunch) -> Self {
        self.scale = Some(Box::new(match self.scale {
            None => scale,
            Some(inner) => inner.scales(scale),
        }));
        self
    }

    /// These values with their windows placed by `table`, whose dims are its keys, as `indexing`
    /// says ([`Indexing`]). The scales they carry move with them wherever they span the placed
    /// axis; state them first, so they are checked here.
    ///
    /// A table keyed by the axis it places and one more, `[sequence, position]` placing the
    /// sequence, reads these values as a pool of pages: the dim the sequence labels holds pages,
    /// the one the position labels a page's positions, and sequence `b`'s position `s` lies in page
    /// `table[b, s / page_size]`, `page_size` the pool's extent along the position. The operand's
    /// extents along the two axes are the table's, checked on every read. The table is read as
    /// each window is cut, a window holding one page or two reading it once. The scales of a pool
    /// spanning either axis are a pool of their own, placed by the same table before they are
    /// stated.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] for a table that is not plain `u32` entries over one dim per key, or for
    /// values, or a scale level spanning the placed axis, whose windows an entry cannot place.
    pub fn indexed(self, table: TileArgLaunch, indexing: Indexing) -> Result<Self, Refusal> {
        self.placed_by(table, indexing, false)
    }

    /// [`indexed`](Self::indexed), the table a pool of pages is read through stated to cover the
    /// launch's extents along both its axes, so a read the launch keeps in bounds is in bounds
    /// through the table: the two axes are checked only where the launch overhangs, as a buffer
    /// read where it lies is. A table that stops short reads its clamped entries there: other
    /// positions of the pool, never past it.
    ///
    /// # Errors
    ///
    /// As [`indexed`](Self::indexed).
    pub fn indexed_covering(
        self,
        table: TileArgLaunch,
        indexing: Indexing,
    ) -> Result<Self, Refusal> {
        self.placed_by(table, indexing, true)
    }

    /// [`indexed`](Self::indexed), a pool's table stated to cover the launch where `covering`.
    fn placed_by(
        mut self,
        table: TileArgLaunch,
        indexing: Indexing,
        covering: bool,
    ) -> Result<Self, Refusal> {
        let ValuesLaunch::Tensor { tensor, width } = &table.values else {
            return Err(Refusal::IndexTableNotU32);
        };
        let plain_u32 = table.element == u32::elem_type_native()
            && *width == 1
            && table.spec.packing == Packing::Plain
            && table.scale.is_none()
            && table.index.is_none();
        if !plain_u32 {
            return Err(Refusal::IndexTableNotU32);
        }
        let keys = table.spec.axes();
        let rank = tensor.shape().len();
        if rank != keys.len() {
            return Err(Refusal::IndexTableRank {
                keys: keys.len(),
                rank,
            });
        }
        if let (Indexing::Coordinate(sequence), &[first, position]) = (&indexing, keys)
            && first == *sequence
        {
            self.spec = Arc::new(self.paged_spec(*sequence, position, covering)?);
            let mut level = self.scale.as_deref();
            while let Some(scale) = level {
                let spans = |axis| scale.spec.axes().contains(&axis);
                if (spans(first) || spans(position)) && scale.spec.paging.is_none() {
                    return Err(Refusal::ScalesOfAPoolUnpaged);
                }
                level = scale.scale.as_deref();
            }
            self.index = Some((Box::new(table), indexing));
            return Ok(self);
        }
        indexing.validate(&self.spec, self.indexed_width()?, keys)?;
        let mut level = self.scale.as_deref();
        while let Some(scale) = level {
            if scale.spec.axes().contains(&indexing.axis()) {
                indexing.places(&scale.spec, scale.indexed_width()?)?;
            }
            level = scale.scale.as_deref();
        }
        self.index = Some((Box::new(table), indexing));
        Ok(self)
    }

    /// This operand's spec read as a pool of pages along `sequence` and `position`: its paging, and
    /// unless the table `covering` the launch, the two axes checked on every read, their extents
    /// being the table's.
    fn paged_spec(
        &self,
        sequence: Axis,
        position: Axis,
        covering: bool,
    ) -> Result<TileSpec, Refusal> {
        let ValuesLaunch::Tensor { tensor, .. } = &self.values else {
            return Err(Refusal::PagingMovedByRuns);
        };
        let mut spec = TileSpec::clone(&self.spec);
        let paging = Paging::of_pool(
            sequence,
            position,
            tensor.shape(),
            &spec.projection,
            &spec.stored_tiles,
            spec.delivery,
        )?;
        if !covering {
            let boundary = spec.boundaries.iter().flatten().next().copied();
            if spec.boundaries.is_empty() {
                spec.boundaries = (0..spec.projection.coordinate_rank())
                    .map(|_| None)
                    .collect();
            }
            let (sequence_dim, position_dim) = paging.dims(&spec.projection)?;
            for dim in [sequence_dim, position_dim] {
                spec.boundaries[dim] = Some(boundary.unwrap_or(Boundary::Zero));
            }
        }
        spec.paging = Some(paging);
        Ok(spec)
    }

    /// The values a line of this buffer holds, where an entry may move its window: a tensor map
    /// lands boxes where they lie, and is never indexed.
    fn indexed_width(&self) -> Result<usize, Refusal> {
        match &self.values {
            ValuesLaunch::Tensor { width, .. } => Ok(*width),
            ValuesLaunch::TensorMap { .. } => Err(Refusal::IndexOfAStatedLayout),
        }
    }
}

/// What a [`TileArg`] compiles to: its values', its spec, its element and its scales'.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileArgCompilationArg {
    values: ValuesCompilationArg,
    spec: Arc<TileSpec>,
    element: ElemType,
    scale: Option<Box<TileArgCompilationArg>>,
    index: Option<(Box<TileArgCompilationArg>, Indexing)>,
    codebook: Option<Box<TileArgCompilationArg>>,
    /// How many dynamic coefficients and offsets the runtime map holds.
    runtime_map: (usize, usize),
}

/// What the values compile to: a buffer's and the width it is bound at, or a tensor map's layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ValuesCompilationArg {
    Tensor {
        tensor: TensorCompilationArg,
        width: usize,
    },
    TensorMap {
        layout: VirtualViewLayoutCompilationArg<CoordsDyn, Sequence<i32>>,
        width: usize,
    },
}

impl<'a> CubeType for TileArg<'a> {
    type ExpandType = TileArgExpand<'a>;
}

impl LaunchArg for TileArg<'static> {
    type RuntimeArg = TileArgLaunch;
    type CompilationArg = TileArgCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let TileArgLaunch {
            values,
            spec,
            element,
            scale,
            index,
            codebook,
            runtime_map: (coefficients, offsets),
        } = arg;
        let values = match values {
            ValuesLaunch::Tensor { tensor, width } => {
                let line_size = element.size() * width;
                let meta = TensorMetaLaunch::new(tensor.size() / width, tensor.shape().len());
                let buffer = match &tensor {
                    TensorArg::Handle { .. } => BufferCompilationArg { inplace: None },
                    TensorArg::Alias { input_pos, .. } => BufferCompilationArg {
                        inplace: Some(*input_pos),
                    },
                };
                launcher.register_tensor(tensor, line_size);
                let meta = TensorMeta::register(meta, launcher);
                ValuesCompilationArg::Tensor {
                    tensor: TensorCompilationArg { meta, buffer },
                    width,
                }
            }
            ValuesLaunch::TensorMap { map, layout, width } => {
                let layout = layout.register(&map, Type::new(element), launcher);
                launcher.register_tensor_map(*map, element.size());
                ValuesCompilationArg::TensorMap { layout, width }
            }
        };
        let scale = scale.map(|level| Box::new(TileArg::register(*level, launcher)));
        let index = index
            .map(|(table, indexing)| (Box::new(TileArg::register(*table, launcher)), indexing));
        let codebook = codebook.map(|table| Box::new(TileArg::register(*table, launcher)));
        for coefficient in &coefficients {
            coefficient.register(launcher);
        }
        for offset in &offsets {
            offset.register(launcher);
        }
        TileArgCompilationArg {
            values,
            spec,
            element,
            scale,
            index,
            codebook,
            runtime_map: (coefficients.len(), offsets.len()),
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TileArgExpand<'static> {
        let values = match &arg.values {
            ValuesCompilationArg::Tensor { tensor, width } => {
                let buffer = match tensor.buffer.inplace {
                    Some(id) => builder.inplace(id),
                    None => {
                        let scope = &builder.scope;
                        let held = arg.element.to_type(scope.ctx());
                        let line = match *width {
                            1 => held,
                            width => VectorType::get(scope.ctx(), held, width).into(),
                        };
                        builder.tensor(line)
                    }
                };
                let meta = TensorMeta::expand(&tensor.meta, builder);
                let scope = &builder.scope;
                let _held = scope.define_type::<Held>(arg.element);
                let len = expand_buffer_length_native(scope, buffer);
                let values = slice::from_raw_parts::<Held>(
                    scope,
                    buffer,
                    0usize.into_expand(scope),
                    len.into(),
                );
                ValuesExpand::Tensor {
                    tensor: scope
                        .create_kernel_ref(TensorExpand::__expand_from_parts(meta, values)),
                    buffer,
                    meta,
                    width: *width,
                }
            }
            // The map and its layout kept apart: a tile builds the view where a box moves, at the
            // element the map serves.
            ValuesCompilationArg::TensorMap { layout, width } => ValuesExpand::TensorMap {
                layout: layout.expand(Type::new(arg.element), builder),
                map: builder.tensor_map(),
                width: *width,
            },
        };
        let scale = arg
            .scale
            .as_ref()
            .map(|level| Box::new(TileArg::expand(level, builder)));
        let index = arg
            .index
            .as_ref()
            .map(|(table, indexing)| (Box::new(TileArg::expand(table, builder)), indexing.clone()));
        let codebook = arg
            .codebook
            .as_ref()
            .map(|table| Box::new(TileArg::expand(table, builder)));
        let (coefficients, offsets) = arg.runtime_map;
        let mut runtime_map = (
            Coords::<u32>::__expand_new(&builder.scope),
            Coords::<i32>::__expand_new(&builder.scope),
        );
        for _ in 0..coefficients {
            let coefficient = u32::expand_scalar(builder);
            runtime_map
                .0
                .__expand_push_method(&builder.scope, coefficient);
        }
        for _ in 0..offsets {
            let offset = i32::expand_scalar(builder);
            runtime_map.1.__expand_push_method(&builder.scope, offset);
        }
        TileArgExpand {
            values,
            spec: TileSpec::clone(&arg.spec),
            element: arg.element,
            scale,
            index,
            codebook,
            runtime_map,
        }
    }
}

impl IntoExpand for TileArgExpand<'_> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}

impl ExpandTypeClone for TileArgExpand<'_> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl IntoMut for TileArgExpand<'_> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl CubeDebug for TileArgExpand<'_> {}

impl AsRefExpand for TileArgExpand<'_> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl AsMutExpand for TileArgExpand<'_> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use cubecl::ir::FloatKind;

    use super::*;

    /// A one-axis buffer of `f32`, expanded as a kernel argument outside any launch.
    fn expanded_f32_buffer(builder: &mut KernelBuilder) -> TileArgExpand<'static> {
        let mut launcher = KernelLauncher::new(builder_settings());
        let meta = TensorMeta::register(TensorMetaLaunch::new(4, 1), &mut launcher);
        let arg = TileArgCompilationArg {
            values: ValuesCompilationArg::Tensor {
                tensor: TensorCompilationArg {
                    meta,
                    buffer: BufferCompilationArg { inplace: None },
                },
                width: 1,
            },
            spec: Arc::new(TileSpec::direct(&[Axis(0)])),
            element: ElemType::Float(FloatKind::F32),
            scale: None,
            index: None,
            codebook: None,
            runtime_map: (0, 0),
        };
        TileArg::expand(&arg, builder)
    }

    fn builder_settings() -> KernelSettings {
        KernelSettings::new(
            cubecl::ir::settings::Dim3::new_single(),
            cubecl::ir::settings::ExecutionMode::Checked,
            cubecl::ir::AddressType::U32,
        )
    }

    #[test]
    #[should_panic(expected = "TileArg::lines: the buffer is bound at f32, not i32")]
    fn lines_refuse_a_buffer_bound_at_another_element() {
        let mut builder = KernelBuilder::new(builder_settings());
        let arg = expanded_f32_buffer(&mut builder);
        arg.__expand_lines_method::<i32, Const<1>>(&builder.scope);
    }

    /// A `[4]` `u32` table keyed by `keys`, expanded as a kernel argument outside any launch.
    fn expanded_table(builder: &mut KernelBuilder, keys: &[Axis]) -> TileArgExpand<'static> {
        let mut launcher = KernelLauncher::new(builder_settings());
        let meta = TensorMeta::register(TensorMetaLaunch::new(4, keys.len()), &mut launcher);
        let arg = TileArgCompilationArg {
            values: ValuesCompilationArg::Tensor {
                tensor: TensorCompilationArg {
                    meta,
                    buffer: BufferCompilationArg { inplace: None },
                },
                width: 1,
            },
            spec: Arc::new(TileSpec::direct(keys)),
            element: u32::elem_type_native(),
            scale: None,
            index: None,
            codebook: None,
            runtime_map: (0, 0),
        };
        TileArg::expand(&arg, builder)
    }

    /// The one-axis `f32` buffer as a run keyed by `keys`, served on a bare scope under `levels`
    /// over `space`.
    fn serve_indexed(keys: &[Axis], space: &[(Axis, usize)], levels: Vec<Level>) {
        let mut builder = KernelBuilder::new(builder_settings());
        let mut values = expanded_f32_buffer(&mut builder);
        let table = expanded_table(&mut builder, keys);
        values.index = Some((Box::new(table), Indexing::Run(Axis(0))));
        let scope = &builder.scope;
        let partitioning = PartitioningExpand {
            space: SpaceExpand {
                shape: Space::new(space).shape().clone(),
                sizes: Sequence::<usize>::__expand_new(scope),
            },
            levels,
        };
        values.__expand_tile_method(scope, &partitioning);
    }

    /// The table is read at one step, the one that cuts its keys; keys cut at two levels name no
    /// such step, and the operand is refused as it is served.
    #[test]
    #[should_panic(
        expected = "the first level cutting one of [Axis(1), Axis(2)] does not cut them all"
    )]
    fn keys_cut_at_two_levels_are_refused_as_the_tile_expands() {
        serve_indexed(
            &[Axis(1), Axis(2)],
            &[(Axis(0), 4), (Axis(1), 2), (Axis(2), 2)],
            vec![Level::every(&[(Axis(1), 1)]), Level::every(&[(Axis(2), 1)])],
        );
    }

    /// No level cuts the key, so it is one tile, and the run is read at the table's first entry as
    /// the operand is served: a grouped product of a single group.
    #[test]
    fn a_table_no_level_cuts_is_read_as_the_tile_expands() {
        serve_indexed(
            &[Axis(1)],
            &[(Axis(0), 4), (Axis(1), 1)],
            vec![Level::every(&[(Axis(0), 2)])],
        );
    }

    #[test]
    fn lines_read_a_buffer_at_the_element_it_is_bound_at() {
        let mut builder = KernelBuilder::new(builder_settings());
        let arg = expanded_f32_buffer(&mut builder);
        arg.__expand_lines_method::<f32, Const<1>>(&builder.scope);
    }
}
