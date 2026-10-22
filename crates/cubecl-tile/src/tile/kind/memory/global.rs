//! A launched memory operand ([`GlobalOperand`]) and the [`Memory`] tile it becomes.

use cubecl::prelude::*;

use cubecl_std::tensor::{ErasedTensor, WriteOnly};

use crate::*;

/// A launched memory operand: a buffer, a fused sink or a fused producer, with its geometry.
#[derive(CubeType)]
pub struct GlobalOperand {
    pub(crate) backing: Backing,
    /// The buffer's physical extents and strides in scalars, one per physical axis.
    pub geometry: RuntimeGeometry,
    /// The binding's own line width, before any packing factor.
    #[cube(comptime)]
    pub bound_width: usize,
    /// The kernel's space.
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub spec: TileSpec,
    /// What a write does to the cell it lands on.
    #[cube(comptime)]
    pub write: Write,
    /// One per [`Scale::Dynamic`](crate::Scale) term and one per
    /// [`Divisor::Dynamic`](crate::Divisor) axis, by physical axis, divisor last.
    pub coefficients: Coords<u32>,
    /// One signed value per [`Offset::Dynamic`](crate::Offset) axis.
    pub offsets: Coords<i32>,
    /// The table a pool of pages is read as sequences through, where the spec states
    /// [`paging`](TileSpec::paging).
    pub(crate) paging: ComptimeOption<PageTable>,
}

#[cube]
impl GlobalOperand {
    /// A launched tensor, `space` projected onto the `spec` axes; shape and strides in scalars.
    pub fn tensor<T: Numeric, E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand {
        GlobalOperand::of_tensor::<T, E>(
            tensor,
            space,
            spec,
            Coords::<u32>::new(),
            Coords::<i32>::new(),
            ComptimeOption::new_None(),
        )
    }

    /// [`tensor`](GlobalOperand::tensor) over a pool of pages, read as the sequences `table` names
    /// ([`Paging`]): the spec states which axes the table resolves, the tensor is the pool as it
    /// lies, and the operand's extent along the two axes is the table's. A pool is never packed,
    /// so `T` is `E`'s scalar.
    pub fn paged<T: Numeric, E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        table: &Tensor<u32>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand {
        let paging = comptime!(spec.paging.unwrap_or_else(|| panic!(
            "GlobalOperand::paged: the spec states no paging for the table to resolve"
        )));
        let projection = comptime!(spec.projection.clone());
        let (sequence_dim, _) = comptime!(
            paging
                .dims(&projection)
                .unwrap_or_else(|refusal| panic!("GlobalOperand::paged: {refusal}"))
        );
        let table = PageTable::new(
            table,
            tensor.shape(sequence_dim) as u32,
            paging,
            projection,
            tensor.vector_size(),
        );
        GlobalOperand::of_tensor::<T, E>(
            tensor,
            space,
            spec,
            Coords::<u32>::new(),
            Coords::<i32>::new(),
            ComptimeOption::new_Some(table),
        )
    }

    /// [`tensor`](GlobalOperand::tensor) for a gather with a runtime affine map.
    /// `coefficients` and `offsets` follow the field order; only their lengths are checked.
    pub fn gathered<T: Numeric, E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> GlobalOperand {
        GlobalOperand::of_tensor::<T, E>(
            tensor,
            space,
            spec,
            coefficients,
            offsets,
            ComptimeOption::new_None(),
        )
    }

    /// The shared body: `E` is the binding element, `T` the served scalar, `paging` the table a
    /// pool of pages is read through, which sets the operand's extent along its two axes.
    fn of_tensor<T: Numeric, E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
        paging: ComptimeOption<PageTable>,
    ) -> GlobalOperand {
        let rank = comptime!(spec.projection.physical_rank());
        // Held at what the binding holds: `T`, or the `u32` words a packed one packs it in.
        let backing = match comptime!(spec.packing) {
            Packing::Plain => Backing::new_Buffer(Raw::<Slice>::of::<T>(unsafe {
                tensor
                    .as_slice()
                    .downcast_unchecked::<T>()
                    .as_boxed_unchecked()
            })),
            Packing::Packed { .. } => Backing::new_Buffer(Raw::<Slice>::of::<u32>(unsafe {
                tensor
                    .as_slice()
                    .downcast_unchecked::<u32>()
                    .as_boxed_unchecked()
            })),
        };
        let pool = RuntimeGeometry::of_tensor::<E>(tensor, rank);
        #[comptime]
        let geometry = match &paging {
            ComptimeOption::Some(table) => table.sequences_over(&pool),
            ComptimeOption::None => pool,
        };
        GlobalOperand {
            backing,
            geometry,
            bound_width: tensor.vector_size(),
            space,
            spec,
            write: comptime!(Write::Replace),
            coefficients,
            offsets,
            paging,
        }
    }

    /// An operand whose values are handed to `sink` instead of stored, under the stated geometry.
    /// Under [`Accumulate`](Write::Accumulate) the buffer must hold zero; under
    /// [`Exclusive`](Write::Exclusive), anything: the first turn replaces it.
    pub fn sink<T: Numeric>(
        sink: ErasedTensor<T, WriteOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        #[comptime] write: Write,
    ) -> GlobalOperand {
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "GlobalOperand::sink: a sink is written at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        GlobalOperand {
            backing: Backing::new_WriteCall(RawCall::of::<T>(sink)),
            geometry,
            bound_width: vector_size,
            space,
            spec,
            write,
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
            paging: ComptimeOption::new_None(),
        }
    }

    /// An operand whose values come from `source` instead of memory, under the stated geometry.
    pub fn source<T: Numeric>(
        source: ErasedTensor<T, ReadOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand {
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "GlobalOperand::source: a source is read at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        GlobalOperand {
            backing: Backing::new_ReadCall(RawCall::of::<T>(source)),
            geometry,
            bound_width: vector_size,
            space,
            spec,
            write: comptime!(Write::Replace),
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
            paging: ComptimeOption::new_None(),
        }
    }

    /// This operand as a [`Tile`] at the top of `levels`.
    pub fn tile<T: Numeric>(self, #[comptime] levels: Vec<Level>) -> Tile {
        let place = comptime!(Placement::root(
            self.space.subspace(self.spec.axes()),
            levels
        ));
        Tile::new::<T>(TileKind::new_Memory(Memory::global(self)), place)
    }
}

#[cube]
impl Memory {
    /// The memory tile a launched operand becomes, its top window boxed over the physical axes.
    // The unrolled loop over the buffer's dims indexes a compile-time list by the dim.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn global(operand: GlobalOperand) -> Memory {
        let backing = operand.backing;
        let geometry = operand.geometry;
        let paging = operand.paging;
        let coefficients = operand.coefficients;
        let offsets = operand.offsets;
        let bound_width = comptime!(operand.bound_width);
        let space = comptime!(operand.space.clone());
        let spec = comptime!(operand.spec.clone());
        let write = comptime!(operand.write);
        let split_share = comptime!(SplitShare::Whole);
        let space = comptime!(space.subspace(spec.axes()));
        let projection = comptime!(spec.projection.clone());
        let coords = comptime!(projection.untiled());
        let packing = comptime!(spec.packing);
        let vector_size = comptime!(packing.served(bound_width));
        let dims_given = geometry.shape.len();
        let coefficients_given = coefficients.len();
        let offsets_given = offsets.len();
        comptime!(assert!(
            dims_given == projection.physical_rank(),
            "GlobalOperand: the projection addresses {} physical dims but {dims_given} were given",
            projection.physical_rank()
        ));
        comptime!(assert!(
            coefficients_given == coords.dynamic_coefficient_count(),
            "GlobalOperand: the projection has {} Dynamic coefficients and divisors but \
             {coefficients_given} were given",
            coords.dynamic_coefficient_count()
        ));
        comptime!(assert!(
            offsets_given == coords.dynamic_offset_count(),
            "GlobalOperand: the projection has {} Dynamic offsets but {offsets_given} were given",
            coords.dynamic_offset_count()
        ));
        comptime!(check_operand(&space, &spec, vector_size));
        comptime!(assert!(
            paging.is_some() == spec.paging.is_some(),
            "GlobalOperand: a spec stating paging is read through its table \
             (GlobalOperand::paged), and only such a spec is"
        ));
        // Gathered buffers have fewer physical axes than logical; storage-tiled ones have more.
        let rank = comptime!(projection.physical_rank());
        // The buffer read in loads: each dim's extent over what one load covers of it, every
        // stride counted in loads. A plain buffer's innermost dim counts lines one line apart and
        // the coarser strides divide by the width; a stored tile a load takes whole is one.
        let load = comptime!(
            VectorTile::new(
                &spec.stored_tiles,
                space.axis_at(space.rank() - 1),
                vector_size
            )
            .unwrap_or_else(|why| panic!("GlobalOperand: {vector_size} values a load: {why}"))
        );
        let parts = comptime!(load.parts(&spec.stored_tiles, &projection.dense_labels(), rank));
        // A stated tile's extent is the statement's, not read off the buffer: the reads over it
        // divide by a constant.
        let stated = comptime!(VectorTile::stated_extents(
            &spec.stored_tiles,
            &projection.dense_labels(),
            rank
        ));
        let mut physical_shape = Coords::<u32>::new();
        let mut physical_strides = Coords::<u32>::new();
        #[unroll]
        for i in 0..rank {
            let part = comptime!(parts[i] as u32);
            if comptime!(stated[i].is_some()) {
                physical_shape.push(comptime!(stated[i].unwrap() as u32 / part));
            } else {
                physical_shape.push(geometry.shape.at(i) / part);
            }
            physical_strides.push(geometry.strides.at(i) * part / comptime!(vector_size as u32));
        }
        let gmem_projection = comptime!(projection.positional());
        // Folded from the physical shape, so tiled (padded) operands get the logical extent.
        let bound = logical_extent(comptime!(gmem_projection.clone()), &physical_shape);
        let (origin, extent, map) = top_window(
            comptime!(space.clone()),
            &bound,
            &offsets,
            coefficients,
            comptime!(load.clone()),
            comptime!(coords.clone()),
        );
        Memory {
            address: comptime!(AddressSpace::Global),
            store: Store {
                backing,
                vector_size: comptime!(vector_size),
                packing: comptime!(packing),
                stored_tiles: comptime!(spec.stored_tiles.clone()),
            },
            layout: BufferLayout {
                physical_shape,
                physical_strides,
                projection: gmem_projection,
                rows: RowArrangement::InOrder,
                paging,
            },
            window: Window::new(
                origin,
                extent,
                bound,
                comptime!(coords.may_underflow()),
                comptime!(spec.boundaries.clone()),
            ),
            source_window: ComptimeOption::new_None(),
            projection: comptime!(coords),
            map,
            offsets,
            window_start: 0u32,
            access: comptime!(Access {
                whole: true,
                overhang: if spec.is_checked() {
                    Overhang::Masked
                } else {
                    Overhang::Fits
                },
                write,
                fill: FillUnits::cube(spec.units),
                storage: WindowStorage::Stored(spec.storage),
                delivery: spec.delivery,
            }),
            unit_share: comptime!(UnitShare::Repeated),
            split_share,
            init_from: comptime!(InitFrom::Cell),
            contraction: comptime!(None),
            codebook: Codebook::none(),
            index: ComptimeOption::new_None(),
        }
    }
}

/// The contract an operand's statement owes, checked on the host.
fn check_operand(space: &Space, spec: &TileSpec, vector_size: usize) {
    let projection = &spec.projection;
    // At expansion no caller takes a refusal; a spec stated by hand, not bound, arrives here
    // unchecked.
    if let Err(refusal) = projection.validate(vector_size) {
        panic!("{refusal}");
    }
    projection.validate_composition(|axis| space.extent(axis));
    if let Some(paging) = &spec.paging {
        let checked = paging.validate(
            projection,
            &spec.stored_tiles,
            spec.packing,
            spec.delivery,
            vector_size,
        );
        if let Err(refusal) = checked {
            panic!("GlobalOperand: {refusal}");
        }
    }
    let coord_rank = projection.coordinate_rank();
    assert!(
        spec.boundaries.is_empty() || spec.boundaries.len() == coord_rank,
        "GlobalOperand: boundaries rank ({}) does not match coordinate rank ({coord_rank})",
        spec.boundaries.len()
    );
    assert!(
        vector_size == 1 || spec.boundaries.last().copied().flatten() != Some(Boundary::Clamp),
        "GlobalOperand: Boundary::Clamp cannot clamp the vectorized innermost axis (served at \
         {vector_size})"
    );
}

/// [`full_window`] for the top gmem tile over the physical axes; `Dynamic` axes read `bound`.
#[cube]
fn top_window(
    #[comptime] space: Space,
    bound: &Coords<u32>,
    offsets: &Coords<i32>,
    coefficients: Coords<u32>,
    #[comptime] load: VectorTile,
    #[comptime] projection: Projection,
) -> (Coords<i32>, Coords<u32>, RuntimeMap) {
    let mut origin = Coords::<i32>::new();
    let mut extent = Coords::<u32>::new();
    let mut residues = Coords::<u32>::new();
    let rank = comptime!(projection.physical_rank());

    #[unroll]
    for pa in 0..rank {
        let size = if comptime!(projection.is_direct()) {
            origin.push(0);
            residues.push(0u32);
            let axis = comptime!(space.axis_at(pa));
            match comptime!(space.extent_raw(axis)) {
                Extent::Static(e) => (comptime!(e / load.extent_along(axis)) as u32).runtime(),
                Extent::Dynamic => bound.at(pa),
            }
        } else {
            let (start, phase) =
                gathered_origin(comptime!(projection.clone()), offsets, &coefficients, pa);
            origin.push(start);
            residues.push(phase);
            bound.at(pa)
        };
        extent.push(size);
    }

    (
        origin,
        extent,
        RuntimeMap {
            coefficients,
            residues,
        },
    )
}

/// Where a gathered physical axis's top window starts, and the phase its division left behind.
#[cube]
fn gathered_origin(
    #[comptime] projection: Projection,
    offsets: &Coords<i32>,
    coefficients: &Coords<u32>,
    #[comptime] pa: usize,
) -> (i32, u32) {
    let axis_map = comptime!(projection.physical_axis(pa));

    if comptime!(axis_map.origin().is_some()) {
        (
            comptime!(axis_map.origin().unwrap() as i32).runtime(),
            comptime!(axis_map.residue().unwrap() as u32).runtime(),
        )
    } else {
        // Floor division: a negative (padding) offset must not round toward zero.
        let offset = match comptime!(axis_map.offset()) {
            Offset::Static(o) => comptime!(o as i32).runtime(),
            Offset::Dynamic => offsets.at(comptime!(projection.dynamic_offset_index(pa).unwrap())),
        };
        if comptime!(!axis_map.is_rational()) {
            (offset, 0u32)
        } else {
            let divisor = match comptime!(axis_map.divisor()) {
                Divisor::Static(d) => comptime!(d as i32).runtime(),
                Divisor::Dynamic { .. } => coefficients
                    .at(comptime!(projection.dynamic_divisor_index(pa).unwrap()))
                    .cast::<i32>(),
            };
            let (start, residue) = floor_div_rem(offset, divisor);
            (start, residue.cast::<u32>())
        }
    }
}
