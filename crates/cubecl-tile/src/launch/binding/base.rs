//! The [`Arg`] builder, which turns a launched tensor into an operand.

use core::marker::PhantomData;

use cubecl::zspace::SmallVec;

use cubecl::ir::ElemType;
use cubecl::prelude::*;

use super::{boundaries::Boundaries, labels::Labels};
use crate::BoundaryPolicy;
use crate::{
    Axis, Delivery, Extent, Field, Geometry, Launcher, LineMisfit, Packing, Paging, Projection,
    Refusal, Space, Storage, StoragePartitioning, TileArgLaunch, TileSpec, VectorTile,
};

/// Typestate marker: the operand's axes are not yet stated.
pub struct Unlabelled;
/// Typestate marker: the operand's axes are stated.
pub struct Labelled;

/// What the builder accumulates.
struct ArgData<'a> {
    launch: &'a Launcher,
    /// The tensor this operand is read from and the element its buffer holds; `None` for a
    /// destination with no address.
    binding: Option<(TensorBinding, ElemType)>,
    geometry: Geometry,
    axes: &'a [Axis],
    batches: &'a [Axis],
    /// The operand's own affine mapping; `None` derives it from the labelled dims.
    projection: Option<Projection>,
    width: usize,
    /// The stated policy; `None` derives one from the overhangs.
    boundary: Option<BoundaryPolicy>,
    packing: Packing,
    /// Who moves the operand into a stage ([`delivery`](Arg::delivery)).
    delivery: Delivery,
    /// Whether the labelled dims bind in stride order ([`in_stride_order`](Arg::in_stride_order)).
    in_stride_order: bool,
    /// The pool of pages this operand is read through, if any ([`paged`](Arg::paged)).
    paged: Option<PagedPool>,
}

/// What [`Arg::paged`] states: the axes a table resolves, and whether it covers the launch
/// ([`Arg::paged_covering`]).
#[derive(Clone, Copy)]
struct PagedPool {
    sequence: Axis,
    position: Axis,
    covering: bool,
}

/// One operand of a launch, being described; `S` says whether its axes are stated yet.
pub struct Arg<'a, S> {
    data: ArgData<'a>,
    _state: PhantomData<S>,
}

impl<'a> Arg<'a, Unlabelled> {
    pub(crate) fn bound(launch: &'a Launcher, binding: TensorBinding, element: ElemType) -> Self {
        let geometry = Geometry::from(&binding);
        Self::over(launch, Some((binding, element)), geometry)
    }

    pub(crate) fn unbound(launch: &'a Launcher, geometry: &Geometry) -> Self {
        Self::over(launch, None, geometry.clone())
    }

    fn over(
        launch: &'a Launcher,
        binding: Option<(TensorBinding, ElemType)>,
        geometry: Geometry,
    ) -> Self {
        Arg {
            data: ArgData {
                launch,
                binding,
                geometry,
                axes: &[],
                batches: &[],
                projection: None,
                width: 1,
                boundary: None,
                packing: Packing::Plain,
                delivery: Delivery::SyncPerUnit,
                in_stride_order: false,
                paged: None,
            },
            _state: PhantomData,
        }
    }

    /// The axes the operand's trailing buffer dims carry.
    pub fn axes(mut self, axes: &'a [Axis]) -> Arg<'a, Labelled> {
        self.data.axes = axes;
        self.labelled()
    }

    /// An explicit affine [`Projection`] for a gathered operand. Overruns past the buffer's tail
    /// are not detected; a gather that overruns must state [`BoundaryPolicy::Every`].
    pub fn gathered(mut self, projection: Projection) -> Arg<'a, Labelled> {
        self.data.projection = Some(projection);
        self.labelled()
    }

    fn labelled(self) -> Arg<'a, Labelled> {
        Arg {
            data: self.data,
            _state: PhantomData,
        }
    }
}

impl<'a> Arg<'a, Labelled> {
    /// The batch axes in the output's order, right-aligned to the leading dims (numpy broadcast).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batches = axes;
        self
    }

    /// Bind the labelled dims in the order they step, coarsest first; batch dims stay put.
    pub fn in_stride_order(mut self) -> Self {
        self.data.in_stride_order = true;
        self
    }

    /// Serve the innermost axis in `width`-wide lines; that axis must be contiguous.
    pub fn vectorize(mut self, width: usize) -> Self {
        self.data.width = width;
        self
    }

    /// How this operand's edges are checked.
    pub fn boundary(mut self, policy: BoundaryPolicy) -> Self {
        self.data.boundary = Some(policy);
        self
    }

    /// This operand's values are `field`-wide fields of a stored word.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.data.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// Who moves this operand into a stage ([`SyncPerUnit`](Delivery::SyncPerUnit) unless
    /// stated); the kernel is the same either way.
    pub fn delivery(mut self, delivery: Delivery) -> Self {
        self.data.delivery = delivery;
        self
    }

    /// Read this buffer, a pool of pages along the `sequence` dim each holding its positions along
    /// the `position` dim, as the sequences a page table names: sequence `b`'s position `s` lies in
    /// page `table[b, s / page_size]`, `page_size` the pool's extent along `position`. The pool is
    /// bound as it lies; the kernel reads it through
    /// [`TileArg::tile_paged`](crate::TileArg::tile_paged) with the table, whose shape is the
    /// operand's extent along the two axes, so both are checked against it on every read.
    pub fn paged(self, sequence: Axis, position: Axis) -> Self {
        self.paged_as(PagedPool {
            sequence,
            position,
            covering: false,
        })
    }

    /// [`paged`](Self::paged) through a table the caller states covers the launch's extents
    /// along both axes, so a read the launch keeps in bounds is in bounds through the table: the
    /// two axes are checked only where the launch overhangs, as a buffer read where it lies is.
    /// A table that stops short reads its clamped entries there: other positions of the pool,
    /// never past it. Where the launch overhangs a table axis its lines run along, wider than
    /// one, the binding is refused ([`Refusal::UncheckableVectorEdge`]), not checked line by line.
    pub fn paged_covering(self, sequence: Axis, position: Axis) -> Self {
        self.paged_as(PagedPool {
            sequence,
            position,
            covering: true,
        })
    }

    fn paged_as(mut self, pool: PagedPool) -> Self {
        self.data.paged = Some(pool);
        self
    }

    /// The operand, bound, or the [`Refusal`] saying why it cannot be.
    pub fn build(self) -> Result<Bound, Refusal> {
        self.realize().map(|(bound, _)| bound)
    }

    /// [`build`](Self::build) without the tensor argument, for an operand with no address.
    pub fn build_spec(self) -> Result<Unbound, Refusal> {
        let (bound, geometry) = self.realize()?;
        Ok(Unbound {
            spec: bound.spec,
            vector_size: bound.vector_size,
            geometry,
        })
    }

    /// The derivation both builds share.
    fn realize(self) -> Result<(Bound, Geometry), Refusal> {
        let data = self.data;
        let launch = data.launch;
        let width = data.width;
        let stored = data
            .binding
            .as_ref()
            .map(|(b, _)| b.tiling)
            .unwrap_or_default();
        let stated = data.projection.is_some() || stored.is_tiled();
        let (geometry, axes) =
            stride_ordered(data.geometry, data.axes, data.in_stride_order, stated)?;
        let (storage, stored_tiles) =
            storage_of(&geometry, &axes, data.projection.is_none(), launch);
        let labels = match data.projection {
            Some(projection) => {
                Labels::stated(&geometry, launch, projection, &axes, data.batches, stored)?
            }
            None => Labels::new(&geometry, &axes, data.batches)?,
        };
        let Labels {
            geometry,
            projection,
            addressed,
        } = labels;
        projection.validate(width)?;
        let packing = data.packing.read_on(launch.client());
        // After the labels settle, so the dims are the ones the kernel addresses, stride order
        // included.
        let paging = data
            .paged
            .map(|pool| {
                pool.paging(
                    &geometry,
                    &projection,
                    &stored_tiles,
                    packing,
                    data.delivery,
                    width,
                )
            })
            .transpose()?;
        // A stated width is checked here: `stride / width` would truncate silently.
        // A line runs along the innermost dim's axis; a buffer with no labelled dim has none.
        let labels = projection.dense_labels();
        let served = match labels.last() {
            Some(&axis) => VectorTile::new(&stored_tiles, axis, width)
                .map_err(LineMisfit::Tile)
                .and_then(|tile| geometry.serves(tile.extents(), &labels)),
            None if width == 1 => Ok(()),
            None => Err(LineMisfit::NoDims),
        };
        served.map_err(|why| Refusal::WidthNotServed { width, why })?;
        let overhangs = launch.overhangs();
        let table_bounded: Vec<Axis> = match data.paged.is_some_and(|pool| pool.covering) {
            true => Vec::new(),
            false => paging
                .iter()
                .flat_map(|p| [p.sequence, p.position])
                .collect(),
        };
        let boundaries = Boundaries::new(
            data.boundary,
            &projection,
            &addressed,
            launch.space(),
            overhangs,
            &table_bounded,
            width,
        )?;
        let spec = TileSpec {
            projection,
            boundaries: boundaries.modes,
            units: launch.cube_dim().num_elems() as usize,
            packing,
            storage,
            stored_tiles,
            delivery: Delivery::SyncPerUnit,
            paging,
        }
        .delivery(data.delivery);
        let tensor = data
            .binding
            .map(|(binding, element)| (settled_tensor(binding, &geometry), element));
        let bound = Bound {
            tensor,
            vector_size: width,
            spec,
        };
        Ok((bound, geometry))
    }
}

/// A bound operand: its tensor argument and the element its buffer holds, comptime [`TileSpec`] and
/// served width.
pub struct Bound {
    tensor: Option<(TensorArg, ElemType)>,
    /// Served width (values per line); a packed binding is narrower by the packing factor.
    pub vector_size: usize,
    pub spec: TileSpec,
}

impl Bound {
    /// The operand as the kernel's [`TileArg`](crate::TileArg) launch argument.
    pub fn arg(self) -> TileArgLaunch {
        let width = self.bound_width();
        let (tensor, element) = self
            .tensor
            .expect("Bound::arg: this operand was built over geometry alone and has no tensor");
        TileArgLaunch::new(tensor, element, width, self.spec)
    }

    /// The tensor argument itself; panics for an operand built over geometry alone.
    pub fn tensor(self) -> TensorArg {
        self.tensor
            .expect("Bound::tensor: this operand was built over geometry alone and has no tensor")
            .0
    }

    /// The width the binding is declared at, narrower than `vector_size` when packed: a packed
    /// binding served fewer values a line than a word holds is still read a whole word at a time.
    pub fn bound_width(&self) -> usize {
        self.spec.packing.physical(self.vector_size).max(1)
    }
}

/// What [`build_spec`](Arg::build_spec) settles for an operand with no tensor to bind.
pub struct Unbound {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}

/// The labelled dims in stride order where asked; refused for a stated layout.
fn stride_ordered(
    geometry: Geometry,
    axes: &[Axis],
    in_stride_order: bool,
    stated: bool,
) -> Result<(Geometry, Vec<Axis>), Refusal> {
    match in_stride_order {
        true if stated => Err(Refusal::StrideOrderOfAStatedLayout),
        true => Ok(geometry.in_stride_order(axes)),
        false => Ok((geometry, axes.to_vec())),
    }
}

/// One level's window along an operand's axes, in their order: the extent the launch cuts each axis
/// to, and whether the kernel hands it down dynamic.
type LevelWindow = SmallVec<[(Extent, bool); Space::MAX_RANK]>;

/// The coarsest level whose windows the operand's storage tiles address with one stride per axis
/// (one contiguous run each), every other window walked through the layout; and the
/// stored tiles themselves, finest first ([`TileSpec::stored_tiles`]).
fn storage_of(
    geometry: &Geometry,
    axes: &[Axis],
    labelled: bool,
    launch: &Launcher,
) -> (Storage, Vec<(Axis, usize)>) {
    if !(labelled && geometry.tiling().is_tiled()) {
        return (Storage::Strided, Vec::new());
    }
    let labels = geometry.labels(axes);
    let partitioning = StoragePartitioning::new(geometry, &labels);
    let stored_tiles = partitioning
        .as_ref()
        .map_or_else(Vec::new, |storage| storage.tiles().to_vec());
    let tiles = partitioning
        .map(|storage| storage.contiguous_tiles(&geometry.extents(&labels)))
        .unwrap_or_default();
    // Each level's window along the operand's axes, outermost first: the extent this launch cuts
    // it to, and whether the kernel hands it down dynamic. A level that cuts an axis gives it its
    // tile; one that does not passes the window above it down.
    let (space, kernel) = (launch.space(), launch.partitioning().space());
    let mut window: LevelWindow = axes
        .iter()
        .map(|&axis| (space.extent_raw(axis), kernel.is_dynamic(axis)))
        .collect();
    let windows: SmallVec<[LevelWindow; 8]> = launch
        .partitioning()
        .levels()
        .iter()
        .map(|level| {
            for (&axis, cut) in axes.iter().zip(window.iter_mut()) {
                if let Some(tile) = level.tile(axis) {
                    *cut = (Extent::Static(tile), false);
                }
            }
            window.clone()
        })
        .collect();
    // A tile the kernel hands down dynamic is no window a storage tile can be, whatever this
    // launch's extent makes it.
    let cuts_to = |window: &[(Extent, bool)], tile: &[(Axis, usize)]| {
        axes.iter().zip(window).all(|(&axis, &(extent, dynamic))| {
            let stored = tile
                .iter()
                .find(|&&(a, _)| a == axis)
                .map_or(1, |&(_, e)| e);
            !dynamic && extent == Extent::Static(stored)
        })
    };
    let level = tiles
        .iter()
        .rev()
        .find_map(|tile| windows.iter().position(|window| cuts_to(window, tile)));
    (Storage::Tiled(level), stored_tiles)
}

/// The binding as the arg ships it, its tiling restated over the settled geometry's dims.
fn settled_tensor(mut binding: TensorBinding, geometry: &Geometry) -> TensorArg {
    binding.shape = geometry.shape().into();
    binding.strides = geometry.strides().into();
    binding.tiling = geometry.tiling();
    binding.into_tensor_arg()
}

impl PagedPool {
    /// The [`Paging`] this pool is read through, its page the pool's extent along `position`, over
    /// the settled `geometry` and `projection`.
    ///
    /// # Errors
    ///
    /// What [`Paging::validate`] refuses, here so a candidate that cannot serve is skipped rather
    /// than panicking as the kernel expands, and a pool of no pages.
    fn paging(
        self,
        geometry: &Geometry,
        projection: &Projection,
        stored_tiles: &[(Axis, usize)],
        packing: Packing,
        delivery: Delivery,
        width: usize,
    ) -> Result<Paging, Refusal> {
        let unsized_paging = Paging {
            sequence: self.sequence,
            position: self.position,
            page_size: 0,
        };
        // Where the axes name no dims, `validate` refuses before it reads the page.
        let page_size = unsized_paging
            .dims(projection)
            .map_or(0, |(_, position)| geometry.shape()[position]);
        let paging = Paging {
            page_size,
            ..unsized_paging
        };
        paging.validate(projection, stored_tiles, packing, delivery, width)?;
        let (sequence, _) = paging.dims(projection)?;
        match geometry.shape()[sequence] {
            0 => Err(Refusal::PagingOfAnEmptyPool),
            _ => Ok(paging),
        }
    }
}
