//! The [`Arg`] builder, which turns a launched tensor into an operand.

use core::marker::PhantomData;

use cubecl::zspace::{SmallVec, Tiling};

use cubecl::ir::ElemType;
use cubecl::prelude::*;

use super::{boundaries::Boundaries, derivation::Derivation, labels::Labels};
use crate::BoundaryPolicy;
use crate::{
    Decline, Delivery, Dimension, Extent, Field, Geometry, Launcher, LineMisfit, Packing,
    Projection, Space, Storage, StoragePartitioning, TileSpec, TileTensorLaunch, VectorTile,
};

/// Typestate marker: the operand's dimensions are not yet stated.
pub struct Unlabelled;
/// Typestate marker: the operand's dimensions are stated.
pub struct Labelled;

/// What the builder accumulates.
pub(crate) struct ArgData<'a> {
    pub(crate) launch: &'a Launcher,
    /// The tensor this operand is read from and the element its buffer holds; `None` for a
    /// destination with no address.
    pub(crate) binding: Option<(TensorBinding, ElemType)>,
    pub(crate) geometry: Geometry,
    pub(crate) dimensions: &'a [Dimension],
    pub(crate) batches: &'a [Dimension],
    /// The operand's own affine mapping; `None` derives it from the labelled axes.
    pub(crate) projection: Option<Projection>,
    pub(crate) width: usize,
    /// The stated policy; `None` derives one from the overhangs.
    pub(crate) boundary: Option<BoundaryPolicy>,
    pub(crate) packing: Packing,
    /// Who moves the operand into a stage ([`delivery`](Arg::delivery)).
    pub(crate) delivery: Delivery,
    /// Whether the labelled axes bind in stride order ([`in_stride_order`](Arg::in_stride_order)).
    pub(crate) in_stride_order: bool,
}

/// One operand of a launch, being described; `S` says whether its dimensions are stated yet.
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
                dimensions: &[],
                batches: &[],
                projection: None,
                width: 1,
                boundary: None,
                packing: Packing::Plain,
                delivery: Delivery::SyncPerUnit,
                in_stride_order: false,
            },
            _state: PhantomData,
        }
    }

    /// The dimensions the operand's trailing buffer axes carry.
    pub fn dimensions(mut self, dimensions: &'a [Dimension]) -> Arg<'a, Labelled> {
        self.data.dimensions = dimensions;
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
    /// The batch dimensions in the output's order, right-aligned to the leading axes (numpy
    /// broadcast).
    pub fn batches(mut self, dimensions: &'a [Dimension]) -> Self {
        self.data.batches = dimensions;
        self
    }

    /// Bind the labelled axes in the order they step, coarsest first; batch axes stay put.
    pub fn in_stride_order(mut self) -> Self {
        self.data.in_stride_order = true;
        self
    }

    /// Serve the innermost dimension in `width`-wide lines; that dimension must be contiguous.
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

    /// This operand's values are `u32` words of `field`-wide values, each word read whole: block
    /// scales as a block-scaled instruction takes them.
    pub fn words(mut self, field: impl Into<Field>) -> Self {
        self.data.packing = Packing::Words {
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

    /// The operand, bound, or the [`Decline`] saying why it cannot be.
    ///
    /// # Errors
    ///
    /// The [`Decline`] of a binding this launch cannot serve.
    ///
    /// # Panics
    ///
    /// Where a [`gathered`](Arg::gathered) operand also states [`batches`](Arg::batches), or its
    /// mapping spans a dimension the launched space does not have.
    pub fn build(self) -> Result<Bound, Decline> {
        self.realize().map(|(bound, _)| bound)
    }

    /// [`build`](Self::build) without the tensor argument, for an operand with no address.
    ///
    /// # Errors
    ///
    /// As [`build`](Self::build).
    ///
    /// # Panics
    ///
    /// As [`build`](Self::build).
    pub fn build_spec(self) -> Result<Unbound, Decline> {
        let (bound, geometry) = self.realize()?;
        Ok(Unbound {
            spec: bound.spec,
            vector_size: bound.vector_size,
            geometry,
        })
    }

    /// The derivation both builds share, looked up where an earlier launch already made it
    /// ([`Derivation`]): it reads nothing a launch changes but the tensor's address.
    fn realize(self) -> Result<(Bound, Geometry), Decline> {
        let packing = self.data.packing.read_on(self.data.launch.client());
        let stored = self
            .data
            .binding
            .as_ref()
            .map(|(b, _)| b.tiling)
            .unwrap_or_default();
        let derived = Derivation::view(&self.data, packing, stored)
            .derived(|| self.derive(packing, stored))?;
        let tensor = self.data.binding.map(|(mut binding, element)| {
            // Its tiling restated over the settled geometry's axes.
            binding.shape = derived.geometry.shape().into();
            binding.strides = derived.geometry.strides().into();
            binding.tiling = derived.geometry.tiling();
            (binding.into_tensor_arg(), element)
        });
        let bound = Bound {
            tensor,
            vector_size: self.data.width,
            spec: derived.spec,
        };
        Ok((bound, derived.geometry))
    }

    /// The derivation itself, its packing read on the device as `packing` and its buffer stored as
    /// `stored`.
    fn derive(&self, packing: Packing, stored: Tiling) -> Result<Derived, Decline> {
        let data = &self.data;
        let launch = data.launch;
        let width = data.width;
        let stated = data.projection.is_some() || stored.is_tiled();
        let (geometry, dimensions) = match data.in_stride_order {
            true if stated => return Err(Decline::StrideOrderOfAStatedLayout),
            true => data.geometry.clone().in_stride_order(data.dimensions),
            false => (data.geometry.clone(), data.dimensions.to_vec()),
        };
        let (storage, stored_tiles) = self.storage(&geometry, &dimensions);
        let labels = match data.projection.clone() {
            Some(projection) => Labels::stated(
                &geometry,
                launch,
                projection,
                &dimensions,
                data.batches,
                stored,
            )?,
            None => Labels::new(&geometry, &dimensions, data.batches)?,
        };
        let Labels {
            geometry,
            projection,
            addressed,
        } = labels;
        if data.projection.is_some() && !stored.is_tiled() {
            projection.dimensions_in_axis_order(width)?;
        }
        projection.validate(width)?;
        // A stated width is checked here: `stride / width` would truncate silently. A line runs
        // along the innermost axis's dimension; a buffer with no labelled axis has none.
        let labels = projection.dense_labels();
        let served = match labels.last() {
            Some(&dimension) => VectorTile::new(&stored_tiles, dimension, width)
                .map_err(LineMisfit::Tile)
                .and_then(|tile| geometry.serves(tile.extents(), &labels)),
            None if width == 1 => Ok(()),
            None => Err(LineMisfit::NoAxes),
        };
        served.map_err(|why| Decline::WidthNotServed { width, why })?;
        let may_overhang;
        let overhangs = match data.boundary {
            Some(BoundaryPolicy::AnyExtent(_)) => {
                may_overhang = launch.may_overhang();
                &may_overhang
            }
            _ => launch.overhangs(),
        };
        let boundaries = Boundaries::new(
            data.boundary,
            &projection,
            &addressed,
            launch.space(),
            overhangs,
            width,
        )?;
        for &boundary in boundaries.modes.iter().flatten() {
            data.delivery.reads_edge(boundary)?;
        }
        // Inside its storage tile a window is one unmasked run: a checked operand whose launch
        // reaches past its padding would read storage tiles it does not have, which the kernel
        // can only find out as it expands.
        if let Storage::Tiled(Some(level)) = storage
            && !boundaries.modes.is_empty()
            && self.reaches_past_storage_tiles(&dimensions, level)
        {
            return Err(Decline::CheckedInsideStorageTile { level });
        }
        let spec = TileSpec {
            projection,
            boundaries: boundaries.modes,
            units: launch.cube_dim().num_elems() as usize,
            packing,
            storage,
            stored_tiles,
            delivery: Delivery::SyncPerUnit,
            paging: None,
        }
        .delivery(data.delivery);
        Ok(Derived { spec, geometry })
    }

    /// The coarsest level whose windows the operand's storage tiles address with one stride per
    /// dimension (one contiguous run each), every other window walked through the layout; and the
    /// stored tiles themselves, finest first ([`TileSpec::stored_tiles`]).
    fn storage(
        &self,
        geometry: &Geometry,
        dimensions: &[Dimension],
    ) -> (Storage, Vec<(Dimension, usize)>) {
        let launch = self.data.launch;
        if !(self.data.projection.is_none() && geometry.tiling().is_tiled()) {
            return (Storage::Strided, Vec::new());
        }
        let labels = geometry.labels(dimensions);
        let partitioning = StoragePartitioning::new(geometry, &labels);
        let stored_tiles = partitioning
            .as_ref()
            .map_or_else(Vec::new, |storage| storage.tiles().to_vec());
        // A staggered buffer reads back as its dense statement: its strides say where a run ends.
        let dense_run = geometry.dense_run();
        let mut tiles = partitioning
            .map(|storage| storage.contiguous_tiles(&geometry.extents(&labels)))
            .unwrap_or_default();
        tiles.retain(|tile| tile.iter().map(|&(_, extent)| extent).product::<usize>() <= dense_run);
        // Each level's window along the operand's dimensions, outermost first: the extent this
        // launch cuts it to, and whether the kernel hands it down dynamic. A level that cuts a
        // dimension gives it its tile; one that does not passes the window above it down.
        let (space, kernel) = (launch.space(), launch.partitioning().space());
        let mut window: LevelWindow = dimensions
            .iter()
            .map(|&dimension| (space.extent_raw(dimension), kernel.is_dynamic(dimension)))
            .collect();
        let windows: SmallVec<[LevelWindow; 8]> = launch
            .partitioning()
            .levels()
            .iter()
            .map(|level| {
                for (&dimension, cut) in dimensions.iter().zip(window.iter_mut()) {
                    if let Some(tile) = level.tile(dimension) {
                        *cut = (Extent::Static(tile), false);
                    }
                }
                window.clone()
            })
            .collect();
        // A tile the kernel hands down dynamic is no window a storage tile can be, whatever this
        // launch's extent makes it.
        let cuts_to = |window: &[(Extent, bool)], tile: &[(Dimension, usize)]| {
            dimensions
                .iter()
                .zip(window)
                .all(|(&dimension, &(extent, dynamic))| {
                    let stored = tile
                        .iter()
                        .find(|&&(a, _)| a == dimension)
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

    /// Whether a window of `level`, the storage tile's, lies wholly past the operand along one of
    /// `dimensions`: the launch's outermost tile reaches past the extent padded to whole storage tiles,
    /// where no storage tile is allocated, or a level down to it is launched in an order that
    /// starts planes past the last of its tiles ([`PlaneOrder::overruns`]), however evenly its
    /// tile divides the extent. A window that only overhangs the extent reads padding.
    ///
    /// [`PlaneOrder::overruns`]: crate::space::PlaneOrder
    fn reaches_past_storage_tiles(&self, dimensions: &[Dimension], level: usize) -> bool {
        let launch = self.data.launch;
        let levels = launch.partitioning().levels();
        let mut space = launch.space().clone();
        let overruns = levels[..=level].iter().any(|level| {
            let overruns = level.launch_order().is_some_and(|order| {
                dimensions.iter().any(|&dimension| {
                    level.tile(dimension).is_some() && order.overruns(level, &space, dimension)
                })
            });
            space = level.child(&space);
            overruns
        });
        overruns
            || dimensions.iter().any(|&dimension| {
                // The outermost level cutting the dimension tiles the grid; the last one down to
                // `level` is the storage tile's extent along it. No cut leaves one window, the
                // whole extent.
                let mut cuts = levels[..=level]
                    .iter()
                    .filter_map(|level| level.tile(dimension));
                let Some(outer) = cuts.next() else {
                    return false;
                };
                let stored = cuts.next_back().unwrap_or(outer);
                let extent = launch.space().extent(dimension);
                extent.next_multiple_of(outer) > extent.next_multiple_of(stored)
            })
    }
}

/// What [`Arg::realize`] derives from an operand's labels and its launch: the comptime spec and the
/// geometry the tensor is bound at, stride order settled.
#[derive(Clone)]
pub(crate) struct Derived {
    pub(crate) spec: TileSpec,
    pub(crate) geometry: Geometry,
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
    /// The operand as the kernel's [`TileTensor`](crate::TileTensor) launch argument.
    pub fn arg(self) -> TileTensorLaunch {
        let width = self.bound_width();
        let (tensor, element) = self
            .tensor
            .expect("Bound::arg: this operand was built over geometry alone and has no tensor");
        TileTensorLaunch::new(tensor, element, width, self.spec)
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
#[derive(Clone)]
pub struct Unbound {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}

/// One level's window along an operand's dimensions, in their order: the extent the launch cuts
/// each dimension to, and whether the kernel hands it down dynamic.
type LevelWindow = SmallVec<[(Extent, bool); Space::MAX_RANK]>;
