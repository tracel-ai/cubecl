//! The [`Arg`] builder, which turns a launched tensor into an operand.

use core::marker::PhantomData;

use cubecl::zspace::{SmallVec, Tiling};

use cubecl::ir::ElemType;
use cubecl::prelude::*;

use super::{boundaries::Boundaries, derivation::Derivation, labels::Labels};
use crate::BoundaryPolicy;
use crate::{
    Axis, Delivery, Extent, Field, Geometry, Launcher, LineMisfit, Packing, Projection, Refusal,
    Space, Storage, StoragePartitioning, TileArgLaunch, TileSpec, VectorTile,
};

/// Typestate marker: the operand's axes are not yet stated.
pub struct Unlabelled;
/// Typestate marker: the operand's axes are stated.
pub struct Labelled;

/// What the builder accumulates.
pub(crate) struct ArgData<'a> {
    pub(crate) launch: &'a Launcher,
    /// The tensor this operand is read from and the element its buffer holds; `None` for a
    /// destination with no address.
    pub(crate) binding: Option<(TensorBinding, ElemType)>,
    pub(crate) geometry: Geometry,
    pub(crate) axes: &'a [Axis],
    pub(crate) batches: &'a [Axis],
    /// The operand's own affine mapping; `None` derives it from the labelled dims.
    pub(crate) projection: Option<Projection>,
    pub(crate) width: usize,
    /// The stated policy; `None` derives one from the overhangs.
    pub(crate) boundary: Option<BoundaryPolicy>,
    pub(crate) packing: Packing,
    /// Who moves the operand into a stage ([`delivery`](Arg::delivery)).
    pub(crate) delivery: Delivery,
    /// Whether the labelled dims bind in stride order ([`in_stride_order`](Arg::in_stride_order)).
    pub(crate) in_stride_order: bool,
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

    /// This operand's values are `u32` words of `field`-wide values, each read whole
    /// ([`Packing::Words`]): block scales as a block-scaled instruction takes them.
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

    /// The derivation both builds share, looked up where an earlier launch already made it
    /// ([`Derivation`]): it reads nothing a launch changes but the tensor's address.
    fn realize(self) -> Result<(Bound, Geometry), Refusal> {
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
            // Its tiling restated over the settled geometry's dims.
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
    fn derive(&self, packing: Packing, stored: Tiling) -> Result<Derived, Refusal> {
        let data = &self.data;
        let launch = data.launch;
        let width = data.width;
        let stated = data.projection.is_some() || stored.is_tiled();
        let (geometry, axes) = match data.in_stride_order {
            true if stated => return Err(Refusal::StrideOrderOfAStatedLayout),
            true => data.geometry.clone().in_stride_order(data.axes),
            false => (data.geometry.clone(), data.axes.to_vec()),
        };
        let (storage, stored_tiles) = self.storage(&geometry, &axes);
        let labels = match data.projection.clone() {
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
    /// axis (one contiguous run each), every other window walked through the layout; and the stored
    /// tiles themselves, finest first ([`TileSpec::stored_tiles`]).
    fn storage(&self, geometry: &Geometry, axes: &[Axis]) -> (Storage, Vec<(Axis, usize)>) {
        let launch = self.data.launch;
        if !(self.data.projection.is_none() && geometry.tiling().is_tiled()) {
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
        // Each level's window along the operand's axes, outermost first: the extent this launch
        // cuts it to, and whether the kernel hands it down dynamic. A level that cuts an axis gives
        // it its tile; one that does not passes the window above it down.
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
#[derive(Clone)]
pub struct Unbound {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}

/// One level's window along an operand's axes, in their order: the extent the launch cuts each axis
/// to, and whether the kernel hands it down dynamic.
type LevelWindow = SmallVec<[(Extent, bool); Space::MAX_RANK]>;
