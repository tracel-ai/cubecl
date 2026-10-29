//! The TMA backing store ([`TmaData`]): a tensor-map source bulk-copied into shared memory.

use cubecl::{ir::Scope, prelude::barrier::Barrier, prelude::*, unexpanded};

use cubecl_std::tensor::{
    ViewMut, ViewMutExpand, VirtualViewMutExpand,
    layout::{CoordsDyn, VirtualLayout},
};

use crate::*;

/// A TMA tensor-map source: the launch-built tensor map and where a position lands in it, the
/// current box origin and the logical bound.
///
/// A packed source is moved as the words it is stored in, so the stage it lands in holds them
/// ([`packing`](Self::packing)); the scales it rides under are ones a stage keeps beside the words,
/// since the engine lands bytes as they lie and decodes nothing.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct TmaData {
    /// The tensor map, at the element it holds.
    map: Raw<TiledMap>,
    /// Where a position in the operand's space lands in the map.
    layout: VirtualLayout<CoordsDyn, Sequence<i32>>,
    pos: CoordsDyn,
    pub(crate) bound: CoordsDyn,
    /// The launch's cube size, `0` when unknown ([`FillUnits::count`](crate::FillUnits)).
    #[cube(comptime)]
    pub(crate) units: usize,
    /// How the values sit in the cells the descriptor moves.
    #[cube(comptime)]
    pub(crate) packing: Packing,
    /// The cells a line of a stage it fills holds ([`TileArgLaunch::served_wide`]).
    #[cube(comptime)]
    pub(crate) width: usize,
}

#[cube]
impl TmaData {
    /// Wrap a TMA tensor `map`, `layout` landing a position in it, as a `TmaGmem` tile payload,
    /// positioned at the origin.
    pub(crate) fn from_tensor_map<T: Numeric>(
        map: TensorMap<T, Tiled>,
        layout: VirtualLayout<CoordsDyn, Sequence<i32>>,
        #[comptime] rank: usize,
        #[comptime] units: usize,
        #[comptime] packing: Packing,
        #[comptime] width: usize,
    ) -> TmaData {
        let bound = layout.shape();
        let mut pos = CoordsDyn::new();
        #[unroll]
        for _ in 0..rank {
            pos.push(0u32);
        }
        TmaData {
            map: Raw::<TiledMap>::of::<T>(map),
            layout,
            pos,
            bound,
            units,
            packing,
            width,
        }
    }

    /// Issue the `tensor_map_load` into `dst` on `barrier` without arriving or waiting.
    /// Only the electing unit may call it, since it alone declares the transaction count.
    pub(crate) fn stage_into<T: Numeric>(&self, dst: &mut Memory, barrier: &Shared<Barrier>) {
        // A box lands its rows dense, swizzled by the descriptor where the stage keeps them
        // swizzled; the launch built the descriptor off the same answer
        // (`StageStorage::tma_swizzle`).
        comptime!(
            dst.layout
                .rows
                .tma_swizzle()
                .unwrap_or_else(|why| panic!("TmaData::stage_into: a TMA box cannot land {why}"))
        );
        // A packed stage holds the words the box moves, which the descriptor names at `T`.
        match comptime!(dst.store.packing) {
            Packing::Plain | Packing::Words { .. } => self.view::<T>().tensor_map_load(
                barrier,
                dst.store.buffer_mut::<T>(),
                self.pos.clone(),
            ),
            Packing::Packed { .. } => self.view::<T>().tensor_map_load(
                barrier,
                unsafe { dst.store.buffer_mut::<u32>().downcast_mut_unchecked::<T>() },
                self.pos.clone(),
            ),
        }
    }

    /// Bulk-copy into shared-memory `dst` and wait, on a local mbarrier.
    pub(crate) fn load_into<T: Numeric>(&self, dst: &mut Memory) {
        comptime!(assert!(
            dst.address == AddressSpace::Shared,
            "TmaData::load_into: a tensor map bulk-copies into shared memory only"
        ));
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        let expected = select(UNIT_POS == 0, dst.size_bytes::<T>(), 0);
        if UNIT_POS == 0 {
            self.stage_into::<T>(dst, &barrier);
        }
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }

    /// Bulk-copy shared-memory `src` out to this box, once every unit of the cube has written its
    /// share of `src`: every unit fences its writes into the async proxy the engine reads through
    /// and meets the others on a cube barrier, then one unit issues the store and waits until the
    /// engine has read `src`, and the cube meets again, so no unit writes `src` before the engine
    /// is done with it. What lies past the tensor is not written: the descriptor clips the box.
    pub(crate) fn store_from<T: Numeric>(&self, src: &Memory) {
        comptime!(assert!(
            src.address == AddressSpace::Shared,
            "TmaData::store_from: a tensor map bulk-copies out of shared memory only"
        ));
        comptime!(
            src.layout
                .rows
                .assert_in_order("TmaData::store_from", "a descriptor stores a box dense")
        );
        sync_async_proxy_shared();
        sync_cube();
        if UNIT_POS == 0 {
            // As `stage_into`: a packed stage holds words the descriptor names at `T`.
            match comptime!(src.store.packing) {
                Packing::Plain | Packing::Words { .. } => self
                    .view::<T>()
                    .tensor_map_store(src.store.buffer::<T>(), self.pos.clone())
                    .wait(),
                Packing::Packed { .. } => self
                    .view::<T>()
                    .tensor_map_store(
                        unsafe { src.store.buffer::<u32>().downcast_unchecked::<T>() },
                        self.pos.clone(),
                    )
                    .wait(),
            }
        }
        sync_cube();
    }

    /// Window down to `step`: advance the global box origin by each axis's tile offset.
    // The layout is `Copy`, but its expand is an `Arc` that the expansion moves: clone it.
    #[allow(clippy::clone_on_copy)]
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> TmaData {
        let mut pos = CoordsDyn::new();

        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            match comptime!(step.level.tile(axis)) {
                Some(tile) => {
                    let index = step.coord(axis);
                    pos.push(self.pos[p] + (index * tile) as u32);
                }
                None => pos.push(self.pos[p]),
            }
        }

        TmaData {
            map: self.map.clone(),
            layout: self.layout.clone(),
            pos,
            bound: self.bound.clone(),
            units: self.units,
            packing: self.packing,
            width: self.width,
        }
    }
}

impl TmaData {
    /// The view a box is moved through, the map named at `T`, the element it holds.
    fn view<T: Numeric>(&self) -> ViewMut<'static, T, CoordsDyn> {
        unexpanded!()
    }
}

impl TmaDataExpand {
    fn __expand_view_method<T: Numeric>(
        &self,
        scope: &Scope,
    ) -> ViewMutExpand<'static, T, CoordsDyn> {
        let map = *self.map.__expand_at_method::<T>(scope);
        let view = VirtualViewMutExpand::<T, CoordsDyn, Sequence<i32>, TensorMap<T, Tiled>>::new(
            map,
            self.layout.clone(),
        );
        ViewMutExpand::new(scope, view)
    }
}
