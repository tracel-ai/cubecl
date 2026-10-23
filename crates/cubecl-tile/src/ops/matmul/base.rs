//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile, each descending to its leaf through
//! [`Descent::contract`].

use cubecl::cmma::{MatrixIdent, MatrixLayout};
use cubecl::prelude::*;

use crate::ops::matmul::leaf::FragmentRead;
use crate::*;

#[cube]
impl Tile {
    /// `c = a · b`, under the semiring the accumulator was opened with, or a memory window stated
    /// [`accumulating`](Tile::accumulating). A memory window starts the sum fresh rather than
    /// reading the cell.
    pub fn mm(&mut self, lhs: &Tile, rhs: &Tile) {
        let in_memory = self.is_memory();
        if comptime!(in_memory) {
            // A clone holds the same window: reading it reads `self`.
            let acc = self.clone();
            let contraction = match &acc.kind {
                TileKind::Memory(g) => g.stated_contraction(),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("Tile::mm: only a memory window states its contraction")
                }
            };
            let semiring = contraction.semiring;
            let init_from = self.request_init_from(comptime!(InitFrom::Identity));
            let define!(Served) = self.element;
            match comptime!(init_from) {
                InitFrom::Identity => {}
                InitFrom::Cell => self.init_identity::<Served>(comptime!(semiring.add())),
            }
            self.mma(lhs, rhs);
            self.request_init_from(comptime!(InitFrom::Cell));
        } else {
            self.reset();
            self.mma(lhs, rhs);
        }
    }

    /// `operand` dequantized at `element` for this accumulator to contract: the plane-level
    /// dequantize, called where the kernel wants it instead of left to the leaf, which loads it as
    /// the multiply would. A cmma grid loads it into its fragments through a landing; one cmma
    /// fragment into the fragment its leaf builds; a walk of steps into the landing the steps
    /// read, where the tensor cores read through one. A plain operand is returned as it is: it
    /// has nothing to decode. So is any operand of a manual-mma sum, whose contraction decodes
    /// it as it loads it, at its other factor's element ([`Tile::mma`]): loaded here, ahead of
    /// that factor, its fragments could not hold `k` in the order the other factor's do.
    pub fn dequantized(&self, operand: &Tile, #[comptime] element: ElemType) -> Tile {
        let (quantized, packing) = (operand.quantized(), operand.packing());
        let decodes = comptime!(quantized || packing != Packing::Plain);
        let mma = self.contracts_through_mma();
        if comptime!(!decodes || mma) {
            operand.clone()
        } else {
            let out = comptime!(self.place.space.clone());
            let own = comptime!(operand.place.space.clone());
            let define!(Target) = element;
            match &self.kind {
                TileKind::PlanePartition(partition) => {
                    let grid = partition.grid();
                    match comptime!(Descent::of(grid, &self.place, &own, &own)) {
                        Descent::Here => {
                            operand.fragment_for::<Target>(&partition.at(0usize, 0usize), out)
                        }
                        Descent::Cells(instruction) => PlanePartition::operand::<Target>(
                            operand,
                            self,
                            instruction,
                            DeepLoad::Steps,
                        ),
                        Descent::Steps(instruction) => match comptime!(instruction) {
                            Instruction::Cmma => {
                                let side = comptime!(Side::of(&own, &out));
                                operand.landed::<Target>(side, out)
                            }
                            Instruction::Mma { .. }
                            | Instruction::Registers { .. }
                            | Instruction::Wgmma => panic!(
                                "Tile::dequantized: a grid that steps its contraction through \
                                 manual-mma fragments holds each region's operand only while it \
                                 contracts; walk the regions and dequantize each one into the \
                                 accumulator's region"
                            ),
                        },
                    }
                }
                TileKind::PlaneTile(fragment) => operand.fragment_for::<Target>(fragment, out),
                TileKind::Memory(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!(
                        "Tile::dequantized: an operand is dequantized for the fragments it meets"
                    )
                }
            }
        }
    }

    /// This operand as the fragment the one-fragment `acc`'s leaf loads it into, at `T`.
    fn fragment_for<T: Numeric>(&self, acc: &PlaneTile, #[comptime] out: Space) -> Tile {
        let own = comptime!(self.place.space.clone());
        let side = comptime!(Side::of(&own, &out));
        let ident = comptime!(match side {
            Side::Lhs => MatrixIdent::A,
            Side::Rhs => MatrixIdent::B,
        });
        let fragment = match acc {
            PlaneTile::Cmma(_) => {
                let read = comptime!(FragmentRead::new(side, &own, &out));
                let fragment = self.cmma_fragment::<T>(ident, comptime!(read.clone()));
                PlaneTile::new_Cmma(CmmaData {
                    fragment: unsafe { fragment.downcast_unchecked::<Held>() },
                    ident,
                    layout: comptime!(read.layout),
                    shape: comptime!((read.m, read.n)),
                    scratch: ComptimeOption::new_None(),
                })
            }
            PlaneTile::Mma(_) => panic!(
                "Tile::dequantized: a manual-mma sum decodes its factor as its contraction loads it"
            ),
            PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => panic!(
                "Tile::dequantized: a register block and a warpgroup MMA read their operands where \
                 they lie"
            ),
        };
        Tile::new::<T>(
            TileKind::new_PlaneTile(fragment),
            comptime!(self.place.clone()),
        )
    }

    /// `c += a · b`, added to what the accumulator or memory window holds. A packed factor is
    /// decoded into `f32`, unless the kernel [`dequantized`](Tile::dequantized) it at another
    /// element first, a manual-mma instruction decodes it at its other factor's element, or the
    /// instruction reads its codes as stored.
    pub fn mma(&mut self, lhs: &Tile, rhs: &Tile) {
        let stored = self.reads_codes_as_stored();
        let mma = self.contracts_through_mma();
        let (lhs_read, rhs_read) = (lhs.read_element(), rhs.read_element());
        let (lhs_packing, rhs_packing) = (lhs.packing(), rhs.packing());
        // A manual-mma instruction multiplies two factors of one element, so a packed factor
        // against a plain one decodes at that one's element.
        let (lhs_read, rhs_read) = comptime!(match (
            mma,
            lhs_packing == Packing::Plain,
            rhs_packing == Packing::Plain
        ) {
            (true, false, true) => (rhs_read, rhs_read),
            (true, true, false) => (lhs_read, lhs_read),
            _ => (lhs_read, rhs_read),
        });
        let define!(Served) = self.element;
        let define!(Lhs) = comptime!(if stored { lhs.element } else { lhs_read });
        let define!(Rhs) = comptime!(if stored { rhs.element } else { rhs_read });
        Descent::contract::<Served, Lhs, Rhs>(self, lhs, rhs)
    }

    /// Whether this accumulator's fragments contract through the device's manual MMA.
    fn contracts_through_mma(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::PlaneTile(tile) => tile.contracts_through_mma(),
            TileKind::PlanePartition(partition) => {
                partition.at(0usize, 0usize).contracts_through_mma()
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether this accumulator's fragments read their factors' packed codes as stored; a memory
    /// window decodes them.
    fn reads_codes_as_stored(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::PlaneTile(tile) => tile.reads_codes_as_stored(),
            TileKind::PlanePartition(partition) => {
                partition.at(0usize, 0usize).reads_codes_as_stored()
            }
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// This plane-resident accumulator back at its semiring's identity, as it was opened.
    pub fn reset(&mut self) {
        // A clone holds the same fragments: reading it reads `self`.
        let acc = self.clone();
        let semiring = match &acc.kind {
            TileKind::PlaneTile(t) => t.semiring(),
            TileKind::PlanePartition(p) => p.at(0usize, 0usize).semiring(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::reset: only a plane-resident accumulator has an identity to return to"
                )
            }
        };
        let define!(Served) = self.element;
        self.init_identity::<Served>(comptime!(semiring.add()));
    }
}

#[cube]
impl Tile {
    /// Commit the warpgroup MMAs this accumulator issued since its last commit, and return their
    /// completion: once it resolves, they are done reading what they read, the stages a walk
    /// holds for them among it. Only a warpgroup accumulator contracts asynchronously.
    pub fn commit(&self) -> Pending<()> {
        let define!(Served) = self.element;
        match &self.kind {
            TileKind::PlanePartition(p) => p.fragment().commit::<Served>(),
            TileKind::PlaneTile(t) => t.commit::<Served>(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::commit: only a plane group's warpgroup accumulator contracts asynchronously"
            ),
        }
    }
}

#[cube]
impl Tile {
    /// This memory window as a contraction's output: each unit's share tiled into `block`, folded
    /// under `semiring`. A plane-resident accumulator states both when it is opened.
    pub fn accumulating(
        &self,
        #[comptime] block: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) -> Tile {
        match &self.kind {
            TileKind::Memory(g) => {
                let mut g = g.clone();
                g.set_contraction(comptime!(Contraction { block, semiring }));
                Tile {
                    kind: TileKind::new_Memory(g),
                    factor: self.factor.clone(),
                    place: comptime!(self.place.clone()),
                    element: comptime!(self.element),
                }
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::accumulating: a memory window states how it is contracted into; a \
                 plane-resident accumulator states it when opened (Tile::accumulator)"
            ),
        }
    }
}

#[cube]
impl PlaneTile {
    /// Contract this plane tile, each factor times whatever scales it carries.
    pub(crate) fn mma<E: Numeric, EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(lhs, rhs);
                strided_2d(lhs, rhs, comptime!(out.clone()), transposed);
                d.mma::<E, EL, ER>(lhs, rhs, out)
            }
            PlaneTile::Mma(d) => {
                // The manual-mma instruction reads registers, where a scaled factor cannot land,
                // unless the device's block-scaled instruction takes the scales beside the values,
                // or a fragment deeper than its instruction decodes them into its registers as it
                // loads, a step at a time out of shared memory
                // ([`steps`](crate::tile::kind::plane::mma::steps)) or in lane runs out of global
                // memory ([`lane_runs`](crate::tile::kind::plane::mma::lane_runs)), each of which
                // says why where it cannot.
                let (m, n, k) = comptime!((d.m, d.n, d.k));
                let block_scaled = comptime!(d.operands == MmaOperands::BlockScaled);
                // An operand already in fragments was held to the instruction as it loaded
                // (`PlanePartition::operand`), and a fragment no longer states its storage.
                let (lhs_in_memory, rhs_in_memory) = (lhs.is_memory(), rhs.is_memory());
                if comptime!(block_scaled && lhs_in_memory) {
                    assert_block_scaled::<EL>(lhs, m, n, k);
                }
                if comptime!(block_scaled && rhs_in_memory) {
                    assert_block_scaled::<ER>(rhs, m, n, k);
                }
                let steps = lane_run_steps::<EL>(m, n, k);
                if comptime!(!block_scaled && steps == 1) {
                    lhs.refuse_factor("PlaneTile::Mma");
                    rhs.refuse_factor("PlaneTile::Mma");
                }
                flattened_k(lhs, rhs, out);
                d.mma::<E, EL, ER>(lhs, rhs)
            }
            PlaneTile::Wgmma(d) => {
                lhs.refuse_factor("PlaneTile::Wgmma");
                rhs.refuse_factor("PlaneTile::Wgmma");
                d.mma::<E, EL, ER>(lhs, rhs, out)
            }
            PlaneTile::Registers(d) => match &lhs.kind {
                TileKind::PlaneTile(block) => match block {
                    PlaneTile::Registers(block) => d.mma_block::<E, EL, ER>(block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::PlanePartition(block) => match block.fragment() {
                    PlaneTile::Registers(block) => d.mma_block::<E, EL, ER>(&block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::Memory(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    let rhs_along_k = comptime!(
                        lhs.place
                            .space
                            .contains(rhs.place.space.axis_at(rhs.place.space.rank() - 1))
                    );
                    strided_2d(lhs, rhs, comptime!(out.clone()), rhs_along_k);
                    d.mma::<E, EL, ER>(lhs, rhs, out)
                }
            },
        }
    }
}

/// Asserts that operands are not gathered and read as one matrix each.
#[cube]
fn strided_2d(lhs: &Tile, rhs: &Tile, #[comptime] out: Space, #[comptime] rhs_along_k: bool) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let kc = Space::merge(&[&lhs.place.space, &rhs.place.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.place.space);
        let cols = axes.cols(&out);
        let rhs_matrix = if rhs_along_k {
            MatrixAxes::new(&rhs.place.space, cols, kc)
        } else {
            MatrixAxes::new(&rhs.place.space, kc, cols)
        };
        MatrixAxes::new(&lhs.place.space, axes.rows(&out), kc).is_ok() && rhs_matrix.is_ok()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `memory` arm of `Descent::contract`"
    ));
}

/// Whether `rhs` is read col-major: a col-major cmma fragment or a staged `(col, k)` window.
#[cube]
fn transposed_rhs(lhs: &Tile, rhs: &Tile) -> comptime_type!(bool) {
    match &rhs.kind {
        TileKind::PlaneTile(t) => match t {
            PlaneTile::Cmma(d) => comptime!(d.layout == MatrixLayout::ColMajor),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
        },
        // The contracted axis is the lhs's trailing one; not every axis the output lacks is.
        // Only a shared stage can answer column-major; a gmem rhs never does.
        TileKind::Memory(m) => comptime!(
            m.address == AddressSpace::Shared
                && crate::ops::matmul::leaf::rhs_layout(
                    &rhs.place.space,
                    lhs.place.space.axis_at(lhs.place.space.rank() - 1)
                ) == MatrixLayout::ColMajor
        ),
        TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => comptime!(false),
    }
}

/// Asserts that operands contract their shared axes in the same order.
#[cube]
fn flattened_k(lhs: &Tile, rhs: &Tile, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.place.space, &rhs.place.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.place.space.difference(&out),
        rhs.place.space.difference(&out)
    ));
}
