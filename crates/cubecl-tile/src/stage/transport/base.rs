//! Choosing the transport that moves one memory tile's cells into another's, and the copy.
//! Cooperative fills assume every unit their destination's level names ([`Level::copied_by`])
//! runs them: the
//! cube's, or the plane's that owns the stage.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::straight::fill_extent;
use crate::*;

/// The line of `d` the stage line `i` of `layout` lies at, written with `value`: the one write
/// every unit's fill makes of a stage line.
#[cube]
pub(crate) fn store_line<T: Numeric, W: Size>(
    d: &mut [Vector<T, W>],
    layout: &BufferLayout,
    i: usize,
    value: Vector<T, W>,
) {
    d[layout.line_offset(i)] = value;
}

/// Each value of the `W`-wide source line at the logical coordinates `line` written at its own
/// cell of `d`, a stage of one value a line laid out as `layout`: the write a fill makes of a
/// source line whose values the stage lays apart.
#[cube]
pub(crate) fn store_scattered<T: Numeric, D: Size, W: Size>(
    d: &mut [Vector<T, D>],
    layout: &BufferLayout,
    line: &CoordsDyn,
    value: Vector<T, W>,
) {
    #[unroll]
    for v in 0..W::value() {
        d[layout.cell_offset(line, W::value(), v)] = Vector::<T, D>::cast_from(value.extract(v));
    }
}

/// The comptime offsets each turn of a copy that `fill` deals over `lines` source lines adds to
/// where its unit's first line lands in a stage laid out as `layout`, where every unit's turns
/// agree ([`Walk::uniform_turn_offsets`]): one offset a turn where each source line lands as one
/// stage line, one a value where the fill `scatters` `width`-wide lines' values over their cells.
#[cube]
pub(crate) fn uniform_store_offsets(
    layout: &BufferLayout,
    #[comptime] fill: &Level,
    lines: usize,
    #[comptime] width: usize,
    #[comptime] scatters: bool,
) -> comptime_type!(Option<Vec<Vec<usize>>>) {
    let constants = layout.constants();
    let lines = lines.constant();
    comptime!(constants.and_then(|layout| {
        Walk::uniform_turn_offsets(
            fill,
            lines.map(|lines| lines as usize),
            |line| match scatters {
                true => {
                    let source = layout.source_line(line, width);
                    (0..width)
                        .map(|v| layout.cell_offset(&source, width, v))
                        .collect()
                }
                false => Some(vec![layout.line_offset(line)]),
            },
        )
    }))
}

/// One side of a fill, as far as choosing a transport goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StoreForm {
    /// How the values sit in the buffer.
    pub(crate) packing: Packing,
    /// The values one physical line holds.
    pub(crate) vector_size: usize,
    /// Whether the coordinates gather.
    pub(crate) gathered: bool,
    /// Whether the gather lands two cells on one, so a write through it aliases.
    pub(crate) overlaps: bool,
    /// Whether a read past the logical bound masks to zero.
    pub(crate) masks: bool,
    /// Whether the store has an address (a buffer, not an erased call).
    pub(crate) addressed: bool,
    /// Who moves the store's lines when it is the source of a fill ([`Access::delivery`]).
    pub(crate) delivery: Delivery,
}

/// Which transport moves a memory tile's cells into another memory tile's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TransportKind {
    /// The packed words verbatim.
    Packed,
    /// Every line in the destination's physical order, or the source's where an engine moves them;
    /// only from an addressed source to a whole, unmasked, addressed destination that replaces.
    Straight,
    /// Source and destination each read as a flat run of its own window.
    Scanned(Scan),
}

/// What a [`Scanned`](TransportKind::Scanned) transport reads out of the source per line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scan {
    /// The source's served element, line for line.
    Element,
    /// `u32` words, unpacked as they are read.
    Words,
}

impl TransportKind {
    /// The transport `src` reaches `dst` by; panics on a pairing with no transport, or an engine
    /// moving memory to memory on one that does not move each line whole
    /// ([`refusal_past_registers`](TransportKind::refusal_past_registers)).
    pub(crate) fn new(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
        let kind = Self::choose(dst, src, access, space);
        if let Some(why) = kind.refusal_past_registers(dst, src) {
            panic!("TransportKind: {why}");
        }
        kind
    }

    /// Why an engine moving memory to memory cannot move `src`'s lines into `dst` by this
    /// transport, `None` where it can or where no such engine moves them. Such an engine, as the
    /// copy engine, moves a source line's bytes to one destination line, by address, without a
    /// unit touching them: only a transport whose every line is such a move takes one. A scan may
    /// sum into or window its destination, and a padded stage assembles each line out of several
    /// source cells.
    fn refusal_past_registers(self, dst: StoreForm, src: StoreForm) -> Option<String> {
        if src.delivery.route() != Route::MemoryToMemory {
            return None;
        }
        if matches!(self, TransportKind::Scanned(_)) {
            return Some(
                "a copy moved memory to memory fills a whole, unmasked stage that replaces, and \
                 this destination is windowed, masked, summing or a sink"
                    .to_string(),
            );
        }
        if dst.vector_size != src.vector_size {
            return Some(format!(
                "a copy moved memory to memory moves whole source lines, but this stage is served \
                 in {}-wide lines out of {}-wide ones",
                dst.vector_size, src.vector_size
            ));
        }
        (!src.addressed).then(|| {
            "a copy moved memory to memory reads its source by address, and this one is an \
             erased call"
                .to_string()
        })
    }

    /// The transport the two forms and the destination's access leave, before any claim a
    /// delivery adds.
    fn choose(dst: StoreForm, src: StoreForm, access: &Access, space: &Space) -> Self {
        if !dst.packing.as_stored() {
            assert!(
                access.whole
                    && !access.overhang.masks()
                    && src.packing == dst.packing
                    && dst.vector_size == src.vector_size,
                "TransportKind: a packed stage is a fresh whole buffer filled from the window it \
                 was shaped over, at the same packing and width"
            );
            return TransportKind::Packed;
        }
        if access.whole
            && !access.overhang.masks()
            && src.packing.as_stored()
            && access.write == Write::Replace
            && dst.addressed
            && src.addressed
        {
            // A padded stage assembles each destination line from scalar source cells, and a
            // stage of one value a line scatters each source line's values.
            assert!(
                dst.vector_size == src.vector_size || src.vector_size == 1 || dst.vector_size == 1,
                "TransportKind: a padded stage is filled from a plain scalar operand, but its \
                 source serves {}-wide lines",
                src.vector_size
            );
            return TransportKind::Straight;
        }
        TransportKind::Scanned(Scan::new(dst, src, space))
    }
}

#[cube]
impl Memory {
    /// Cooperative cyclic copy of `src` into `self`, whole lines at `self`'s width.
    /// The caller must `sync_cube`, or `sync_plane` for a stage a plane owns, between this fill
    /// and its readers.
    pub(crate) fn fill_from<T: Numeric>(&mut self, src: &Memory, #[comptime] space: &Space) {
        let size!(W) = comptime!(self.store.vector_size);
        // A gathered stage records where it read from: nothing in the staged window says which
        // taps took the boundary value.
        #[comptime]
        match &mut self.source_window {
            ComptimeOption::Some(source) => {
                source.origin = src.window.origin.clone();
                source.bound = src.window.bound.clone();
            }
            ComptimeOption::None => {}
        }
        let addressed = self.store.addressed();
        let src_addressed = src.store.addressed();
        let transport = comptime!(TransportKind::new(
            StoreForm {
                packing: self.store.packing,
                vector_size: self.store.vector_size,
                gathered: !self.projection.is_direct(),
                overlaps: self.projection.composition() != Composition::Disjoint,
                masks: self.access.overhang.masks(),
                addressed,
                delivery: self.access.delivery,
            },
            StoreForm {
                packing: src.store.packing,
                vector_size: src.store.vector_size,
                gathered: !src.projection.is_direct(),
                overlaps: src.projection.composition() != Composition::Disjoint,
                masks: src.access.overhang.masks(),
                addressed: src_addressed,
                delivery: src.access.delivery,
            },
            &self.access,
            space
        ));
        match transport {
            TransportKind::Packed => {
                let size!(WP) = comptime!(self.store.packing.physical(self.store.vector_size));
                self.fill_straight::<u32, WP>(src, space);
            }
            TransportKind::Straight => self.fill_straight::<T, W>(src, space),
            TransportKind::Scanned(scan) => self.fill_scanned::<T, W>(src, comptime!(scan)),
        }
    }
}

impl Scan {
    /// What the scan reads per line, decided by the source's form.
    fn new(dst: StoreForm, src: StoreForm, space: &Space) -> Self {
        // The scan walks both windows cell by cell in their logical order and reaches each
        // buffer through its own map: a destination whose gather partitions its buffer takes
        // every cell once, and only one that lands two cells on one would alias a write.
        assert!(
            !src.gathered && !dst.overlaps,
            "TransportKind: a gathered tile fills only a whole, unmasked, unpacked \
             destination (a stage), and a scan writes no destination whose gather overlaps"
        );
        assert!(
            dst.vector_size == src.vector_size,
            "TransportKind: a plain source is scanned at the destination's width, so a padded \
             stage has to take the straight fill"
        );
        // `storage_extents` rounds the innermost extent up; the scan would miss an unfillable line.
        fill_extent(space, src.vector_size, dst.vector_size, src.masks);
        match src.packing {
            Packing::Plain | Packing::Words { .. } => Scan::Element,
            Packing::Packed { field: _ } => Scan::Words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::quant::scheme::QuantValue;

    const K: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);

    fn space() -> Space {
        Space::new(&[(K, 8), (N, 32)])
    }

    fn access() -> Access {
        Access {
            whole: true,
            overhang: Overhang::Never,
            write: Write::Replace,
            copy: Level::copied_by_cube(64),
            storage: WindowStorage::Contiguous,
            delivery: Delivery::SyncPerUnit,
        }
    }

    fn plain(vector_size: usize) -> StoreForm {
        StoreForm {
            packing: Packing::Plain,
            vector_size,
            gathered: false,
            overlaps: false,
            masks: false,
            addressed: true,
            delivery: Delivery::SyncPerUnit,
        }
    }

    fn asynchronous(form: StoreForm) -> StoreForm {
        StoreForm {
            delivery: Delivery::AsyncPerUnit,
            ..form
        }
    }

    fn words(vector_size: usize) -> StoreForm {
        StoreForm {
            packing: Packing::Packed {
                field: Field::Quant(QuantValue::Q4F),
            },
            ..plain(vector_size)
        }
    }

    fn kind(dst: StoreForm, src: StoreForm, access: &Access) -> TransportKind {
        TransportKind::new(dst, src, access, &space())
    }

    #[test]
    fn a_whole_plain_stage_is_filled_straight() {
        assert_eq!(kind(plain(4), plain(4), &access()), TransportKind::Straight);
    }

    #[test]
    fn a_padded_stage_is_still_straight() {
        assert_eq!(kind(plain(4), plain(1), &access()), TransportKind::Straight);
    }

    /// A stage of one value a line takes each wider source line whole and scatters its values.
    #[test]
    fn a_one_value_stage_of_a_wider_source_is_filled_straight() {
        assert_eq!(kind(plain(1), plain(4), &access()), TransportKind::Straight);
    }

    /// A copy engine moves each source line whole to one stage line, so it scatters nothing.
    #[test]
    #[should_panic(expected = "memory to memory moves whole source lines")]
    fn an_async_copy_refuses_a_scattered_stage() {
        kind(plain(1), asynchronous(plain(4)), &access());
    }

    #[test]
    fn a_windowed_masked_or_summing_destination_scans() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let masked = Access {
            overhang: Overhang::Masked,
            ..access()
        };
        let summing = Access {
            write: Write::Accumulate,
            ..access()
        };
        let relayed = Access {
            write: Write::Exclusive(Schedule::Sequential),
            ..access()
        };
        for access in [windowed, masked, summing, relayed] {
            assert_eq!(
                kind(plain(4), plain(4), &access),
                TransportKind::Scanned(Scan::Element),
                "{access:?}"
            );
        }
    }

    #[test]
    fn a_whole_sink_scans() {
        let sink = StoreForm {
            addressed: false,
            ..plain(4)
        };
        assert_eq!(
            kind(sink, plain(4), &access()),
            TransportKind::Scanned(Scan::Element)
        );
    }

    #[test]
    fn a_packed_stage_moves_words_and_nothing_else() {
        assert_eq!(kind(words(8), words(8), &access()), TransportKind::Packed);
    }

    #[test]
    fn the_source_names_what_a_scan_reads() {
        let scanned = Access {
            whole: false,
            ..access()
        };
        for (src, scan) in [(plain(4), Scan::Element), (words(4), Scan::Words)] {
            assert_eq!(
                kind(plain(4), src, &scanned),
                TransportKind::Scanned(scan),
                "{src:?}"
            );
        }
    }

    /// An async copy rides the transports whose every line is one source line moved whole: the
    /// straight and packed ones, never a scan.
    #[test]
    fn an_async_copy_rides_a_straight_or_packed_fill() {
        assert_eq!(
            kind(plain(4), asynchronous(plain(4)), &access()),
            TransportKind::Straight
        );
        assert_eq!(
            kind(words(8), asynchronous(words(8)), &access()),
            TransportKind::Packed
        );
    }

    #[test]
    #[should_panic(expected = "memory to memory fills a whole, unmasked stage")]
    fn an_async_copy_refuses_a_scan() {
        let masked = Access {
            overhang: Overhang::Masked,
            ..access()
        };
        kind(plain(4), asynchronous(plain(4)), &masked);
    }

    /// A padded stage assembles each line out of scalar cells, which no single copy moves.
    #[test]
    #[should_panic(expected = "memory to memory moves whole source lines")]
    fn an_async_copy_refuses_a_padded_stage() {
        kind(plain(4), asynchronous(plain(1)), &access());
    }

    /// A destination whose gather partitions its buffer, as a block-split output's does, is
    /// scanned cell by cell through its map: every cell lands once.
    #[test]
    fn a_partitioning_destination_scans() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let partitioned = StoreForm {
            gathered: true,
            ..plain(4)
        };
        assert_eq!(
            kind(partitioned, plain(4), &windowed),
            TransportKind::Scanned(Scan::Element)
        );
    }

    /// A destination whose gather lands two cells on one would take a value twice.
    #[test]
    #[should_panic(expected = "no destination whose gather overlaps")]
    fn an_overlapping_destination_refuses_a_scan() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let overlapping = StoreForm {
            gathered: true,
            overlaps: true,
            ..plain(4)
        };
        kind(overlapping, plain(4), &windowed);
    }

    #[test]
    #[should_panic(expected = "a gathered tile fills only a whole")]
    fn a_gathered_source_refuses_a_scan() {
        let windowed = Access {
            whole: false,
            ..access()
        };
        let gathered = StoreForm {
            gathered: true,
            ..plain(4)
        };
        kind(plain(4), gathered, &windowed);
    }
}
