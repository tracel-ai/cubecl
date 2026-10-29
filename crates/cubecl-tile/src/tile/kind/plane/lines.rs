//! A tile the plane holds in its units: loaded once, coalesced, and shared by shuffle.
//! Unit `t` holds line `t`, so the box must not hold more lines than the plane has units.

use cubecl::{ir::ElemType, prelude::*};

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

// Words one line holds, bound where the tile is opened.
define_size!(pub(crate) LW);

/// Bind the line width `LW` for the rest of the kernel's scope.
#[cube]
fn register_line_words(#[comptime] words: usize) {
    intrinsic!(|scope| {
        scope.register_size::<LW>(words);
    });
}

/// The lines a plane holds in its units, and where inside them a window sits.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct Lines {
    /// This unit's line, as words (unit `t` holds line `t`); filled under [`UnitRead::Shuffle`].
    line: Array<Vector<u32, LW>>,
    /// Every line, in this plane's shared-memory window; only under [`UnitRead::PlaneShared`].
    window: ComptimeOption<Shared<[u32]>>,
    /// Where this window starts inside the box, in scalars, one entry per axis of it.
    origin: Coords<u32>,
    /// The box the lines were loaded as: the lines' axes, then the line's.
    #[cube(comptime)]
    loaded: Space,
    /// What a coordinate along each axis of the box counts in lines; zero if not addressed.
    #[cube(comptime)]
    strides: Vec<usize>,
    /// Lines the box holds, which is the units it takes.
    #[cube(comptime)]
    lines: usize,
    /// The projection of the operand these lines stage.
    #[cube(comptime)]
    projection: Projection,
    /// The slot one value occupies in a word.
    #[cube(comptime)]
    field: Field,
    /// Words one line holds.
    #[cube(comptime)]
    words: usize,
    /// How a value gets from the unit that loaded its line to the unit that asks.
    #[cube(comptime)]
    read: UnitRead,
}

impl Lines {
    /// Each axis of `loaded` counted in lines, and the total line count.
    fn line_strides(loaded: &Space, projection: &Projection) -> (Vec<usize>, usize) {
        let rank = loaded.rank();
        let mut strides = vec![0; rank];
        let mut lines = 1;
        #[allow(clippy::needless_range_loop)]
        for p in (0..rank - 1).rev() {
            if projection.addresses(loaded.axis_at(p)) {
                strides[p] = lines;
                lines *= loaded.extent_at(p);
            }
        }
        (strides, lines)
    }
}

#[cube]
impl Lines {
    /// The empty box one region of `level` over `operand` fills, one line per unit.
    pub(crate) fn new(
        operand: &Tile,
        #[comptime] level: Level,
        #[comptime] read: UnitRead,
    ) -> Lines {
        let loaded = comptime!(level.child(&operand.place.space));
        let rank = comptime!(loaded.rank());
        let line = comptime!(loaded.extent_at(rank - 1));
        let projection = operand.projection();
        comptime!(assert!(
            projection.addresses(loaded.axis_at(rank - 1)),
            "Lines: the line runs along the operand's innermost axis, which it must address"
        ));
        let (strides, lines) = comptime!(Lines::line_strides(&loaded, &projection));
        let packing = operand.packing();
        let served = comptime!(operand.element);
        let field = comptime!(match packing {
            Packing::Packed { field } => field,
            Packing::Plain | Packing::Words { .. } => match served {
                ElemType::Float(kind) if Field::float_bits(kind) == 32 => Field::Float(kind),
                other => panic!(
                    "Lines: a plain operand is held as whole 32-bit words, and {other:?} is not \
                     one; bind the operand packed, or serve it as `f32`"
                ),
            },
        });
        let per_word = comptime!(field.per_word());
        comptime!(assert!(
            line.is_multiple_of(per_word),
            "Lines: a line of {line} values is not whole words of {per_word}"
        ));
        let words = comptime!(line / per_word);
        register_line_words(words);
        let window = match comptime!(read) {
            UnitRead::Shuffle => ComptimeOption::new_None(),
            UnitRead::PlaneShared => {
                let planes = comptime!(plane_windows(&operand.place.space, &operand.place.levels));
                let cells = comptime!(lines * words);
                let start = PLANE_POS.cast::<usize>() * cells;
                let end = start + cells;
                ComptimeOption::new_Some(
                    Shared::<[u32]>::new_slice(comptime!(cells * planes))
                        .map(|all| &all[start..end]),
                )
            }
        };
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for _axis in 0..rank {
            origin.push(0u32.runtime());
        }
        Lines {
            line: Array::<Vector<u32, LW>>::new(1usize),
            window,
            origin,
            loaded,
            strides,
            lines,
            projection,
            field,
            words,
            read,
        }
    }

    /// Lines the box holds.
    pub(crate) fn lines(&self) -> comptime_type!(usize) {
        comptime!(self.lines)
    }

    /// Values one line holds: what a read of the operand these lines stage serves.
    pub(crate) fn line(&self) -> comptime_type!(usize) {
        let rank = comptime!(self.loaded.rank());
        comptime!(self.loaded.extent_at(rank - 1))
    }

    /// The projection of the operand these lines stage.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        comptime!(self.projection.clone())
    }

    /// How the operand these lines stage is packed, as they hold it.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(Packing::Packed { field: self.field })
    }

    /// Load from `src`, the box's memory window: unit `t` reads line `t`.
    pub(crate) fn load<S: Numeric>(&mut self, src: &Tile) {
        let rank = comptime!(self.loaded.rank());
        let line = self.line();
        let lines = self.lines();
        let served = src.vector_size();
        comptime!(assert!(
            line.is_multiple_of(served),
            "Lines::load: a unit reads its line of {line} values in whole reads, and the source \
             serves {served} a read"
        ));
        let reads = comptime!(line / served);
        let words = comptime!(self.words);
        let per_read = comptime!(words / reads);
        let size!(WR) = per_read;
        let packing = src.packing();
        let unit = UNIT_POS_PLANE;
        let mut held = Vector::<u32, LW>::empty();
        if unit < comptime!(lines as u32) {
            #[unroll]
            for r in 0..reads {
                // Coordinates along addressed axes, zero elsewhere, the read `r` along the line.
                let mut at = CoordsDyn::new();
                #[unroll]
                for p in 0..rank - 1 {
                    let stride = comptime!(self.strides[p]);
                    if comptime!(stride > 0) {
                        at.push(
                            unit.divided_by(comptime!(stride as u32))
                                .remainder(comptime!(self.loaded.extent_at(p) as u32)),
                        );
                    } else {
                        at.push(0u32.runtime());
                    }
                }
                at.push(comptime!(r as u32).runtime());
                let got = match comptime!(packing) {
                    Packing::Packed { field: _ } => {
                        src.nd_words::<WR>(comptime!(Guard::Checked)).read(at)
                    }
                    Packing::Plain | Packing::Words { .. } => {
                        let size!(SV) = served;
                        let values = src.nd_packed::<S, SV>(comptime!(Guard::Checked)).read(at);
                        let mut bits = Vector::<u32, WR>::empty();
                        #[unroll]
                        for j in 0..per_read {
                            bits.insert(j, u32::reinterpret(values.extract(j)));
                        }
                        bits
                    }
                };
                #[unroll]
                for j in 0..per_read {
                    held.insert(comptime!(r * per_read + j), got.extract(j));
                }
            }
        }
        match comptime!(self.read) {
            UnitRead::Shuffle => self.line[0usize] = held,
            UnitRead::PlaneShared =>
            {
                #[comptime]
                match &mut self.window {
                    ComptimeOption::Some(window) => {
                        if unit < comptime!(lines as u32) {
                            let base = (unit * comptime!(words as u32)) as usize;
                            #[unroll]
                            for j in 0..words {
                                window[base + j] = held.extract(j);
                            }
                        }
                        sync_plane();
                    }
                    ComptimeOption::None => {}
                }
            }
        }
    }

    /// This window one level down, to `step`'s box: only the origin moves.
    pub(crate) fn at(&self, step: &Step, #[comptime] space: Space) -> Lines {
        let rank = comptime!(space.rank());
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(space.axis_at(p));
            let edge = comptime!(step.level.extent_in(&space, axis).get());
            origin.push(
                self.origin
                    .at(p)
                    .plus(step.coord(axis).times(edge).cast::<u32>()),
            );
        }
        Lines {
            line: self.line,
            window: self.window.clone(),
            origin,
            loaded: comptime!(self.loaded.clone()),
            strides: comptime!(self.strides.clone()),
            lines: comptime!(self.lines),
            projection: comptime!(self.projection.clone()),
            field: comptime!(self.field),
            words: comptime!(self.words),
            read: comptime!(self.read),
        }
    }

    /// The value at `coords` of this window, one entry per axis of the box, as `E`.
    /// The shuffle is the whole plane's, so a caller keeps its units converged around this.
    #[allow(dead_code)] // Reached through its expand, from [`Tile::scale_at`].
    pub(crate) fn read<E: Numeric>(&self, coords: &Coords<u32>) -> E {
        let rank = comptime!(self.loaded.rank());
        let mut line = 0u32.runtime();
        #[unroll]
        for p in 0..rank - 1 {
            let stride = comptime!(self.strides[p] as u32);
            if comptime!(stride > 0) {
                line = line.plus(self.origin.at(p).plus(coords.at(p)).times(stride));
            }
        }
        let byte = self.origin.at(rank - 1).plus(coords.at(rank - 1));
        let per_word = comptime!(self.field.per_word());
        let words = comptime!(self.words);
        let word = byte.divided_by(comptime!(per_word as u32));
        let field = byte.remainder(comptime!(per_word as u32));
        let held = match comptime!(self.read) {
            UnitRead::Shuffle => {
                // `word` is runtime, so every word is shuffled and one kept.
                let mine = self.line[0usize];
                let mut got = Vector::<u32, LW>::empty();
                #[unroll]
                for j in 0..words {
                    got.insert(j, plane_shuffle(mine.extract(j), line));
                }
                if comptime!(words > 1) {
                    got.extract_dynamic(word.cast::<usize>())
                } else {
                    got.extract(0usize)
                }
            }
            UnitRead::PlaneShared =>
            {
                #[comptime]
                match &self.window {
                    ComptimeOption::Some(window) => {
                        window[(line.times(comptime!(words as u32)).plus(word)) as usize]
                    }
                    ComptimeOption::None => panic!("Lines: no window was opened"),
                }
            }
        };
        // Shift the wanted field down first: decoding the whole word costs `per_word` decodes.
        let bits = comptime!(self.field.size_bits() as u32);
        let only = match comptime!(per_word > 1) {
            true => held >> field.times(bits),
            false => held,
        };
        unpack_line::<E, Const<1>, Const<1>>(
            Vector::<u32, Const<1>>::new(only),
            comptime!(self.field),
        )
        .extract(0usize)
    }
}
