//! Who moves a copy's lines ([`FillUnits`]), and the one loop that hands each unit its own
//! ([`FillUnits::move_lines`]).

use cubecl::prelude::*;

use crate::*;

/// The most turns a copy writes out one after another: past it, a loop strides by the units.
///
/// A comptime worker count unrolls the turns: a runtime stride (`CUBE_DIM`, or a plane's
/// `CUBE_DIM_X`) stalls each store on Metal's in-order pipe. Unknown or tiny cubes stay rolled.
const MOST_UNROLLED_TURNS: usize = 8;

/// The units that share a cooperative fill of a window: every unit of the cube, the units of the
/// planes the cube sets aside to fill its stages, or the units of one plane, for a stage that
/// plane owns and fills alone ([`Stages::smem`](crate::Stages::smem)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FillUnits {
    /// Whose units they are: the cube's, or one plane's.
    pub scope: ComputeScope,
    /// How many there are, `0` when unknown: a stage filled from this tile emits its fill
    /// straight-line when it knows. The launch's cube size, or one plane's width.
    pub count: usize,
    /// The planes at the end of the cube set aside to fill its stages, whose units alone share
    /// the fill of a cube's stage; `0` where every unit of the cube does.
    pub fillers: usize,
}

/// How the units walk the lines they share: line `w + t·n` is unit `w`'s turn `t`, of `n` units.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Turns {
    /// Every turn written out, `count` of them; the last checked against the lines where it can
    /// run past them (`ragged`).
    Unrolled { count: usize, ragged: bool },
    /// A loop striding by the units, for a unit count or a line count only the launch knows.
    Strided,
}

impl FillUnits {
    /// Every unit of the cube, `count` of them (`0` when unknown).
    pub(crate) fn cube(count: usize) -> Self {
        FillUnits {
            scope: ComputeScope::Cube,
            count,
            fillers: 0,
        }
    }

    /// The units of the `fillers` planes at the end of the cube set aside to fill its stages
    /// ([`Level::fillers`](crate::Level::fillers)), how many unknown: the launch states the
    /// cube's units, not the plane's width.
    pub(crate) fn filling_planes(fillers: usize) -> Self {
        FillUnits {
            scope: ComputeScope::Cube,
            count: 0,
            fillers,
        }
    }

    /// These units narrowed to `holder`'s where it has fewer: a plane's window of a buffer the
    /// cube fills is the plane's to fill.
    pub(crate) fn held_by(self, holder: ComputeScope) -> Self {
        match holder < self.scope {
            true => FillUnits {
                scope: holder,
                count: 0,
                fillers: 0,
            },
            false => self,
        }
    }

    /// How these units walk `lines` lines (`None` when only the launch knows how many): written
    /// out where both counts are known and each unit takes a few, a strided loop otherwise.
    pub(crate) fn turns(self, lines: Option<usize>) -> Turns {
        match lines {
            Some(lines) if self.count > 0 && lines.div_ceil(self.count) <= MOST_UNROLLED_TURNS => {
                self.unrolled(lines)
            }
            _ => Turns::Strided,
        }
    }

    /// Every turn of these units over `lines` lines written out, however many: what a copy that
    /// holds each turn's line in a register of its own needs.
    ///
    /// # Panics
    ///
    /// Where the unit count is unknown.
    pub(crate) fn unrolled(self, lines: usize) -> Turns {
        assert!(
            self.count > 0,
            "FillUnits: a copy that writes its turns out spreads its lines over the launch's \
             units, which this operand's spec does not state: bind it through `Launcher::arg`, \
             or set its `units`"
        );
        Turns::Unrolled {
            count: lines.div_ceil(self.count),
            ragged: !lines.is_multiple_of(self.count),
        }
    }
}

impl Turns {
    /// How many turns each unit takes, where they are written out.
    pub(crate) fn count(self) -> Option<usize> {
        match self {
            Turns::Unrolled { count, .. } => Some(count),
            Turns::Strided => None,
        }
    }
}

/// What a copy does with one of its lines: what [`FillUnits::move_lines`] hands each unit.
#[cube]
pub(crate) trait LineMove: CubeType {
    /// Move line `line`, this unit's turn `turn` where the turns are written out (`None` in a
    /// strided loop).
    fn move_line(&mut self, line: usize, #[comptime] turn: Option<usize>);
}

#[cube]
impl FillUnits {
    /// This unit's position among the units `fill` names, which a cooperative fill takes its
    /// lines at: its position in the cube, among the filling planes' units, or in its plane.
    ///
    /// A plane is the launch's `x` ([`Partitioning::cube_dim`]), so a unit's position in its plane
    /// is `UNIT_POS_X` and the plane's width `CUBE_DIM_X`; the filling planes are the cube's last.
    pub(crate) fn worker(#[comptime] fill: FillUnits) -> usize {
        comptime!(assert!(
            fill.scope != ComputeScope::Unit,
            "FillUnits::worker: a cooperative fill is shared by a cube or a plane, never one unit"
        ));
        if fill.fillers > 0 {
            (UNIT_POS - (CUBE_DIM - comptime!(fill.fillers as u32) * CUBE_DIM_X)) as usize
        } else {
            ComputeScope::unit(fill.scope)
        }
    }

    /// How many units `fill` names at runtime: the cube's, the filling planes', or one plane's
    /// ([`worker`](FillUnits::worker)).
    pub(crate) fn workers(#[comptime] fill: FillUnits) -> usize {
        if fill.fillers > 0 {
            return fill.fillers * CUBE_DIM_X as usize;
        }
        match fill.scope {
            ComputeScope::Cube => CUBE_DIM as usize,
            ComputeScope::Plane => CUBE_DIM_X as usize,
            ComputeScope::PlaneGroup { planes } => planes * CUBE_DIM_X as usize,
            ComputeScope::Unit => comptime!(panic!(
                "FillUnits::workers: a cooperative fill is shared by a cube or a plane, never \
                 one unit"
            )),
        }
    }

    /// Hand this unit each of its lines of the `lines` a copy moves, in `turns`: line `w + t·n`
    /// to unit `w` of the `n` that `fill` names, so the units take neighbouring lines at once.
    pub(crate) fn move_lines<M: LineMove>(
        #[comptime] fill: FillUnits,
        lines: usize,
        #[comptime] turns: Turns,
        mover: &mut M,
    ) {
        match turns {
            Turns::Unrolled { count, ragged } => {
                let units = fill.count;
                #[unroll]
                for t in 0..count {
                    let line = FillUnits::worker(fill) + (t * units);
                    if comptime!(ragged && t + 1 == count) {
                        if line < lines {
                            mover.move_line(line, comptime!(Some(t)));
                        }
                    } else {
                        mover.move_line(line, comptime!(Some(t)));
                    }
                }
            }
            Turns::Strided => {
                let stride = FillUnits::workers(fill);
                let mut line = FillUnits::worker(fill);
                while line < lines {
                    mover.move_line(line, comptime!(None));
                    line += stride;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_few_turns_a_unit_are_written_out() {
        assert_eq!(
            FillUnits::cube(64).turns(Some(256)),
            Turns::Unrolled {
                count: 4,
                ragged: false
            }
        );
        assert_eq!(
            FillUnits::cube(64).turns(Some(200)),
            Turns::Unrolled {
                count: 4,
                ragged: true
            }
        );
    }

    #[test]
    fn many_turns_or_an_unknown_count_stride() {
        assert_eq!(FillUnits::cube(64).turns(Some(64 * 9)), Turns::Strided);
        assert_eq!(FillUnits::cube(64).turns(None), Turns::Strided);
        assert_eq!(FillUnits::cube(0).turns(Some(64)), Turns::Strided);
    }

    #[test]
    fn a_copy_held_in_registers_writes_every_turn_out() {
        assert_eq!(FillUnits::cube(64).unrolled(64 * 9).count(), Some(9));
        assert_eq!(FillUnits::cube(64).unrolled(1).count(), Some(1));
    }

    #[test]
    #[should_panic(expected = "spreads its lines over the launch's units")]
    fn turns_written_out_need_the_unit_count() {
        FillUnits::cube(0).unrolled(256);
    }
}
