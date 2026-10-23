//! A compute plane's walk over stages whose reads outlast the consume that issues them
//! ([`Stages::consume_async`]), a warpgroup MMA's.

use cubecl::prelude::*;

use super::payload::pair::OperandPair;
use crate::*;

/// What a compute plane does with each stage of a walk whose reads run on after it returns: it
/// issues them, and hands back their completion, which frees the stage once it resolves.
#[cube]
pub trait AsyncStage: CubeType {
    /// Issue the reads of the stage `lhs` and `rhs` hold for `region`, and return their
    /// completion.
    fn issue(&self, region: &Region, lhs: &Tile, rhs: &Tile) -> Pending<()>;

    /// A completion of the same work with nothing in flight, which is done at once: what a slot
    /// holds before the walk first reads it.
    fn done(&self) -> Pending<()>;
}

#[cube]
impl Stages<OperandPair> {
    /// A compute plane's side of `walk`, whose stages the filling planes fill by TMA: each stage
    /// issued through `stage`, and its slot freed only once what it issued completes. That is a
    /// stage later, so the reads of the previous stage run on beside the ones just issued, where
    /// the ring holds more than one slot; a ring of one waits for them at once.
    pub fn consume_async<S: AsyncStage>(&mut self, walk: &Walk, stage: &S) {
        let lhs = self.sources.lhs.delivery();
        let rhs = self.sources.rhs.delivery();
        comptime!(assert!(
            self.fillers > 0 && lhs == Delivery::Tma && rhs == Delivery::Tma,
            "Stages::consume_async: reads that outlast their consume see a stage only the TMA \
             engine filled, beside planes of their own that wait for its release; these stages \
             are filled by {} plane(s) of their own, their sources delivered {lhs:?} and {rhs:?}",
            self.fillers
        ));
        let depth = comptime!(self.depth);
        let total = walk.total();
        let laps = (total + comptime!(depth - 1)) / comptime!(depth);
        // Each slot's last issued reads. A ring of one holds none: it waits for them at once.
        let mut issued = Sequence::<Pending<()>>::new();
        if comptime!(depth > 1) {
            #[unroll]
            for _ in 0..depth {
                // A slot assigned again holds a variable, not the value itself.
                #[allow(unused_mut)]
                let mut done = stage.done();
                issued.push(done);
            }
        }
        for lap in 0..laps {
            #[unroll]
            for slot in 0..depth {
                let index = lap * comptime!(depth) + slot;
                if index < total {
                    let region = walk.region(index);
                    let held = self.slots.index_mut(slot);
                    held.acquire_read();
                    let reads = stage.issue(&region, &held.data.lhs, &held.data.rhs);
                    if comptime!(depth == 1) {
                        reads.wait();
                        held.release_read();
                    } else {
                        let previous = comptime!((slot + depth - 1) % depth);
                        issued.index(previous).wait();
                        // The first stage has no slot before it to free.
                        if index > 0 {
                            self.slots.index_mut(previous).release_read();
                        }
                        *issued.index_mut(slot) = reads;
                    }
                }
            }
        }
        // The last stage has no stage after it to free it: free it once its reads complete, as
        // `consume` frees every stage, so the ring is whole again for another walk.
        if comptime!(depth > 1) {
            let last = (total + comptime!(depth - 1)) % comptime!(depth);
            #[unroll]
            for slot in 0..depth {
                if total > 0 && slot == last {
                    issued.index(slot).wait();
                    self.slots.index_mut(slot).release_read();
                }
            }
        }
    }
}
