//! Host model of a barrier slot's protocol, checked over every interleaving of producers and
//! consumers for deadlock, early reads and early refills.

use std::collections::HashSet;

/// One mbarrier: arrivals per phase, arrivals so far, and the phase's parity.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Mbarrier {
    arrivals: u32,
    arrived: u32,
    parity: u32,
}

impl Mbarrier {
    fn new(arrivals: u32) -> Mbarrier {
        Mbarrier {
            arrivals,
            arrived: 0,
            parity: 0,
        }
    }

    fn arrive(&mut self) {
        self.arrived += 1;
        if self.arrived == self.arrivals {
            self.arrived = 0;
            self.parity ^= 1;
        }
    }

    fn passes(&self, waited: u32) -> bool {
        self.parity != waited
    }
}

/// One slot: its two barriers, the region it holds, and its current readers.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Slot {
    full: Mbarrier,
    empty: Mbarrier,
    holds: Option<usize>,
    readers: u32,
}

/// One unit's place in its side's walk.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Unit {
    step: usize,
    waiting: bool,
}

impl Unit {
    fn new() -> Unit {
        Unit {
            step: 0,
            waiting: true,
        }
    }

    fn parity(&self, depth: usize) -> u32 {
        ((self.step / depth) % 2) as u32
    }
}

/// The protocol's whole state.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct State {
    slots: Vec<Slot>,
    producers: Vec<Unit>,
    consumers: Vec<Unit>,
}

/// The shape of one run; producer 0 is the elected unit, the only one that writes.
#[derive(Clone, Copy, Debug)]
struct Shape {
    depth: usize,
    regions: usize,
    producers: u32,
    consumers: u32,
    /// Whether `full` counts every producer's arrival or the elected unit's alone.
    publishes_all: bool,
}

impl Shape {
    fn publishers(&self) -> u32 {
        if self.publishes_all {
            self.producers
        } else {
            1
        }
    }

    fn start(&self) -> State {
        State {
            slots: (0..self.depth)
                .map(|_| Slot {
                    full: Mbarrier::new(self.publishers()),
                    empty: Mbarrier::new(self.consumers),
                    holds: None,
                    readers: 0,
                })
                .collect(),
            producers: (0..self.producers).map(|_| Unit::new()).collect(),
            consumers: (0..self.consumers).map(|_| Unit::new()).collect(),
        }
    }

    fn done(&self, state: &State) -> bool {
        let walked = |units: &Vec<Unit>| units.iter().all(|u| u.step == self.regions && u.waiting);
        walked(&state.producers) && walked(&state.consumers)
    }

    /// Walk every reachable state, checking the protocol; returns the state count.
    fn explore(&self) -> usize {
        self.explore_from(self.start())
    }

    /// [`explore`](Shape::explore) from a caller-supplied state.
    fn explore_from(&self, start: State) -> usize {
        let mut seen = HashSet::new();
        let mut stack = vec![start.clone()];
        seen.insert(start);

        while let Some(state) = stack.pop() {
            let next = state.moves(*self);
            if next.is_empty() {
                assert!(
                    self.done(&state),
                    "every unfinished unit is waiting and none can pass: {self:?} deadlocks at \
                     {state:?}"
                );
                for (s, slot) in state.slots.iter().enumerate() {
                    assert_eq!(
                        slot.full.parity, slot.empty.parity,
                        "slot {s} was filled and read a different number of times"
                    );
                    assert_eq!(slot.readers, 0, "slot {s} ends with a reader in it");
                }
                continue;
            }
            for state in next {
                if seen.insert(state.clone()) {
                    stack.push(state);
                }
            }
        }

        seen.len()
    }
}

impl State {
    /// Every state one unit's move could lead to.
    fn moves(&self, shape: Shape) -> Vec<State> {
        let mut next = Vec::new();

        for u in 0..self.producers.len() {
            let unit = self.producers[u].clone();
            if unit.step == shape.regions {
                continue;
            }
            let slot = unit.step % shape.depth;
            let mut after = self.clone();
            if unit.waiting {
                if !after.slots[slot].empty.passes(unit.parity(shape.depth) ^ 1) {
                    continue;
                }
                // Only the elected unit writes, so only its pass must mean the slot is free.
                if u == 0 {
                    assert!(
                        after.slots[slot].holds.is_none() || after.slots[slot].readers == 0,
                        "a slot was refilled while a consumer was still reading it"
                    );
                    after.slots[slot].holds = Some(unit.step);
                }
                after.producers[u].waiting = false;
            } else {
                if shape.publishes_all || u == 0 {
                    after.slots[slot].full.arrive();
                }
                after.producers[u].step += 1;
                after.producers[u].waiting = true;
            }
            next.push(after);
        }

        for u in 0..self.consumers.len() {
            let unit = self.consumers[u].clone();
            if unit.step == shape.regions {
                continue;
            }
            let slot = unit.step % shape.depth;
            let mut after = self.clone();
            if unit.waiting {
                if !after.slots[slot].full.passes(unit.parity(shape.depth)) {
                    continue;
                }
                assert_eq!(
                    after.slots[slot].holds,
                    Some(unit.step),
                    "a slot was read holding a region other than the one being walked"
                );
                after.slots[slot].readers += 1;
                after.consumers[u].waiting = false;
            } else {
                after.slots[slot].readers -= 1;
                after.slots[slot].empty.arrive();
                after.consumers[u].step += 1;
                after.consumers[u].waiting = true;
            }
            next.push(after);
        }

        next
    }
}

#[test]
fn the_two_sides_agree_however_they_interleave() {
    for depth in 1..=3 {
        for regions in 1..=4 {
            for consumers in 1..=3 {
                let shape = Shape {
                    depth,
                    regions,
                    producers: 2,
                    consumers,
                    publishes_all: true,
                };
                assert!(shape.explore() > 0);
            }
        }
    }
}

#[test]
fn a_single_slot_never_runs_ahead() {
    let shape = Shape {
        depth: 1,
        regions: 3,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    shape.explore();
}

#[test]
fn the_producer_runs_at_most_a_ring_ahead() {
    let shape = Shape {
        depth: 2,
        regions: 4,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    let mut seen = HashSet::new();
    let mut stack = vec![shape.start()];
    seen.insert(shape.start());
    let mut lead = 0;
    while let Some(state) = stack.pop() {
        lead = lead.max(state.producers[0].step - state.consumers[0].step);
        for state in state.moves(shape) {
            if seen.insert(state.clone()) {
                stack.push(state);
            }
        }
    }
    assert_eq!(lead, shape.depth);
}

/// `empty` armed for more units than read it deadlocks.
#[test]
#[should_panic(expected = "deadlocks")]
fn a_slot_freed_by_fewer_units_than_it_waits_for_hangs() {
    let shape = Shape {
        depth: 1,
        regions: 2,
        producers: 1,
        consumers: 1,
        publishes_all: true,
    };
    let mut start = shape.start();
    start.slots[0].empty.arrivals = 2;
    shape.explore_from(start);
}

/// `full` published by the elected unit alone lets a second filling plane drift and deadlock.
#[test]
#[should_panic(expected = "deadlocks")]
fn a_slot_published_by_the_elected_unit_alone_lets_a_second_filling_plane_drift() {
    let shape = Shape {
        depth: 1,
        regions: 1,
        producers: 2,
        consumers: 1,
        publishes_all: false,
    };
    shape.explore();
}
