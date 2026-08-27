//! What a launch pays for the taint bookkeeping when nothing goes wrong.
//!
//! The write scope taints every buffer a launch writes on the way in and
//! releases it on the way out, so the success path — every launch of every
//! working program — runs the whole cycle. This measures that cycle, in
//! allocations as much as in time: an allocation per launch is the cost the
//! design set out not to have.

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use cubecl_runtime::memory_management::{ErrorGraph, Taint};
use std::time::Instant;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards every call to the system allocator unchanged; the counter
// is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// One launch's worth of bookkeeping over `buffers` write-set entries: mint
/// the provisional failure, claim every buffer, then release them all.
fn cycle(taints: &mut [Taint], graph: &mut ErrorGraph) {
    let provisional = graph.insert(cubecl_runtime::server::ServerError::TornDown);
    for taint in taints.iter_mut() {
        taint.taint(0..4096, provisional, graph);
    }
    for taint in taints.iter_mut() {
        taint.written(0..4096, graph);
    }
    graph.prune(provisional);
}

fn measure(buffers: usize, launches: usize) -> (usize, Duration) {
    let mut taints: Vec<Taint> = (0..buffers).map(|_| Taint::default()).collect();
    let mut graph = ErrorGraph::default();

    // Warm up, so the numbers are the steady state rather than first touch.
    for _ in 0..1_000 {
        cycle(&mut taints, &mut graph);
    }

    let before = ALLOCS.load(Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..launches {
        cycle(&mut taints, &mut graph);
    }
    let elapsed = start.elapsed();
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert!(
        taints.iter().all(|taint| taint.is_clean()) && graph.is_empty(),
        "the cycle has to leave nothing behind, or it is measuring a leak"
    );
    (allocs, elapsed)
}

fn main() {
    let launches = 200_000;
    println!(
        "Taint is {} bytes, carried by every slice\n",
        core::mem::size_of::<Taint>()
    );
    println!("{launches} launch-shaped cycles, success path\n");
    println!(
        "{:>8}  {:>14}  {:>12}  {:>10}",
        "buffers", "allocs", "allocs/launch", "ns/launch"
    );
    for buffers in [1, 2, 4] {
        let (allocs, elapsed) = measure(buffers, launches);
        println!(
            "{buffers:>8}  {allocs:>14}  {:>13.2}  {:>10.1}",
            allocs as f64 / launches as f64,
            elapsed.as_nanos() as f64 / launches as f64,
        );
    }
}
