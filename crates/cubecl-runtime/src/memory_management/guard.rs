use super::ManagedMemoryBinding;
use alloc::sync::Arc;

/// Keeps a page as it is for as long as it lives: it is never released,
/// relocation leaves everything on it in place, and its free room is carved
/// only by the temporaries of graph captures on the stream whose memory holds
/// it.
///
/// What a graph holds on the memory it replays against. A graph keeps raw
/// addresses, so the bytes behind them must neither move nor be handed to
/// work that outlives a replay. What the graph writes past its capture is held
/// by the graph's own bindings, and what it reads by the caller, who keeps a
/// graph's inputs alive for as long as it replays; the room its temporaries used
/// is free on the page, and the guard keeps it from everything but another
/// capture's temporaries. Those replay in stream order with this graph, so
/// overwriting each other between replays costs nothing. The allocations on
/// the page keep their handles untouched, so one can still be updated in
/// place while the page is guarded.
///
/// A resolved resource needs no guard: the binding it holds keeps its own
/// allocation in place, and leaves the rest of the page to other allocations.
#[derive(Debug, Clone)]
pub struct PageGuard {
    _hold: Hold,
}

#[derive(Debug, Clone)]
enum Hold {
    /// A page carved into slices, which counts the guards it hands out.
    Page { _token: Arc<()> },
    /// A slice that is its own device allocation. Holding its binding is what
    /// keeps it from being handed out again or released, whether an
    /// allocation is on it or not.
    Allocation { _binding: ManagedMemoryBinding },
}

impl PageGuard {
    /// A guard on a page that counts its guards with `token`.
    #[doc(hidden)]
    pub fn page(token: &Arc<()>) -> Self {
        Self {
            _hold: Hold::Page {
                _token: token.clone(),
            },
        }
    }

    /// A guard on a page that is one allocation, held through `binding`.
    #[doc(hidden)]
    pub fn allocation(binding: ManagedMemoryBinding) -> Self {
        Self {
            _hold: Hold::Allocation { _binding: binding },
        }
    }
}
