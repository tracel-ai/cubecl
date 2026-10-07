//! Source locations of the ops that kernel expansion inserts.
//!
//! The macro-generated code of each `#[cube]` function opens a [frame](DebugState::enter_fn) and
//! moves its [position](DebugState::set_pos). [`LocationListener`] gives each inserted op the
//! location of the current frame, nested in the call sites of its callers.
//!
//! Expansion inlines every `#[cube]` call, so the call chain is the only record of the source-level
//! call stack. For an op in `inner`, called from `outer`, called from kernel `k`:
//!
//! ```text
//! CallSite {
//!     callee: Named("inner", SrcPos(op)),
//!     caller: CallSite {
//!         callee: Named("outer", SrcPos(call to inner)),
//!         caller: Named("k", SrcPos(call to outer)),
//!     },
//! }
//! ```

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec::Vec,
};
use pliron::{
    basic_block::BasicBlock,
    combine::stream::position::SourcePosition,
    context::{Context, Ptr},
    irbuild::listener::InsertionListener,
    location::{Located, Location, Source},
    operation::Operation,
    std_deps::path::PathBuf,
};

use crate::ContextExt;

/// The `#[cube]` call stack during kernel expansion.
///
/// A function without debug data, such as a hand-written expand function, opens no frame. Its ops
/// get the location of the call to it.
#[derive(Debug, Default)]
pub struct DebugState {
    enabled: bool,
    frames: Vec<Frame>,
    /// The source text of each file, by the path of its [`Source`]. Only full debug data records
    /// it.
    sources: BTreeMap<String, &'static str>,
    /// The path of each file, at all levels.
    files: BTreeSet<String>,
}

/// One `#[cube]` function on the call stack.
#[derive(Debug)]
struct Frame {
    name: &'static str,
    file: Source,
    /// The expression the function is at: the call site, while a callee runs.
    pos: SourcePosition,
}

impl DebugState {
    /// An empty call stack that records locations only when `enabled`.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    /// Whether expansion records locations.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Opens the frame of function `name`, defined at `line` and `column` of `file`.
    pub fn enter_fn(&mut self, name: &'static str, file: Source, line: u32, column: u32) {
        self.frames.push(Frame {
            name,
            file,
            pos: position(line, column),
        });
    }

    /// Closes the innermost frame.
    pub fn exit_fn(&mut self) {
        self.frames.pop();
    }

    /// Moves the innermost frame to `line` and `column`.
    pub fn set_pos(&mut self, line: u32, column: u32) {
        if let Some(frame) = self.frames.last_mut() {
            frame.pos = position(line, column);
        }
    }

    /// Records the file `path`, and `text` as its source text. An empty text records only the
    /// path.
    pub fn add_source(&mut self, path: &str, text: &'static str) {
        if !self.files.contains(path) {
            self.files.insert(path.into());
        }
        if !text.is_empty() && !self.sources.contains_key(path) {
            self.sources.insert(path.into(), text);
        }
    }

    /// The source text of each file, by the path of its [`Source`]. It stays after
    /// [`finish`](Self::finish), for the compiler.
    #[must_use]
    pub fn sources(&self) -> &BTreeMap<String, &'static str> {
        &self.sources
    }

    /// The path of each file, with or without its source text. It stays after
    /// [`finish`](Self::finish), for the compiler.
    #[must_use]
    pub fn files(&self) -> &BTreeSet<String> {
        &self.files
    }

    /// Stops recording, at the end of kernel expansion. Ops that compiler passes insert later get
    /// the location of the op they replace.
    pub fn finish(&mut self) {
        self.enabled = false;
        self.frames.clear();
    }

    /// The location of the current expression, or `None` outside a `#[cube]` function.
    fn location(&self) -> Option<Location> {
        let (kernel, callees) = self.frames.split_first()?;
        let location =
            callees
                .iter()
                .fold(kernel.location(), |caller, callee| Location::CallSite {
                    callee: Box::new(callee.location()),
                    caller: Box::new(caller),
                });
        Some(location)
    }
}

impl Frame {
    fn location(&self) -> Location {
        Location::Named {
            name: self.name.into(),
            child_loc: Box::new(Location::SrcPos {
                src: self.file,
                pos: self.pos,
            }),
        }
    }
}

/// The position at `line` and `column`. A value past `i32::MAX`, which no source file reaches,
/// saturates.
fn position(line: u32, column: u32) -> SourcePosition {
    SourcePosition {
        line: i32::try_from(line).unwrap_or(i32::MAX),
        column: i32::try_from(column).unwrap_or(i32::MAX),
    }
}

/// Gives each inserted op without a location the location of the current expression.
#[derive(Default)]
pub struct LocationListener;

impl InsertionListener for LocationListener {
    fn notify_operation_inserted(&mut self, ctx: &Context, operation: Ptr<Operation>) {
        let Some(debug) = ctx.try_aux_ty::<DebugState>() else {
            return;
        };
        if let Some(loc) = debug.location()
            && operation.deref(ctx).loc().is_unknown()
        {
            operation.deref_mut(ctx).set_loc(loc);
        }
    }

    fn notify_block_inserted(&mut self, _ctx: &Context, _block: Ptr<BasicBlock>) {}
}

/// The source line of an op: the file and the line of the innermost frame of `loc`. A target that
/// cannot show inlined frames, such as a `#line` directive, uses it. `None` for an unknown
/// location, or for a position in memory.
pub fn leaf_line<'c>(ctx: &'c Context, loc: &Location) -> Option<(&'c PathBuf, u32)> {
    match loc {
        Location::CallSite { callee, .. } => leaf_line(ctx, callee),
        Location::Named { child_loc, .. } => leaf_line(ctx, child_loc),
        Location::SrcPos {
            src: Source::File(key),
            pos,
        } => Some((
            pliron::uniqued_any::get(ctx, *key),
            u32::try_from(pos.line).unwrap_or(0),
        )),
        Location::Fused { locations, .. } => locations.iter().find_map(|loc| leaf_line(ctx, loc)),
        Location::SrcPos {
            src: Source::InMemory,
            ..
        }
        | Location::Unknown => None,
    }
}
