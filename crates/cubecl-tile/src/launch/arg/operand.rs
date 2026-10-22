//! What a kernel leaves by: [`Output`].

use cubecl::prelude::*;

use crate::{AccumulateArg, AccumulateArgLaunch, Bound, Partitioning, Tile, TileArg};

/// The operand a kernel leaves by: a tensor read and replaced ([`TileArg`]), or an atomic tensor
/// added into and never read ([`AccumulateArg`]).
#[derive(CubeType, CubeLaunch)]
pub enum Output<'a, E: Numeric> {
    Tensor(TileArg<'a>),
    Atomic(AccumulateArg<'a, E>),
}

#[cube]
impl<'a, E: Numeric> Output<'a, E> {
    /// Serve this operand as a [`Tile`] under the kernel's one `partitioning`.
    pub fn tile(&self, partitioning: &Partitioning) -> Tile {
        match self {
            Output::Tensor(arg) => arg.tile(partitioning),
            Output::Atomic(arg) => arg.tile(partitioning),
        }
    }
}

impl Bound {
    /// This operand as an [`Output::Tensor`]: read and replaced.
    pub fn output<E: Numeric>(self) -> OutputArgs<'static, E> {
        OutputArgs::Tensor(self.arg())
    }

    /// This operand as an [`Output::Atomic`]; the buffer must arrive holding zero, since it only
    /// ever adds.
    pub fn atomic<E: Numeric>(self) -> OutputArgs<'static, E> {
        let (spec, width) = (self.spec.clone(), self.vector_size);
        OutputArgs::Atomic(AccumulateArgLaunch::new(self.tensor(), width, spec))
    }
}
