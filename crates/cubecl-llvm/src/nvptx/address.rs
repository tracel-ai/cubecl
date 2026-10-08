//! NVPTX address spaces, and the casts between them.
//!
//! A pointer reaches the lowering in the generic space unless the entry ABI or the shared memory
//! block gave it a specific one. Instructions that only address one space (`ldmatrix`, TMA,
//! `mbarrier`, the wgmma descriptors) cast to it; WMMA loads and stores narrow to the space the
//! pointer came from when that is known, because the specific form is faster than the generic one.

use crate::prelude::*;

/// An NVPTX address space, numbered as the NVPTX backend numbers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NvptxSpace {
    Generic,
    Global,
    /// The cube's own shared memory, `.shared::cta`.
    Shared,
    /// The shared memory of every cube in the cluster, `.shared::cluster`, of which the cube's
    /// own shared memory is a part.
    SharedCluster,
}

impl From<NvptxSpace> for u32 {
    fn from(space: NvptxSpace) -> Self {
        match space {
            NvptxSpace::Generic => 0,
            NvptxSpace::Global => 1,
            NvptxSpace::Shared => 3,
            NvptxSpace::SharedCluster => 7,
        }
    }
}

impl TryFrom<u32> for NvptxSpace {
    type Error = u32;

    fn try_from(number: u32) -> core::result::Result<Self, u32> {
        [
            NvptxSpace::Generic,
            NvptxSpace::Global,
            NvptxSpace::Shared,
            NvptxSpace::SharedCluster,
        ]
        .into_iter()
        .find(|&space| u32::from(space) == number)
        .ok_or(number)
    }
}

impl NvptxSpace {
    /// The space `ptr` is in, when it is an LLVM pointer into one of these.
    pub(crate) fn new(ctx: &Context, ptr: Value) -> Option<Self> {
        let number = ptr
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<LlvmPointerType>()
            .map(LlvmPointerType::address_space)?;
        Self::try_from(number).ok()
    }

    pub(crate) fn pointer_ty(self, ctx: &Context) -> TypeHandle {
        LlvmPointerType::get(ctx, self.into()).into()
    }

    /// `ptr` as a pointer into this space, cast when it is in another. A shared memory pointer
    /// reaches `.shared::cluster` through `.shared::cta`, which is the only cast LLVM lowers.
    pub(crate) fn cast(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        ptr: Value,
    ) -> Value {
        let ptr = match self {
            NvptxSpace::SharedCluster => NvptxSpace::Shared.cast(ctx, rw, ptr),
            _ => ptr,
        };
        let ty = self.pointer_ty(ctx);
        if ptr.get_type(ctx) == ty {
            return ptr;
        }
        let op = llvm::AddrSpaceCastOp::new(ctx, ptr, ty);
        insert(ctx, rw, &op)
    }

    /// `ptr` in the space it was derived from, when that is global or shared memory and the
    /// derivation can be followed. Any other pointer is returned as is.
    pub(crate) fn narrow(
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        ptr: Value,
    ) -> Value {
        match Self::origin(ctx, ptr) {
            Some(space @ (NvptxSpace::Global | NvptxSpace::Shared)) => space.cast(ctx, rw, ptr),
            _ => ptr,
        }
    }

    /// The space `ptr` was derived from, followed back through element offsets and casts to the
    /// generic space, or `None` when the derivation leaves those.
    fn origin(ctx: &Context, ptr: Value) -> Option<Self> {
        let mut ptr = ptr;
        loop {
            match Self::new(ctx, ptr) {
                Some(NvptxSpace::Generic) => {}
                space => return space,
            }
            let op = ptr.defining_op()?;
            let derives_from_its_pointer = Operation::get_op::<llvm::GetElementPtrOp>(op, ctx)
                .is_some()
                || Operation::get_op::<llvm::AddrSpaceCastOp>(op, ctx).is_some();
            if !derives_from_its_pointer {
                return None;
            }
            ptr = op.deref(ctx).get_operand(0);
        }
    }
}
