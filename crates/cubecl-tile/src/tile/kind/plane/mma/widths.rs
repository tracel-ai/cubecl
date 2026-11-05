//! The widths a manual-mma fragment's registers take, bound at allocation to the instruction's
//! `vector_size`: one per role, and for a block-scaled fragment its values' per role and its
//! scales'; and a block-scaled fragment's elements, its codes' per role and its scales', bound at
//! allocation to the instruction's form ([`BlockScaledOperands`](crate::BlockScaledOperands)).

use cubecl::prelude::*;

define_size!(pub(crate) NL);
define_size!(pub(crate) NR);
define_size!(pub(crate) NA);
define_size!(pub(crate) NLB);
define_size!(pub(crate) NRB);
define_size!(pub(crate) NSB);
define_scalar!(pub(crate) LhsCodes);
define_scalar!(pub(crate) RhsCodes);
define_scalar!(pub(crate) BlockScale);
