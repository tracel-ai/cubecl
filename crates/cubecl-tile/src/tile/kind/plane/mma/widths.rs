//! The widths a manual-mma fragment's registers take, bound at allocation to the instruction's
//! `vector_size`: one per role, and for a block-scaled fragment its values' per role and its
//! scales'.

use cubecl::prelude::*;

define_size!(pub(crate) NL);
define_size!(pub(crate) NR);
define_size!(pub(crate) NA);
define_size!(pub(crate) NLB);
define_size!(pub(crate) NRB);
define_size!(pub(crate) NSB);
