//! Binding tensors to a kernel and launching it: the [`Launcher`](crate::Launcher), each operand's
//! comptime `spec`, the `binding` that builds it from a tensor, the `arg`s the kernel receives, and
//! the [`Vectorization`] that settles each operand's width.

pub(crate) mod arg;
pub(crate) mod base;
pub(crate) mod binding;
pub(crate) mod refusal;
pub(crate) mod spec;
pub(crate) mod vectorization;

pub use arg::accumulate::{AccumulateArg, AccumulateArgLaunch};
pub use arg::destination::{
    Buffered, Destination, DestinationLaunch, SinkLaunch, SinkLaunchVisitor, TmaStored,
};
pub use arg::last_arrival::LastArrival;
pub use arg::operand::{Output, OutputArgs};
pub use arg::relay::Relay;
pub use arg::tma::{TmaBox, TmaLanding, TmaOperand, TmaTiles};
pub use base::{Grid, spawnable};
pub use binding::{Arg, Bound, Labelled, Unbound, Unlabelled};
pub use refusal::Refusal;
pub use spec::{BoundaryPolicy, Delivery};
pub use vectorization::{VectorRequirement, Vectorization};
