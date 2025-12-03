use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    global::{
        GlobalReaderConfig,
        read::{
            async_full_tma::AsyncFullTmaLoading, sync_full_cyclic::SyncFullCyclicLoading,
            sync_full_ordered::SyncFullOrderedLoading, sync_full_strided::SyncFullStridedLoading,
            sync_full_tilewise::SyncFullTilewiseLoading,
        },
    },
    stage::TilingOrder,
};

use crate::components::global::{
    args::{RuntimeArgs, RuntimeArgsExpand},
    read::full_reader::FullLoadingStrategy,
};
use cubecl_matmul::components::global::read::FullLoadingStrategy as MatmulFullLoadingStrategy;

macro_rules! impl_full_load_strategy {
    ($( $($ty: ident)::* $(<$($l: lifetime,)* $($T: ident $(: $($(+)? $B: ident)*)?),+>)?,)*) => {
        $(
            impl$(<$($l,)* $($T: CubeType $($(+ $B)*)? ),*>)? FullLoadingStrategy for $($ty)::* $(<$($l,)* $($T),+>)? {
                type TilingLayout = <Self as MatmulFullLoadingStrategy>::TilingLayout;
                type SyncStrategy = <Self as MatmulFullLoadingStrategy>::SyncStrategy;

                type Job<EG: Numeric, ES: Numeric> =
                    <Self as MatmulFullLoadingStrategy>::Job<EG, ES>;

                fn new_job<EG: Numeric, ES: Numeric>(
                    _runtime_args: RuntimeArgs,
                    line_size: u32,
                    config: GlobalReaderConfig,
                ) -> Self::Job<EG, ES> {
                    <Self as MatmulFullLoadingStrategy>::new_job::<EG, ES>(
                        line_size, config,
                    )
                }

                fn __expand_new_job<EG: Numeric, ES: Numeric>(
                    scope: &mut Scope,
                    _runtime_args: RuntimeArgsExpand,
                    line_size: u32,
                    config: GlobalReaderConfig,
                ) -> <Self::Job<EG, ES> as CubeType>::ExpandType {
                    <Self as MatmulFullLoadingStrategy>::__expand_new_job::<EG, ES>(
                        scope, line_size, config,
                    )
                }
            }
        )*
    };
}

// These work as is with the correct layout. They don't need the runtime args to function properly.
impl_full_load_strategy!(
    SyncFullCyclicLoading<TO: TilingOrder>,
    SyncFullStridedLoading,
    SyncFullOrderedLoading,
    SyncFullTilewiseLoading<TO: TilingOrder>,
    AsyncFullTmaLoading,
);
