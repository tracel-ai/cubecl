//! The expand form of a type the kernel only ever holds at compile time.

/// Makes a comptime-only type expand as itself.
macro_rules! comptime_only {
    ($ty:ty) => {
        impl CubeType for $ty {
            type ExpandType = Self;
        }

        impl IntoExpand for $ty {
            type Expand = Self;

            fn into_expand(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl IntoMut for $ty {
            fn into_mut(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                Clone::clone(self)
            }
        }

        impl CubeDebug for $ty {}

        impl AsRefExpand for $ty {
            fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
                self
            }
        }

        impl AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
                self
            }
        }
    };
}

pub(crate) use comptime_only;
