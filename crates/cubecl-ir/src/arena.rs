use alloc::vec::Vec;

use bumpalo::Bump;
use cubecl_macros_internal::TypeHash;

/// Bump allocator that runs drop on non-trivially droppable values on reset.
/// This is useful to prevent memory leaks without drastically affecting performance for the vast
/// majority of values that do not need custom drops.
///
/// # SAFETY
///
/// Values inserted *must not* access external borrowed data in a custom `Drop` implementation.
/// This shouldn't happen for normal types, but should be kept in mind regardless.
/// Contrived Example:
///
/// ```no_run
/// # use cubecl_ir::arena::DropBump;
/// struct Wrapper<'a>(&'a mut String);
///
/// impl<'a> Drop for Wrapper<'a> {
///     fn drop(&mut self) {
///         self.0.push_str(" dropped"); // Accesses external borrow
///     }
/// }
///
/// let mut external = String::from("hello");
///
/// let mut pool = DropBump::new();
/// let w = pool.alloc(Wrapper(&mut external));
/// drop(external);
/// pool.reset(); // Runs `drop_in_place::<Wrapper>`
/// ```
///
/// So don't do that.
///
#[derive(Default, Debug, TypeHash)]
pub struct DropBump {
    bump: Bump,
    drop_thunks: Vec<DropThunk>,
}

#[derive(Debug, TypeHash)]
struct DropThunk {
    func: fn(*mut ()),
    ptr: *mut (),
}

impl DropBump {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn reset(&mut self) {
        for drop in self.drop_thunks.drain(..).rev() {
            (drop.func)(drop.ptr)
        }
        self.bump.reset();
    }

    pub fn alloc<T>(&mut self, val: T) -> &mut T {
        let ret = self.bump.alloc(val);

        if core::mem::needs_drop::<T>() {
            self.drop_thunks.push(DropThunk {
                func: drop_glue::<T>,
                ptr: ret as *mut T as *mut (),
            });
        }

        ret
    }
}

fn drop_glue<T>(ptr: *mut ()) {
    unsafe { core::ptr::drop_in_place::<T>(ptr as *mut T) };
}
