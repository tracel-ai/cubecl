//! A thread's slot holding a value for the length of one call.

/// Runs `call` with the slot `replace` writes holding `value`, then puts back
/// what the slot held before, however `call` ends.
///
/// `replace` stores what it is given in the slot and returns what the slot
/// held.
pub fn with_value<T, R>(
    replace: fn(Option<T>) -> Option<T>,
    value: T,
    call: impl FnOnce() -> R,
) -> R {
    let _previous = PreviousValue {
        replace,
        value: replace(Some(value)),
    };
    call()
}

/// What a slot held before [`with_value`] wrote it, put back as it drops.
struct PreviousValue<T> {
    replace: fn(Option<T>) -> Option<T>,
    value: Option<T>,
}

impl<T> Drop for PreviousValue<T> {
    fn drop(&mut self) {
        (self.replace)(self.value.take());
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use core::cell::Cell;

    std::thread_local! {
        static SLOT: Cell<Option<u32>> = const { Cell::new(None) };
    }

    fn replace(value: Option<u32>) -> Option<u32> {
        SLOT.with(|slot| slot.replace(value))
    }

    #[test]
    fn the_slot_holds_the_value_during_the_call() {
        let held = with_value(replace, 7, || SLOT.with(Cell::get));
        assert_eq!(held, Some(7));
    }

    #[test]
    fn the_slot_gets_back_what_it_held_after_the_call() {
        let outer = with_value(replace, 1, || {
            with_value(replace, 2, || ());
            SLOT.with(Cell::get)
        });
        assert_eq!(outer, Some(1));
    }

    #[test]
    fn the_slot_gets_back_what_it_held_after_a_panic() {
        let panicked = std::panic::catch_unwind(|| with_value(replace, 3, || panic!("the call")));
        assert!(panicked.is_err());
        assert_eq!(SLOT.with(Cell::get), None);
    }
}
