/// Describes the set of inputs a [`TunableSet`](crate::tune::TunableSet) accepts.
///
/// The associated type [`At`](Self::At) gives the concrete input type at a specific
/// lifetime. The indirection exists because a [`TunableSet`](crate::tune::TunableSet) is
/// `'static`, but some callers want to pass *borrowed* inputs. A `'static` marker type with a
/// lifetime-parameterized `At` works here: the set is `'static`, but every
/// tunable function accepts `Self::At<'a>` for any `'a` via HRTB.
///
/// Implementing it manually is needed when the inputs are genuinely borrowed. Example:
/// burn's fusion autotune passes `TuneInput<'a, R, O>` (which wraps `&'a mut Context`)
/// by defining a `FusionTuneInputs<R, O>` marker with
/// `At<'a> = TuneInput<'a, R, O>`. Such a marker must not implement `Clone`, otherwise
/// it would overlap with the blanket impl below. Owned inputs, through that blanket
/// implementation, have a clone for a spare.
///
/// # Guarantees
///
/// - Every implementation's [`spare`](Self::spare) is cheap: every call to a decided candidate
///   takes one.
pub trait TuneInputs: Send + Sync + 'static {
    /// What a candidate is called with.
    type At<'a>: Clone + Send;

    /// A copy of a call's inputs, to run another candidate on when the decided one rejects
    /// the call.
    ///
    /// `None`, the default, is for inputs whose copy cannot stand in for the call's, such as ones
    /// that borrow what the candidate writes: their decided candidate must serve every call of
    /// its key.
    fn spare<'a>(_inputs: &Self::At<'a>) -> Option<Self::At<'a>> {
        None
    }
}

impl<T: Clone + Send + Sync + 'static> TuneInputs for T {
    type At<'a> = T;

    fn spare<'a>(inputs: &Self::At<'a>) -> Option<Self::At<'a>> {
        Some(inputs.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_inputs_have_a_spare() {
        assert_eq!(<u32 as TuneInputs>::spare(&7), Some(7));
    }

    #[test]
    fn inputs_implemented_by_hand_have_no_spare() {
        struct Borrowed;

        impl TuneInputs for Borrowed {
            type At<'a> = &'a u32;
        }

        assert_eq!(Borrowed::spare(&&7), None);
    }
}
