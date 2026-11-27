use crate::tune::{InputGenerator, TuneInputs};

/// An [`InputGenerator`] that clones the reference inputs verbatim.
#[derive(Copy, Clone, Debug, Default)]
pub struct CloneInputGenerator;

impl<K, I: TuneInputs> InputGenerator<K, I> for CloneInputGenerator {
    fn generate<'a>(&self, _key: &K, inputs: &I::At<'a>) -> I::At<'a> {
        inputs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generated_inputs_are_the_reference_inputs() {
        assert_eq!(
            InputGenerator::<(), u32>::generate(&CloneInputGenerator, &(), &7),
            7
        );
    }
}
