/// Checks if the tensor associated with the given strides is contiguous.
pub fn is_contiguous(strides: &[usize]) -> bool {
    let mut current = 0;

    for stride in strides.iter().rev() {
        if current > *stride {
            return false;
        }
        current = *stride;
    }

    true
}
