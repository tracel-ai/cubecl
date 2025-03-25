/// Checks if the tensor associated with the given strides is contiguous.
pub fn is_contiguous(strides: &[usize]) -> bool {
    let mut current = 1;

    for stride in strides.iter().rev() {
        if current > *stride {
            return false;
        }
        current = *stride;
    }

    true
}

pub fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
