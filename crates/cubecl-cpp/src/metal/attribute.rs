use std::fmt::Display;

pub enum BufferAttribute {
    Buffer,
    ThreadGroup,
    None,
}

impl BufferAttribute {
    pub fn indexed_fmt(&self, index: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, " [[{self}({index})]]")
    }
}

impl Display for BufferAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buffer => f.write_str("buffer"),
            Self::ThreadGroup => f.write_str("threadgroup"),
            Self::None => Ok(()),
        }
    }
}

pub enum BuiltInAttribute {
    SIMDgroupIndexInThreadgroup,
    ThreadIndexInSIMDgroup,
    ThreadIndexInThreadgroup,
    ThreadPositionInGrid,
    ThreadPositionInThreadgroup,
    ThreadgroupPositionInGrid,
    ThreadgroupsPerGrid,
    ThreadsPerSIMDgroup,
    ThreadsPerThreadgroup,
    None,
}

impl Display for BuiltInAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BuiltInAttribute::SIMDgroupIndexInThreadgroup => {
                f.write_str("[[simdgroup_index_in_threadgroup]]")
            }
            BuiltInAttribute::ThreadIndexInSIMDgroup => {
                f.write_str("[[thread_index_in_simdgroup]]")
            }
            BuiltInAttribute::ThreadIndexInThreadgroup => {
                f.write_str("[[thread_index_in_threadgroup]]")
            }
            BuiltInAttribute::ThreadPositionInGrid => f.write_str("[[thread_position_in_grid]]"),
            BuiltInAttribute::ThreadPositionInThreadgroup => {
                f.write_str("[[thread_position_in_threadgroup]]")
            }
            BuiltInAttribute::ThreadgroupPositionInGrid => {
                f.write_str("[[threadgroup_position_in_grid]]")
            }
            BuiltInAttribute::ThreadgroupsPerGrid => f.write_str("[[threadgroups_per_grid]]"),
            BuiltInAttribute::ThreadsPerSIMDgroup => f.write_str("[[threads_per_simdgroup]]"),
            BuiltInAttribute::ThreadsPerThreadgroup => f.write_str("[[threads_per_threadgroup]]"),
            BuiltInAttribute::None => Ok(()),
        }
    }
}
