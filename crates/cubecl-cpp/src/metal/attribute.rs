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
            Self::ThreadIndexInSIMDgroup => f.write_str("[[thread_index_in_simdgroup]]"),
            Self::ThreadIndexInThreadgroup => f.write_str("[[thread_index_in_threadgroup]]"),
            Self::ThreadPositionInGrid => f.write_str("[[thread_position_in_grid]]"),
            Self::ThreadPositionInThreadgroup => f.write_str("[[thread_position_in_threadgroup]]"),
            Self::ThreadgroupPositionInGrid => f.write_str("[[threadgroup_position_in_grid]]"),
            Self::ThreadgroupsPerGrid => f.write_str("[[threadgroups_per_grid]]"),
            Self::ThreadsPerSIMDgroup => f.write_str("[[threads_per_simdgroup]]"),
            Self::ThreadsPerThreadgroup => f.write_str("[[threads_per_threadgroup]]"),
            Self::None => Ok(()),
        }
    }
}
