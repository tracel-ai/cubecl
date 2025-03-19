#[derive(Clone)]
pub enum BufferId {
    A,
    B,
}

impl BufferId {
    pub fn to_u32(&self) -> u32 {
        match self {
            BufferId::A => 0,
            BufferId::B => 1,
        }
    }
}
