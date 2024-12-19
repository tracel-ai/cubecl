use super::CubeContext;

pub trait TypeMap<const POS: u8> {
    type ExpandGeneric;

    fn register(context: &mut CubeContext);
}
