use std::marker::PhantomData;

use crate::{
    ir::{Instruction, Operator, UnaryOperator},
    prelude::{CubeContext, CubeType, ExpandElement, ExpandElementBaseInit, ExpandElementTyped},
    unexpanded,
};

use super::CubePrimitive;

pub trait AsPtr: CubeType<ExpandType = ExpandElementTyped<Self>> + Sized + 'static {}

/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
pub struct Ptr<E: AsPtr> {
    _e: PhantomData<E>,
}

impl<E: AsPtr> CubeType for Ptr<E> {
    type ExpandType = ExpandElementTyped<Ptr<E>>;
}

impl<E: AsPtr> ExpandElementBaseInit for Ptr<E> {
    fn init_elem(
        _context: &mut crate::prelude::CubeContext,
        elem: crate::prelude::ExpandElement,
    ) -> crate::prelude::ExpandElement {
        elem
    }
}

impl<C: CubePrimitive> AsPtr for C {}

impl<E: AsPtr> Ptr<E> {
    pub fn of(_elem: &E) -> Ptr<E> {
        unexpanded!()
    }

    pub fn as_ref(&self) -> &E {
        unexpanded!()
    }
    pub fn as_ref_mut(&self) -> &mut E {
        unexpanded!()
    }

    pub fn __expand_as_ref(
        _context: &mut CubeContext,
        this: ExpandElementTyped<Ptr<E>>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = this.into();
        elem.into()
    }

    pub fn __expand_as_ref_mut(
        _context: &mut CubeContext,
        this: ExpandElementTyped<Ptr<E>>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = this.into();
        elem.into()
    }
    pub fn __expand_of(
        context: &mut CubeContext,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<Ptr<E>> {
        let elem: ExpandElement = elem.into();
        let var = *elem;

        let out = context.create_ptr(var);
        context.register(Instruction::new(
            Operator::Ptr(UnaryOperator { input: var }),
            *out,
        ));

        out.into()
    }
}
