use crate::ir::{self, Elem, Instruction, Item, ReusingAllocator, Scope, Variable, VariableKind};
use crate::{frontend::ExpandElement, ir::LocalAllocator};
use alloc::rc::Rc;
use core::cell::RefCell;
use cubecl_runtime::debug::DebugLogger;
use std::any::TypeId;
use std::collections::HashMap;

pub struct CubeContext {
    pub root: Rc<RefCell<Scope>>,
    pub scope: Rc<RefCell<Scope>>,
    pub local_allocator: Rc<dyn LocalAllocator>,
    pub debug_enabled: bool,
    pub typemap: Rc<RefCell<HashMap<TypeId, Elem>>>,
}

impl Default for CubeContext {
    fn default() -> Self {
        Self::root(ReusingAllocator::default())
    }
}

impl CubeContext {
    /// Create a new cube context, with a root scope
    /// A root scope is at the root of a compute shader
    /// Therefore there is one cube context per shader
    /// The allocator will define the strategy for creating local intermediates and mutable variables
    pub fn root(allocator: impl LocalAllocator + 'static) -> CubeContext {
        let root = Rc::new(RefCell::new(Scope::root()));
        let typemap = Rc::new(RefCell::new(HashMap::new()));
        let scope = root.clone();

        Self {
            local_allocator: Rc::new(allocator),
            scope,
            root,
            debug_enabled: DebugLogger::default().is_activated(),
            typemap,
        }
    }

    pub fn register<O: Into<Instruction>>(&mut self, op: O) {
        self.scope.borrow_mut().register(op)
    }

    /// Resolve the element type of the given generic type.
    pub fn resolve_elem<T: 'static>(&self) -> Option<Elem> {
        let map = self.typemap.borrow();
        let result = map.get(&TypeId::of::<T>());

        result.cloned()
    }

    /// Register the element type for the given generic type.
    pub fn register_elem<T: 'static>(&mut self, elem: Elem) {
        let mut map = self.typemap.borrow_mut();

        map.insert(TypeId::of::<T>(), elem);
    }

    pub fn child(&mut self) -> CubeContext {
        let scope = self.scope.borrow_mut().child();

        Self {
            scope: Rc::new(RefCell::new(scope)),
            root: self.root.clone(),
            local_allocator: self.local_allocator.clone(),
            debug_enabled: self.debug_enabled,
            typemap: self.typemap.clone(),
        }
    }

    pub fn into_scope(self) -> Scope {
        core::mem::drop(self.root);

        Rc::into_inner(self.scope)
            .expect("Only one reference")
            .into_inner()
    }

    /// Create a new mutable local variable
    pub fn create_local_variable(&mut self, item: Item) -> ExpandElement {
        self.local_allocator
            .create_local_variable(self.root.clone(), self.scope.clone(), item)
    }

    /// Create a new immutable local binding
    pub fn create_local_binding(&mut self, item: Item) -> ExpandElement {
        self.local_allocator
            .create_local_binding(self.root.clone(), self.scope.clone(), item)
    }

    /// Create a new immutable local binding that must never be a reused variable, regardless of
    /// allocator
    pub fn create_local_undeclared(&mut self, item: Item) -> ExpandElement {
        self.local_allocator
            .create_local_undeclared(self.root.clone(), self.scope.clone(), item)
    }

    /// Create a new matrix element.
    pub fn create_matrix(&mut self, matrix: ir::Matrix) -> ExpandElement {
        let variable = self.scope.borrow_mut().create_matrix(matrix);
        ExpandElement::Plain(variable)
    }

    /// Create a new slice element.
    pub fn create_slice(&mut self, item: Item) -> ExpandElement {
        let variable = self.scope.borrow_mut().create_slice(item);
        ExpandElement::Plain(variable)
    }

    pub fn create_shared(&mut self, item: Item, size: u32) -> ExpandElement {
        ExpandElement::Plain(self.root.borrow_mut().create_shared(item, size))
    }

    pub fn create_local_array(&mut self, item: Item, size: u32) -> ExpandElement {
        ExpandElement::Plain(self.root.borrow_mut().create_local_array(item, size))
    }

    pub fn create_const_array(&mut self, item: Item, data: Vec<Variable>) -> ExpandElement {
        ExpandElement::Plain(self.root.borrow_mut().create_const_array(item, data))
    }

    /// Obtain the index-th input
    pub fn input(&mut self, id: u16, item: Item) -> ExpandElement {
        ExpandElement::Plain(crate::ir::Variable::new(
            VariableKind::GlobalInputArray(id),
            item,
        ))
    }

    /// Obtain the index-th output
    pub fn output(&mut self, id: u16, item: Item) -> ExpandElement {
        let var = crate::ir::Variable::new(VariableKind::GlobalOutputArray(id), item);
        self.scope.borrow_mut().write_global_custom(var);
        ExpandElement::Plain(var)
    }

    /// Obtain the index-th scalar
    pub fn scalar(&self, id: u16, elem: Elem) -> ExpandElement {
        ExpandElement::Plain(crate::ir::Variable::new(
            VariableKind::GlobalScalar(id),
            Item::new(elem),
        ))
    }
}
