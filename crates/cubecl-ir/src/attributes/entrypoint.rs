use core::cell::{Ref, RefMut};

use alloc::boxed::Box;
use derive_new::new;
use pliron::{
    attribute::{AttrObj, Attribute, AttributeDict},
    builtin::{
        attributes::{DictAttr, UnitAttr, VecAttr},
        ops::FuncOp,
    },
    dict_key,
    identifier::Identifier,
};

use crate::{prelude::*, settings::Dim3};

#[pliron_attr(
    name = "cube.entrypoint_abi",
    format = "`<cube_dim: ` $cube_dim opt($cluster_dim) `>`",
    verifier = "succ"
)]
#[derive(new, PartialEq, Clone, Debug)]
pub struct EntrypointAbiAttr {
    pub cube_dim: Dim3,
    pub cluster_dim: Option<Dim3>,
}

dict_key!(
    /// Key for entry point attribute when the operation defines an entry point.
    ATTR_KEY_ENTRY_POINT, "entry_point"
);
dict_key!(ATTR_BUFFER_BINDING, "buffer_binding");
dict_key!(ATTR_TENSOR_MAP_BINDING, "tensor_map_binding");

#[pliron_attr(name = "cube.buffer_io", format, verifier = "succ")]
#[derive(new, PartialEq, Clone, Copy, Debug)]
pub enum BufferIOAttr {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Dead,
}

impl BufferIOAttr {
    pub fn is_readable(&self) -> bool {
        matches!(self, BufferIOAttr::ReadOnly | BufferIOAttr::ReadWrite)
    }

    pub fn is_writable(&self) -> bool {
        matches!(self, BufferIOAttr::WriteOnly | BufferIOAttr::ReadWrite)
    }

    pub fn is_dead(&self) -> bool {
        matches!(self, BufferIOAttr::Dead)
    }
}

dict_key!(ATTR_BUFFER_IO, "binding_io");

#[pliron_attr(
    name = "cube.buffer_binding",
    format = "`<(` $buffer_pos `, ` opt($ext_meta_pos) `)`",
    verifier = "succ"
)]
#[derive(new, PartialEq, Clone, Copy, Debug)]
pub struct BufferBindingAttr {
    pub buffer_pos: usize,
    pub ext_meta_pos: Option<usize>,
}

/// [Op] that may define an entry point.
///
/// ### Attribute(s):
/// | Name | Static Name Identifier | Type |
/// |------|------------------------| -----|
/// | entry_point | [ATTR_KEY_ENTRY_POINT] | [EntrypointAbiAttr](crate::attributes::EntrypointAbiAttr) |
#[op_interface]
pub trait EntrypointInterface {
    verify_op_succ!();

    /// Get the entry point ABI defined by this operation.
    fn get_entrypoint_abi(&self, ctx: &Context) -> Option<EntrypointAbiAttr> {
        let self_op = self.get_operation().deref(ctx);
        let s_attr = self_op
            .attributes
            .get::<EntrypointAbiAttr>(&ATTR_KEY_ENTRY_POINT);
        s_attr.cloned()
    }

    /// Set the entry point ABI defined by this operation.
    fn set_entrypoint_abi(&self, ctx: &mut Context, entry_point_abi: EntrypointAbiAttr) {
        let mut self_op = self.get_operation().deref_mut(ctx);
        self_op
            .attributes
            .set(ATTR_KEY_ENTRY_POINT.clone(), entry_point_abi);
    }
}

#[op_interface_impl]
impl EntrypointInterface for FuncOp {}

dict_key!(ATTR_KEY_ARG_ATTRS, "arg_attrs");
dict_key!(ATTR_KEY_RES_ATTRS, "res_attrs");

/// [Op] that may define an entry point.
///
/// ### Attribute(s):
/// | Name | Static Name Identifier | Type |
/// |------|------------------------| -----|
/// | arg_attrs | [ATTR_KEY_ARG_ATTRS] | [VecAttr](pliron::builtin::attributes::VecAttr) |
/// | res_attrs | [ATTR_KEY_RES_ATTRS] | [VecAttr](pliron::builtin::attributes::VecAttr) |
pub trait FuncInterface: Op {
    fn get_arg_attrs<'a>(&self, ctx: &'a Context, arg_idx: usize) -> Option<Ref<'a, DictAttr>> {
        let self_op = self.get_operation().deref(ctx);
        Ref::filter_map(self_op, |self_op| {
            let args_attrs = self_op.attributes.get::<VecAttr>(&ATTR_KEY_ARG_ATTRS)?;
            args_attrs.0.get(arg_idx)?.downcast_ref()
        })
        .ok()
    }

    fn get_arg_attr<'a, T: Attribute>(
        &self,
        ctx: &'a Context,
        arg_idx: usize,
        key: &Identifier,
    ) -> Option<Ref<'a, T>> {
        let arg_attrs = self.get_arg_attrs(ctx, arg_idx)?;
        Ref::filter_map(arg_attrs, |arg_attrs| {
            arg_attrs.lookup(key).and_then(|attr| attr.downcast_ref())
        })
        .ok()
    }

    fn has_arg_attr(&self, ctx: &Context, arg_idx: usize, key: &Identifier) -> bool {
        self.get_arg_attrs(ctx, arg_idx)
            .is_some_and(|arg_attrs| arg_attrs.lookup(key).is_some())
    }

    fn get_arg_attrs_mut<'a>(&self, ctx: &'a Context, arg_idx: usize) -> RefMut<'a, DictAttr> {
        let self_op = self.get_operation().deref_mut(ctx);
        RefMut::map(self_op, |self_op| {
            get_arg_or_init_mut(&mut self_op.attributes, arg_idx)
        })
    }

    fn get_res_attrs<'a>(&self, ctx: &'a Context, res_idx: usize) -> Option<Ref<'a, DictAttr>> {
        let self_op = self.get_operation().deref(ctx);
        Ref::filter_map(self_op, |self_op| {
            let res_attrs = self_op.attributes.get::<VecAttr>(&ATTR_KEY_RES_ATTRS)?;
            res_attrs.0.get(res_idx)?.downcast_ref()
        })
        .ok()
    }

    fn get_res_attr<'a, T: Attribute>(
        &self,
        ctx: &'a Context,
        res_idx: usize,
        key: &Identifier,
    ) -> Option<Ref<'a, T>> {
        let res_attrs = self.get_res_attrs(ctx, res_idx)?;
        Ref::filter_map(res_attrs, |res_attrs| {
            res_attrs.lookup(key).and_then(|attr| attr.downcast_ref())
        })
        .ok()
    }

    fn has_res_attr(&self, ctx: &Context, res_idx: usize, key: &Identifier) -> bool {
        self.get_res_attrs(ctx, res_idx)
            .is_some_and(|res_attrs| res_attrs.lookup(key).is_some())
    }

    fn get_res_attrs_mut<'a>(&self, ctx: &'a Context, res_idx: usize) -> RefMut<'a, DictAttr> {
        let self_op = self.get_operation().deref_mut(ctx);
        RefMut::map(self_op, |self_op| {
            get_res_or_init_mut(&mut self_op.attributes, res_idx)
        })
    }

    fn set_arg_attrs(&self, ctx: &Context, arg_idx: usize, dict: DictAttr) {
        *self.get_arg_attrs_mut(ctx, arg_idx) = dict;
    }

    fn set_arg_attr(&self, ctx: &Context, arg_idx: usize, key: &Identifier, value: AttrObj) {
        self.get_arg_attrs_mut(ctx, arg_idx).insert(key, value);
    }

    fn set_arg_attr_unit(&self, ctx: &Context, arg_idx: usize, key: &Identifier) {
        self.get_arg_attrs_mut(ctx, arg_idx)
            .insert(key, Box::new(UnitAttr::new()));
    }

    fn remove_arg_attr(&self, ctx: &Context, arg_idx: usize, key: &Identifier) {
        self.get_arg_attrs_mut(ctx, arg_idx).remove(key);
    }

    fn set_res_attrs(&self, ctx: &Context, res_idx: usize, dict: DictAttr) {
        *self.get_res_attrs_mut(ctx, res_idx) = dict;
    }

    fn set_res_attr(&self, ctx: &Context, res_idx: usize, key: &Identifier, value: AttrObj) {
        self.get_res_attrs_mut(ctx, res_idx).insert(key, value);
    }

    fn remove_res_attr(&self, ctx: &Context, res_idx: usize, key: &Identifier) {
        self.get_res_attrs_mut(ctx, res_idx).remove(key);
    }

    fn set_res_attr_unit(&self, ctx: &Context, res_idx: usize, key: &Identifier) {
        self.get_res_attrs_mut(ctx, res_idx)
            .insert(key, Box::new(UnitAttr::new()));
    }
}

fn get_arg_or_init_mut(dict: &mut AttributeDict, arg_idx: usize) -> &mut DictAttr {
    let args_attrs = attr_get_or_insert_mut(dict, &ATTR_KEY_ARG_ATTRS, || VecAttr::new(vec![]))
        .expect("Should be `VecAttr`");
    vec_get_or_insert_mut(args_attrs, arg_idx, || DictAttr::new(vec![]))
        .expect("Should be `DictAttr`")
}

fn get_res_or_init_mut(dict: &mut AttributeDict, arg_idx: usize) -> &mut DictAttr {
    let res_attrs = attr_get_or_insert_mut(dict, &ATTR_KEY_RES_ATTRS, || VecAttr::new(vec![]))
        .expect("Should be `VecAttr`");
    vec_get_or_insert_mut(res_attrs, arg_idx, || DictAttr::new(vec![]))
        .expect("Should be `DictAttr`")
}

fn attr_get_or_insert_mut<'a, T: Attribute>(
    dict: &'a mut AttributeDict,
    key: &Identifier,
    init: impl FnOnce() -> T,
) -> Option<&'a mut T> {
    if !dict.0.contains_key(key) {
        dict.set(key.clone(), init());
    }
    dict.get_mut(key)
}

fn vec_get_or_insert_mut<T: Attribute>(
    vec: &mut VecAttr,
    idx: usize,
    mut init: impl FnMut() -> T,
) -> Option<&mut T> {
    if vec.0.len() <= idx {
        vec.0.resize_with(idx + 1, || Box::new(init()));
    }
    vec.0[idx].downcast_mut()
}

impl FuncInterface for FuncOp {}
