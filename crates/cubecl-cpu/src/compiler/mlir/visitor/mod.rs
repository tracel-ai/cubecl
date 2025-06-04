pub(super) mod arithmetic;
pub(super) mod block;
pub(super) mod elem;
pub(super) mod instruction;
pub(super) mod item;
pub(super) mod operation;
pub(super) mod operator;
pub(super) mod variable;

use std::collections::HashMap;

use cubecl_core::prelude::KernelDefinition;
use cubecl_opt::Optimizer;
use melior::{
    Context,
    dialect::{arith, func, scf},
    ir::{
        Attribute, Block, BlockLike, BlockRef, Identifier, Location, Operation, Region, RegionLike,
        Type, Value,
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::FunctionType,
    },
};

pub struct Visitor<'a> {
    pub block_stack: Vec<BlockRef<'a, 'a>>,
    pub context: &'a Context,
    pub location: Location<'a>,
    pub current_local_variables: HashMap<u32, Value<'a, 'a>>,
    pub current_version_variables: HashMap<(u32, u16), Value<'a, 'a>>,
    pub global_buffers: Vec<Value<'a, 'a>>,

    pub cube_dim_x: Option<Value<'a, 'a>>,
    pub cube_dim_y: Option<Value<'a, 'a>>,
    pub cube_dim_z: Option<Value<'a, 'a>>,
    pub cube_count_x: Option<Value<'a, 'a>>,
    pub cube_count_y: Option<Value<'a, 'a>>,
    pub cube_count_z: Option<Value<'a, 'a>>,
    pub unit_pos_x: Option<Value<'a, 'a>>,
    pub unit_pos_y: Option<Value<'a, 'a>>,
    pub unit_pos_z: Option<Value<'a, 'a>>,
    pub cube_pos_x: Option<Value<'a, 'a>>,
    pub cube_pos_y: Option<Value<'a, 'a>>,
    pub cube_pos_z: Option<Value<'a, 'a>>,
    pub absolute_pos_x: Option<Value<'a, 'a>>,
    pub absolute_pos_y: Option<Value<'a, 'a>>,
    pub absolute_pos_z: Option<Value<'a, 'a>>,
    pub absolute_pos: Option<Value<'a, 'a>>,
}

impl<'a> Visitor<'a> {
    pub fn new(context: &'a Context, location: Location<'a>) -> Self {
        let current_local_variables = HashMap::new();
        let current_version_variables = HashMap::new();
        let global_buffers = vec![];
        let block_stack = vec![];

        let cube_dim_x = None;
        let cube_dim_y = None;
        let cube_dim_z = None;
        let cube_count_x = None;
        let cube_count_y = None;
        let cube_count_z = None;
        let unit_pos_x = None;
        let unit_pos_y = None;
        let unit_pos_z = None;
        let cube_pos_x = None;
        let cube_pos_y = None;
        let cube_pos_z = None;
        let absolute_pos_x = None;
        let absolute_pos_y = None;
        let absolute_pos_z = None;
        let absolute_pos = None;
        Self {
            block_stack,
            context,
            location,
            current_local_variables,
            current_version_variables,
            global_buffers,
            cube_dim_x,
            cube_dim_y,
            cube_dim_z,
            cube_count_x,
            cube_count_y,
            cube_count_z,
            unit_pos_x,
            unit_pos_y,
            unit_pos_z,
            cube_pos_x,
            cube_pos_y,
            cube_pos_z,
            absolute_pos_x,
            absolute_pos_y,
            absolute_pos_z,
            absolute_pos,
        }
    }

    pub fn block(&self) -> BlockRef<'a, 'a> {
        self.block_stack.last().unwrap().clone()
    }

    pub fn append_operation_with_result(
        &self,
        operation: impl Into<Operation<'a>>,
    ) -> Value<'a, 'a> {
        self.block()
            .append_operation(operation.into())
            .result(0)
            .unwrap()
            .into()
    }

    pub(super) fn visit_kernel<'b: 'a>(
        &'a mut self,
        kernel: &'b KernelDefinition,
        module: &melior::ir::Module<'a>,
        opt: &Optimizer,
    ) {
        let name = StringAttribute::new(self.context, "kernel");

        let attributes = &[(
            Identifier::new(self.context, "llvm.emit_c_interface"),
            Attribute::unit(self.context).into(),
        )];

        let mut inputs = Vec::with_capacity(kernel.buffers.len() + 9);
        let mut block_input = Vec::with_capacity(kernel.buffers.len());

        for binding in kernel.buffers.iter() {
            let memref = self.item_to_memref_buffer_type(binding.item).into();
            inputs.push(memref);
            block_input.push((memref, self.location));
        }

        for _ in 0..9 {
            let index = Type::index(self.context);
            inputs.push(index);
            block_input.push((index, self.location));
        }

        let func_type = TypeAttribute::new(FunctionType::new(self.context, &inputs, &[]).into());

        let location = self.location;
        self.add_external_function_to_module(module);
        module.body().append_operation(func::func(
            self.context,
            name,
            func_type,
            {
                let region = Region::new();
                let block = Block::new(&block_input);
                region.append_block(block);

                let block = region.first_block().unwrap();
                let argument_count = block.argument_count() - 9;
                for i in 0..argument_count {
                    self.global_buffers.push(block.argument(i).unwrap().into());
                }

                self.insert_builtin_loop(block, opt);

                block.append_operation(func::r#return(&[], location));

                region
            },
            attributes,
            location,
        ));
    }

    // TODO: cleanup this abomination by refactoring melior to make it at least not as bulky and verbose
    pub fn insert_builtin_loop(&mut self, block: BlockRef<'a, 'a>, opt: &Optimizer) {
        let basic_block_id = opt.entry();
        let c0: Value<'_, '_> = block
            .append_operation(arith::constant(
                self.context,
                IntegerAttribute::new(Type::index(self.context), 0).into(),
                self.location,
            ))
            .result(0)
            .unwrap()
            .into();

        let c1: Value<'_, '_> = block
            .append_operation(arith::constant(
                self.context,
                IntegerAttribute::new(Type::index(self.context), 1).into(),
                self.location,
            ))
            .result(0)
            .unwrap()
            .into();

        let to_assign = [
            &mut self.cube_dim_x,
            &mut self.cube_dim_y,
            &mut self.cube_dim_z,
            &mut self.cube_count_x,
            &mut self.cube_count_y,
            &mut self.cube_count_z,
            &mut self.unit_pos_x,
            &mut self.unit_pos_y,
            &mut self.unit_pos_z,
        ];

        for (i, v) in to_assign.into_iter().enumerate() {
            *v = Some(
                block
                    .argument(self.global_buffers.len() + i)
                    .unwrap()
                    .into(),
            );
        }

        let absolute_pos_tmp_0: Value<'a, 'a> = block
            .append_operation(arith::muli(
                self.cube_count_x.unwrap(),
                self.cube_dim_x.unwrap(),
                self.location,
            ))
            .result(0)
            .unwrap()
            .into();

        let absolute_pos_tmp_1: Value<'a, 'a> = block
            .append_operation(arith::muli(
                self.cube_count_y.unwrap(),
                self.cube_dim_y.unwrap(),
                self.location,
            ))
            .result(0)
            .unwrap()
            .into();

        let absolute_pos_tmp_2: Value<'a, 'a> = block
            .append_operation(arith::muli(
                absolute_pos_tmp_0,
                absolute_pos_tmp_1,
                self.location,
            ))
            .result(0)
            .unwrap()
            .into();

        block.append_operation(
            scf::r#for(
                c0,
                self.cube_count_x.unwrap(),
                c1,
                {
                    let region = Region::new();
                    let block = Block::new(&[(Type::index(&self.context), self.location)]);
                    self.cube_pos_x = Some(block.argument(0).unwrap().into());

                    let absolute_pos_x_tmp = block
                        .append_operation(arith::muli(
                            self.cube_pos_x.unwrap(),
                            self.cube_dim_x.unwrap(),
                            self.location,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    self.absolute_pos_x = Some(
                        block
                            .append_operation(arith::addi(
                                absolute_pos_x_tmp,
                                self.unit_pos_x.unwrap(),
                                self.location,
                            ))
                            .result(0)
                            .unwrap()
                            .into(),
                    );

                    block.append_operation(scf::r#for(
                        c0,
                        self.cube_count_y.unwrap(),
                        c1,
                        {
                            let region = Region::new();
                            let block = Block::new(&[(Type::index(&self.context), self.location)]);
                            self.cube_pos_y = Some(block.argument(0).unwrap().into());

                            let absolute_pos_y_tmp = block
                                .append_operation(arith::muli(
                                    self.cube_pos_y.unwrap(),
                                    self.cube_dim_y.unwrap(),
                                    self.location,
                                ))
                                .result(0)
                                .unwrap()
                                .into();
                            self.absolute_pos_y = Some(
                                block
                                    .append_operation(arith::addi(
                                        absolute_pos_y_tmp,
                                        self.unit_pos_y.unwrap(),
                                        self.location,
                                    ))
                                    .result(0)
                                    .unwrap()
                                    .into(),
                            );

                            let absolute_pos_tmp_3 = block
                                .append_operation(arith::muli(
                                    self.absolute_pos_y.unwrap(),
                                    absolute_pos_tmp_0,
                                    self.location,
                                ))
                                .result(0)
                                .unwrap()
                                .into();

                            block.append_operation(scf::r#for(
                                c0,
                                self.cube_count_y.unwrap(),
                                c1,
                                {
                                    let region = Region::new();
                                    let block =
                                        Block::new(&[(Type::index(&self.context), self.location)]);

                                    self.cube_pos_z = Some(block.argument(0).unwrap().into());

                                    let absolute_pos_z_tmp = block
                                        .append_operation(arith::muli(
                                            self.cube_pos_z.unwrap(),
                                            self.cube_dim_z.unwrap(),
                                            self.location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();
                                    self.absolute_pos_z = Some(
                                        block
                                            .append_operation(arith::addi(
                                                absolute_pos_z_tmp,
                                                self.unit_pos_z.unwrap(),
                                                self.location,
                                            ))
                                            .result(0)
                                            .unwrap()
                                            .into(),
                                    );

                                    let absolute_pos_tmp_4 = block
                                        .append_operation(arith::muli(
                                            self.absolute_pos_z.unwrap(),
                                            absolute_pos_tmp_2,
                                            self.location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();
                                    let absolute_pos_tmp_5 = block
                                        .append_operation(arith::addi(
                                            self.absolute_pos_x.unwrap(),
                                            absolute_pos_tmp_3,
                                            self.location,
                                        ))
                                        .result(0)
                                        .unwrap()
                                        .into();

                                    self.absolute_pos = Some(
                                        block
                                            .append_operation(arith::addi(
                                                absolute_pos_tmp_5,
                                                absolute_pos_tmp_4,
                                                self.location,
                                            ))
                                            .result(0)
                                            .unwrap()
                                            .into(),
                                    );

                                    region.append_block(block);
                                    let block = region.first_block().unwrap();
                                    self.block_stack.push(block);
                                    self.visit_basic_block(basic_block_id, opt);
                                    self.block_stack.pop();
                                    block.append_operation(scf::r#yield(&[], self.location).into());
                                    region
                                },
                                self.location,
                            ));
                            block.append_operation(scf::r#yield(&[], self.location).into());
                            region.append_block(block);
                            region
                        },
                        self.location,
                    ));

                    block.append_operation(scf::r#yield(&[], self.location).into());
                    region.append_block(block);
                    region
                },
                self.location,
            )
            .into(),
        );
    }
}
